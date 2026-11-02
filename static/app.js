(() => {
  const $ = (id) => document.getElementById(id);

  const els = {
    chip: $("modelChip"), banner: $("demoBanner"),
    dropzone: $("dropzone"), file: $("fileInput"),
    examples: $("examples"), examplesEmpty: $("examplesEmpty"),
    original: $("originalViewer"), mask: $("maskViewer"),
    overlay: $("overlayViewer"), crop: $("cropViewer"),
    verdict: $("verdict"), prob: $("probability"),
    ratio: $("ratio"), ratioBar: $("ratioBar"),
    notice: $("notice"), noticeMain: $("noticeMain"), noticeSub: $("noticeSub"),
    meta: $("meta"), loading: $("loading"), toast: $("toast"),
  };

  const MODEL_NAMES = {
    patch_topk: "PatchGuard (Top-k)", patch_avg: "PatchGuard (Average)",
    global_cls: "Global (CLS)", global_cls_mean: "Global (CLS+Mean)",
    segformer: "SegFormer-B0", demo: "Demo",
  };

  // ---------------------------------------------------------------- helpers
  function toast(msg) {
    els.toast.textContent = msg;
    els.toast.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => (els.toast.hidden = true), 3500);
  }

  function placeholder(viewer, text) {
    viewer.innerHTML = `<div class="placeholder">${text}</div>`;
  }

  /** 이미지 + 점선 박스(정규화 좌표 [x0,y0,x1,y1]) */
  function showImage(viewer, src, boxes = []) {
    viewer.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "img-wrap";
    const img = new Image();
    img.src = src;
    img.alt = "";
    wrap.appendChild(img);
    for (const [x0, y0, x1, y1] of boxes) {
      const b = document.createElement("div");
      b.className = "box";
      Object.assign(b.style, {
        left: `${x0 * 100}%`, top: `${y0 * 100}%`,
        width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%`,
      });
      wrap.appendChild(b);
    }
    viewer.appendChild(wrap);
  }

  function fmtPct(v) {
    if (v === 0) return "0%";
    if (v < 0.1) return "<0.1%";
    return `${Number(v.toFixed(1))}%`;
  }

  // ---------------------------------------------------------------- render
  function resetResult() {
    els.verdict.className = "badge badge-empty";
    els.verdict.textContent = "—";
    els.prob.textContent = "—";
    els.ratio.textContent = "—";
    els.ratioBar.style.width = "0";
    els.ratioBar.classList.remove("on");
  }

  function render(r) {
    // 판단 결과
    els.verdict.className = `badge ${r.is_edited ? "badge-edited" : "badge-real"}`;
    els.verdict.textContent = r.label;
    els.prob.textContent = `${r.probability}%`;

    const boxes = r.is_edited && r.has_mask ? r.regions || [] : [];
    showImage(els.original, r.image, boxes);

    els.notice.classList.toggle("real", !r.is_edited);

    if (!r.has_mask) {
      els.ratio.textContent = "—";
      els.ratioBar.style.width = "0";
      els.ratioBar.classList.remove("on");
      ["mask", "overlay", "crop"].forEach((k) =>
        placeholder(els[k], "이 모델은 위치 예측(마스크)을 지원하지 않습니다"));
      els.noticeMain.textContent = r.is_edited
        ? "AI로 편집되었을 가능성이 높습니다."
        : "AI 편집 흔적이 발견되지 않았습니다.";
      els.noticeSub.textContent = "현재 모델은 이미지 단위 판정만 제공합니다.";
    } else {
      els.ratio.textContent = fmtPct(r.edit_ratio);
      els.ratioBar.style.width = `${Math.min(100, r.edit_ratio)}%`;
      els.ratioBar.classList.toggle("on", r.edit_ratio > 0);

      showImage(els.mask, r.mask, boxes);
      showImage(els.overlay, r.overlay, boxes);

      if (r.crop && r.is_edited) {
        els.crop.innerHTML = "";
        const frame = document.createElement("div");
        frame.className = "img-wrap crop-frame";
        const img = new Image();
        img.src = r.crop;
        img.alt = "편집 영역 확대";
        frame.appendChild(img);
        els.crop.appendChild(frame);
      } else {
        placeholder(els.crop, "추출할 편집 영역이 없습니다");
      }

      if (r.is_edited) {
        els.noticeMain.textContent = r.edit_ratio > 0
          ? `전체 이미지의 약 ${fmtPct(r.edit_ratio)} 영역이 AI로 편집되었을 가능성이 높습니다.`
          : "AI로 편집되었을 가능성이 높지만, 임계값 이상의 편집 영역은 찾지 못했습니다.";
        els.noticeSub.textContent = "작은 영역의 편집도 탐지할 수 있습니다.";
      } else {
        els.noticeMain.textContent = "AI 편집 흔적이 발견되지 않았습니다.";
        els.noticeSub.textContent =
          `편집 확률이 판정 기준(${Math.round(r.cls_threshold * 100)}%)보다 낮습니다.`;
      }
    }

    els.meta.textContent = `${r.filename} · ${r.latency_ms} ms`;
  }

  // ---------------------------------------------------------------- requests
  async function predict(body, isForm) {
    els.loading.hidden = false;
    try {
      const res = await fetch("/api/predict", isForm
        ? { method: "POST", body }
        : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const data = await res.json().catch(() => ({ error: `서버 오류 (${res.status})` }));
      if (!res.ok || data.error) throw new Error(data.error || `서버 오류 (${res.status})`);
      render(data);
    } catch (e) {
      toast(e.message);
    } finally {
      els.loading.hidden = true;
    }
  }

  function setActiveExample(name) {
    els.examples.querySelectorAll(".example").forEach((b) =>
      b.classList.toggle("active", b.dataset.name === name));
  }

  function handleFile(file) {
    if (!file) return;
    if (!/^image\//.test(file.type)) return toast("이미지 파일(JPG/PNG)만 올릴 수 있습니다.");
    setActiveExample(null);
    resetResult();
    // 업로드 직후 원본을 먼저 보여준다
    showImage(els.original, URL.createObjectURL(file));
    const fd = new FormData();
    fd.append("image", file);
    predict(fd, true);
  }

  // ---------------------------------------------------------------- events
  els.file.addEventListener("change", (e) => { handleFile(e.target.files[0]); e.target.value = ""; });
  els.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.file.click(); }
  });
  ["dragenter", "dragover"].forEach((t) => els.dropzone.addEventListener(t, (e) => {
    e.preventDefault(); els.dropzone.classList.add("drag");
  }));
  ["dragleave", "drop"].forEach((t) => els.dropzone.addEventListener(t, (e) => {
    e.preventDefault(); els.dropzone.classList.remove("drag");
  }));
  els.dropzone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
  // 페이지 어디에 떨어뜨려도 받기
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    if (!els.dropzone.contains(e.target)) handleFile(e.dataTransfer.files[0]);
  });

  // ---------------------------------------------------------------- init
  async function init() {
    try {
      const info = await (await fetch("/api/info")).json();
      els.banner.hidden = info.mode !== "demo";
      els.chip.innerHTML = info.mode === "demo"
        ? "<b>데모 모드</b>"
        : `<b>${MODEL_NAMES[info.model] || info.model}</b> · ${info.device}` +
          (info.checkpoint ? ` · ${info.checkpoint}` : "");
    } catch {
      els.chip.textContent = "서버 연결 실패";
    }

    try {
      const list = await (await fetch("/api/examples")).json();
      els.examplesEmpty.hidden = list.length > 0;
      list.forEach((ex, i) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "example";
        b.dataset.name = ex.name;
        b.title = ex.name;
        b.innerHTML = `<img src="${ex.url}" alt="${ex.name}" loading="lazy">`;
        b.addEventListener("click", () => {
          setActiveExample(ex.name);
          resetResult();
          showImage(els.original, ex.url);
          predict({ example: ex.name }, false);
        });
        els.examples.appendChild(b);
      });
      // 첫 예시를 자동으로 분석
      if (list.length) els.examples.querySelector(".example").click();
    } catch {
      els.examplesEmpty.hidden = false;
    }
  }

  init();
})();
