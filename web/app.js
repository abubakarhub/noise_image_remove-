/* Denoise Lab front end — talks to the FastAPI backend in server.py */
(() => {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const FILTER_COLORS = { mean: "#8d99a8", median: "#ffb547", gaussian: "#4fd1c5", bilateral: "#b18cff" };

  // Edit these statuses as the project moves on: done | active | todo
  const TIMELINE = [
    { phase: 1, work: "Literature survey and problem understanding", from: 1, to: 2, status: "done" },
    { phase: 2, work: "Study of image noise and filtering techniques", from: 3, to: 3, status: "done" },
    { phase: 3, work: "Dataset / image collection and pre-processing", from: 4, to: 4, status: "done" },
    { phase: 4, work: "Implementation of noise models", from: 5, to: 5, status: "done" },
    { phase: 5, work: "Implementation of different filters", from: 6, to: 7, status: "done" },
    { phase: 6, work: "Performance evaluation and comparison", from: 8, to: 8, status: "done" },
    { phase: 7, work: "User interface / final integration", from: 9, to: 9, status: "active" },
    { phase: 8, work: "Testing and documentation", from: 10, to: 10, status: "todo" },
    { phase: 9, work: "Final report and presentation", from: 11, to: 12, status: "todo" },
  ];

  const state = {
    cfg: null, sample: "cameraman", file: null,
    noise: "gaussian", level: 25, seed: 0,
    filter: "median", ksize: 3, sigma: 0, sigmaColor: 75, sigmaSpace: 75,
    view: "slider", vsOrig: false, metric: "ssim",
    last: null, compare: null,
  };

  // ------------------------------------------------------------ utils
  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => t.classList.remove("show"), 3200);
  }

  async function api(path, fd, signal) {
    const res = await fetch(path, { method: fd ? "POST" : "GET", body: fd, signal });
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).detail || msg; } catch (e) {}
      throw new Error(msg);
    }
    return res.json();
  }

  function formData(extra = {}) {
    const fd = new FormData();
    if (state.file) fd.append("file", state.file);
    else fd.append("sample", state.sample);
    const fields = {
      noise: state.noise, level: state.level, seed: state.seed, filter: state.filter, ksize: state.ksize,
      sigma: state.sigma, sigma_color: state.sigmaColor, sigma_space: state.sigmaSpace, ...extra,
    };
    for (const [k, v] of Object.entries(fields)) fd.append(k, v);
    return fd;
  }

  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

  const fmt = {
    mse: (v) => (v >= 1000 ? v.toFixed(0) : v.toFixed(1)),
    psnr: (v) => v.toFixed(2),
    ssim: (v) => v.toFixed(4),
  };
  const noiseName = (k) => (k === "none" ? "Little / no noise" : state.cfg.noises[k].name);
  const filterName = (k) => state.cfg.filters[k] || state.cfg.enhanced[k] || k;
  const isEnhanced = (k) => k in state.cfg.enhanced;

  function countUp(el, to, format, dur = 650) {
    const from = parseFloat(el.dataset.val ?? "NaN");
    el.dataset.val = to;
    if (reduced.matches || !isFinite(from)) { el.textContent = format(to); return; }
    const t0 = performance.now();
    cancelAnimationFrame(el._raf);
    const step = (now) => {
      const p = Math.min(1, (now - t0) / dur);
      const e = 1 - Math.pow(1 - p, 3);
      el.textContent = format(from + (to - from) * e);
      if (p < 1) el._raf = requestAnimationFrame(step);
    };
    el._raf = requestAnimationFrame(step);
  }

  function swapImg(img, src) {
    if (img.getAttribute("src") === src) return;
    img.src = src;
    img.classList.remove("img-swap");
    void img.offsetWidth;
    img.classList.add("img-swap");
  }

  function setRangeFill(input) {
    const p = ((input.value - input.min) / (input.max - input.min)) * 100;
    input.style.setProperty("--p", p + "%");
  }

  function dataURLtoBlob(url) {
    const [head, b64] = url.split(",");
    const mime = head.match(/:(.*?);/)[1];
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type: mime });
  }

  // ------------------------------------------------------------ segmented control
  function buildSeg(root, items, current, onChange) {
    root.innerHTML = "";
    const ind = document.createElement("span");
    ind.className = "ind";
    root.append(ind);
    for (const [key, label] of items) {
      const b = document.createElement("button");
      b.type = "button";
      b.setAttribute("role", "radio");
      b.dataset.key = key;
      b.textContent = label;
      b.setAttribute("aria-checked", String(key === current));
      b.tabIndex = key === current ? 0 : -1;
      b.addEventListener("click", () => { setSeg(root, key); onChange(key); });
      root.append(b);
    }
    root.addEventListener("keydown", (e) => {
      const btns = $$("button", root);
      const i = btns.findIndex((b) => b.getAttribute("aria-checked") === "true");
      let j = null;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") j = (i + 1) % btns.length;
      if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = (i - 1 + btns.length) % btns.length;
      if (j === null) return;
      e.preventDefault();
      btns[j].focus();
      btns[j].click();
    });
    requestAnimationFrame(() => moveInd(root));
  }

  function setSeg(root, key) {
    for (const b of $$("button", root)) {
      const on = b.dataset.key === key || b.dataset.m === key;
      b.setAttribute("aria-checked", String(on));
      b.tabIndex = on ? 0 : -1;
    }
    moveInd(root);
  }

  function moveInd(root) {
    const ind = $(".ind", root);
    const on = $('button[aria-checked="true"]', root);
    if (!ind || !on) return;
    ind.style.width = on.offsetWidth + "px";
    ind.style.height = on.offsetHeight + "px";
    ind.style.transform = `translate(${on.offsetLeft}px, ${on.offsetTop}px)`;
  }
  addEventListener("resize", debounce(() => $$(".seg").forEach(moveInd), 100));

  // ------------------------------------------------------------ before/after slider
  function initCompare(root) {
    const handle = $(".handle", root);
    let pos = 50, target = 50, raf = 0;
    const apply = (p) => {
      root.style.setProperty("--pos", p + "%");
      handle.setAttribute("aria-valuenow", Math.round(p));
    };
    const animate = () => {
      // critically-damped follow so the handle feels physical, not glued
      pos += (target - pos) * 0.35;
      if (Math.abs(target - pos) < 0.05) pos = target;
      apply(pos);
      if (pos !== target) raf = requestAnimationFrame(animate);
      else raf = 0;
    };
    const go = (p, instant) => {
      target = Math.max(0, Math.min(100, p));
      if (instant || reduced.matches) { pos = target; apply(pos); return; }
      if (!raf) raf = requestAnimationFrame(animate);
    };
    const fromEvent = (e) => {
      const r = root.getBoundingClientRect();
      return ((e.clientX - r.left) / r.width) * 100;
    };
    root.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      root.setPointerCapture(e.pointerId);
      root.classList.add("is-drag");
      go(fromEvent(e));
    });
    root.addEventListener("pointermove", (e) => { if (root.classList.contains("is-drag")) go(fromEvent(e), true); });
    const end = () => root.classList.remove("is-drag");
    root.addEventListener("pointerup", end);
    root.addEventListener("pointercancel", end);
    handle.addEventListener("keydown", (e) => {
      const step = e.shiftKey ? 10 : 2;
      if (e.key === "ArrowLeft") { e.preventDefault(); go(target - step); }
      if (e.key === "ArrowRight") { e.preventDefault(); go(target + step); }
      if (e.key === "Home") { e.preventDefault(); go(0); }
      if (e.key === "End") { e.preventDefault(); go(100); }
    });
    apply(50);
    return { go };
  }

  // ------------------------------------------------------------ pipeline indicator
  const pipe = {
    timers: [],
    start() {
      this.timers.forEach(clearTimeout);
      const steps = $$("#pipeline li");
      steps.forEach((li) => li.classList.remove("is-on", "is-busy"));
      steps.forEach((li, i) => {
        if (i > 4) return;
        this.timers.push(setTimeout(() => {
          steps.forEach((s) => s.classList.remove("is-busy"));
          for (let k = 0; k < i; k++) steps[k].classList.add("is-on");
          li.classList.add("is-busy");
        }, reduced.matches ? 0 : i * 110));
      });
    },
    done() {
      this.timers.forEach(clearTimeout);
      $$("#pipeline li").forEach((li, i) => {
        li.classList.remove("is-busy");
        setTimeout(() => li.classList.add("is-on"), reduced.matches ? 0 : i * 45);
      });
    },
    fail() {
      this.timers.forEach(clearTimeout);
      $$("#pipeline li").forEach((li) => li.classList.remove("is-busy"));
    },
  };

  // ------------------------------------------------------------ lab
  let procCtrl = null;
  async function runProcess() {
    procCtrl?.abort();
    procCtrl = new AbortController();
    const stage = $("#stage");
    stage.classList.add("is-busy");
    stage.setAttribute("aria-busy", "true");
    pipe.start();
    const t0 = performance.now();
    try {
      const r = await api("/api/process", formData(), procCtrl.signal);
      state.last = r;
      renderProcess(r, performance.now() - t0);
      pipe.done();
    } catch (e) {
      if (e.name === "AbortError") return;
      pipe.fail();
      toast("Processing failed: " + e.message);
    } finally {
      stage.classList.remove("is-busy");
      stage.setAttribute("aria-busy", "false");
    }
  }
  const runSoon = debounce(runProcess, 220);

  function renderProcess(r, ms) {
    const im = r.images;
    swapImg($("#cmpA"), state.vsOrig ? im.original : im.noisy);
    swapImg($("#cmpB"), im.denoised);
    swapImg($("#gOrig"), im.original);
    swapImg($("#gNoisy"), im.noisy);
    swapImg($("#gOut"), im.denoised);
    swapImg($("#resImg"), im.residual);

    const src = state.file ? state.file.name : state.cfg.samples[state.sample].split(" - ")[0];
    $("#imgMeta").textContent =
      `${src} · ${r.size[0]}×${r.size[1]} ${r.color ? "colour" : "grayscale"} · ${noiseName(state.noise)} → ` +
      `${r.method ? r.method.label : filterName(state.filter)}${r.method && r.method.strength ? " (" + r.method.strength + ")" : ""} · round trip ${Math.round(ms)} ms`;

    for (const card of $$(".metric")) {
      const key = card.dataset.key;
      const n = r.metrics.noisy[key], d = r.metrics.denoised[key];
      countUp($('[data-v="denoised"]', card), d, fmt[key]);
      countUp($('[data-v="noisy"]', card), n, fmt[key]);
      const better = key === "mse" ? d < n : d > n;
      const delta = $(".delta", card);
      let txt;
      if (key === "mse") txt = (n > 0 ? ((d - n) / n) * 100 : 0).toFixed(0) + "%";
      else if (key === "psnr") txt = (d - n >= 0 ? "+" : "") + (d - n).toFixed(2) + " dB";
      else txt = (d - n >= 0 ? "+" : "") + (d - n).toFixed(3);
      delta.textContent = txt;
      delta.className = "delta " + (better ? "up" : "down");
      let w = key === "ssim" ? d : key === "psnr" ? d / 50 : Math.max(0, 1 - d / Math.max(n, 1e-9));
      $(".bar i", card).style.transform = `scaleX(${Math.max(0, Math.min(1, w))})`;
    }

    drawHistogram(r.histograms);
    renderAnalysis(r.analysis);
  }

  // histogram with tweened paths
  let histPrev = null;
  function drawHistogram(h) {
    const svg = $("#hist");
    const keys = ["original", "noisy", "denoised"];
    const colors = { original: "var(--orig)", noisy: "var(--noisy)", denoised: "var(--clean)" };
    if (!svg.childElementCount) {
      for (const k of keys) {
        svg.insertAdjacentHTML("beforeend",
          `<path class="area" data-k="${k}" fill="${colors[k]}"></path><path class="ln" data-k="${k}" stroke="${colors[k]}"></path>`);
      }
    }
    const max = Math.max(...keys.flatMap((k) => h[k])) || 1;
    const W = 400, H = 180;
    const toPath = (arr) => {
      const n = arr.length;
      let d = "";
      arr.forEach((v, i) => {
        const x = (i / (n - 1)) * W;
        const y = H - 4 - (Math.sqrt(v / max)) * (H - 12); // sqrt scale keeps low bins visible
        d += (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
      });
      return d;
    };
    const target = Object.fromEntries(keys.map((k) => [k, h[k]]));
    const from = histPrev;
    histPrev = target;
    const paint = (data) => {
      for (const k of keys) {
        const d = toPath(data[k]);
        $(`path.ln[data-k="${k}"]`, svg).setAttribute("d", d);
        $(`path.area[data-k="${k}"]`, svg).setAttribute("d", d + `L${W} ${H}L0 ${H}Z`);
      }
    };
    if (!from || reduced.matches) { paint(target); return; }
    const t0 = performance.now(), dur = 450;
    cancelAnimationFrame(drawHistogram._raf);
    const step = (now) => {
      const p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      const mid = {};
      for (const k of keys) mid[k] = target[k].map((v, i) => from[k][i] + (v - from[k][i]) * e);
      paint(mid);
      if (p < 1) drawHistogram._raf = requestAnimationFrame(step);
    };
    drawHistogram._raf = requestAnimationFrame(step);
  }

  function renderAnalysis(a) {
    $("#detType").textContent = noiseName(a.type);
    $("#detConf").textContent = `${Math.round(a.confidence * 100)}% confidence`;
    const chip = $("#detMatch");
    const ok = a.type === state.noise;
    chip.textContent = ok ? "✓ matches the added noise" : `added: ${noiseName(state.noise)}`;
    chip.className = "chip " + (ok ? "ok" : "warn");
    renderScores($("#detScores"), a.scores);
    $("#detSigma").textContent = a.sigma.toFixed(1);
    $("#detImpulse").textContent = (a.impulse_ratio * 100).toFixed(2) + "%";
    $("#detSlope").textContent = a.slope == null ? "n/a" : a.slope.toFixed(2);
    const rec = a.recommendation, enh = a.enhanced;
    $("#detRec").innerHTML = "";
    const s = document.createElement("strong");
    const est = Object.entries(a.params || {}).map(([k, v]) => `${k} ≈ ${v}`).join(", ");
    s.textContent = enh.filter === "none"
      ? "Recommended: leave it as it is. "
      : `Recommended: ${filterName(enh.filter)}${est ? " (estimated " + est + ")" : ""}. `;
    const classic = document.createElement("span");
    classic.className = "muted";
    classic.textContent = ` Best classic filter: ${filterName(rec.filter)} ${rec.ksize}×${rec.ksize}.`;
    $("#detRec").append(s, document.createTextNode(enh.why), classic);
  }

  function renderScores(ul, scores) {
    const entries = Object.entries(scores);
    const top = entries.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
    if (!ul.childElementCount) {
      for (const [k] of entries) {
        ul.insertAdjacentHTML("beforeend",
          `<li data-k="${k}"><span>${noiseName(k)}</span><span class="track"><i style="transform:scaleX(0)"></i></span><b>0%</b></li>`);
      }
    }
    for (const [k, v] of entries) {
      const li = $(`li[data-k="${k}"]`, ul);
      li.classList.toggle("top", k === top && v > 0);
      $("i", li).style.transform = `scaleX(${v})`;
      $("b", li).textContent = Math.round(v * 100) + "%";
    }
  }

  // ------------------------------------------------------------ controls
  function syncNoiseSlider() {
    const p = state.cfg.noises[state.noise];
    const r = $("#level");
    r.min = p.min; r.max = p.max; r.step = p.step;
    state.level = p.default;
    r.value = p.default;
    $("#levelLabel").textContent = p.label;
    $("#levelOut").textContent = p.default;
    setRangeFill(r);
  }

  function syncFilterFields() {
    for (const f of $$(".field[data-for]")) {
      const show = f.dataset.for === state.filter;
      if (show && f.hidden) { f.classList.remove("field-enter"); void f.offsetWidth; f.classList.add("field-enter"); }
      f.hidden = !show;
    }
    $("#ksizeLabel").textContent = state.filter === "bilateral" ? "Diameter d" : "Kernel size";
    const kf = $("#ksizeField");
    if (kf) kf.hidden = isEnhanced(state.filter);
    const note = $("#enhNote");
    if (note) note.hidden = !isEnhanced(state.filter);
  }

  function bindRange(id, key, outId, parse = parseFloat, fmtOut = (v) => v) {
    const r = $("#" + id);
    const out = $("#" + outId);
    const update = () => {
      state[key] = parse(r.value);
      out.textContent = fmtOut(state[key]);
      setRangeFill(r);
    };
    r.addEventListener("input", () => { update(); runSoon(); });
    update();
  }

  function selectSample(name) {
    state.sample = name;
    state.file = null;
    $("#drop").classList.remove("has-file");
    $("#dropText").innerHTML = "Drop your own image or <u>browse</u>";
    $$(".thumb").forEach((t) => t.setAttribute("aria-pressed", String(t.dataset.name === name)));
    runProcess();
  }

  function useFile(file) {
    if (!file || !file.type.startsWith("image/")) { toast("Please choose an image file."); return; }
    state.file = file;
    $$(".thumb").forEach((t) => t.setAttribute("aria-pressed", "false"));
    $("#drop").classList.add("has-file");
    $("#dropText").textContent = file.name;
    runProcess();
  }

  function wireDrop(label, input, onFile) {
    input.addEventListener("change", () => input.files[0] && onFile(input.files[0]));
    ["dragenter", "dragover"].forEach((ev) => label.addEventListener(ev, (e) => { e.preventDefault(); label.classList.add("is-over"); }));
    ["dragleave", "drop"].forEach((ev) => label.addEventListener(ev, (e) => { e.preventDefault(); label.classList.remove("is-over"); }));
    label.addEventListener("drop", (e) => e.dataTransfer.files[0] && onFile(e.dataTransfer.files[0]));
  }

  function buildControls() {
    const cfg = state.cfg;
    const thumbs = $("#thumbs");
    for (const [name, desc] of Object.entries(cfg.samples)) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "thumb";
      b.dataset.name = name;
      b.title = desc;
      b.setAttribute("aria-label", desc);
      b.setAttribute("aria-pressed", String(name === state.sample));
      b.innerHTML = `<img src="/api/sample/${name}" alt="" loading="lazy">`;
      b.addEventListener("click", () => selectSample(name));
      thumbs.append(b);
    }
    wireDrop($("#drop"), $("#fileInput"), useFile);

    buildSeg($("#noiseSeg"), Object.entries(cfg.noises).map(([k, v]) => [k, v.name]), state.noise, (k) => {
      state.noise = k;
      syncNoiseSlider();
      runProcess();
    });
    buildSeg($("#filterSeg"), [...Object.entries(cfg.filters), ...Object.entries(cfg.enhanced)], state.filter, (k) => {
      state.filter = k;
      syncFilterFields();
      runProcess();
    });
    // label the second group: methods that tune themselves from the noise analysis
    const firstEnh = $(`#filterSeg button[data-key="${Object.keys(cfg.enhanced)[0]}"]`);
    if (firstEnh) {
      const sep = document.createElement("span");
      sep.className = "seg-sep";
      sep.textContent = "Noise-aware · self-tuning";
      firstEnh.before(sep);
      Object.keys(cfg.enhanced).forEach((k) => $(`#filterSeg button[data-key="${k}"]`).classList.add("enh"));
    }
    syncNoiseSlider();
    syncFilterFields();

    $("#level").addEventListener("input", (e) => {
      state.level = parseFloat(e.target.value);
      $("#levelOut").textContent = state.level;
      setRangeFill(e.target);
      runSoon();
    });
    bindRange("ksize", "ksize", "ksizeOut", (v) => parseInt(v, 10), (v) => `${v}×${v}`);
    bindRange("sigma", "sigma", "sigmaOut", parseFloat, (v) => (v === 0 ? "auto" : v.toFixed(1)));
    bindRange("sigmaColor", "sigmaColor", "sigmaColorOut");
    bindRange("sigmaSpace", "sigmaSpace", "sigmaSpaceOut");

    $("#seed").addEventListener("change", (e) => { state.seed = Math.max(0, parseInt(e.target.value, 10) || 0); runProcess(); });
    $("#reseed").addEventListener("click", () => {
      state.seed = Math.floor(Math.random() * 99999);
      $("#seed").value = state.seed;
      runProcess();
    });

    $("#useRec").addEventListener("click", () => {
      if (!state.last?.analysis) return;
      state.filter = "auto";
      setSeg($("#filterSeg"), "auto");
      syncFilterFields();
      runProcess();
      toast("Applied Auto: detect → estimate → matching noise-aware filter");
    });

    // view tabs
    for (const tab of $$("#viewTabs button")) {
      tab.addEventListener("click", () => {
        state.view = tab.dataset.view;
        $$("#viewTabs button").forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
        $$(".stage .view").forEach((v) => v.classList.toggle("is-active", v.dataset.view === state.view));
      });
    }
    $("#vsOriginal").addEventListener("change", (e) => {
      state.vsOrig = e.target.checked;
      $("#compare").classList.toggle("vs-orig", state.vsOrig);
      $("#cmpLabelA").textContent = state.vsOrig ? "Original" : "Noisy";
      if (state.last) swapImg($("#cmpA"), state.vsOrig ? state.last.images.original : state.last.images.noisy);
    });
    initCompare($("#compare"));
  }

  // ------------------------------------------------------------ detect (no-reference)
  const DETECT_SAMPLES = [
    ["moon", "speckle", 0.12, "Speckle · moon"],
    ["coffee", "poisson", 12, "Poisson · coffee"],
    ["coins", "salt_pepper", 0.1, "Salt & pepper · coins"],
    ["astronaut", "gaussian", 30, "Gaussian · astronaut"],
  ];

  async function runDetect(blob, label) {
    const fd = new FormData();
    fd.append("file", blob, label || "image.png");
    const out = $(".detect-out");
    out.style.opacity = ".6";
    try {
      const r = await api("/api/detect", fd);
      const a = r.analysis;
      swapImg($("#d2A"), r.images.noisy);
      swapImg($("#d2B"), r.images.denoised);
      $("#d2Empty").classList.add("gone");
      $("#d2Type").textContent = noiseName(a.type);
      $("#d2Conf").textContent = Math.round(a.confidence * 100) + "%";
      $("#d2Filter").textContent = r.method.label.replace("Auto → ", "") + (r.method.strength ? ` (${r.method.strength})` : "");
      $("#d2Sigma").textContent = `${a.sigma.toFixed(1)} → ${r.sigma_after.toFixed(1)}`;
      $("#d2Why").textContent = r.method.why;
      drawLogLog(a);
    } catch (e) {
      toast("Detection failed: " + e.message);
    } finally {
      out.style.opacity = "";
    }
  }

  function drawLogLog(a) {
    const svg = $("#loglog");
    const W = 360, H = 220, P = { l: 40, r: 12, t: 12, b: 28 };
    const pts = a.variance_curve.filter(([m, v]) => m > 0 && v > 0).map(([m, v]) => [Math.log10(m), Math.log10(v)]);
    svg.innerHTML = "";
    if (pts.length < 2) {
      svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle">Not enough mid-tone area to plot</text>`;
      $("#loglogNote").textContent = a.type === "salt_pepper"
        ? "Salt-and-pepper was identified from isolated 0/255 impulse pixels."
        : "This image has too little mid-tone area to measure intensity dependence.";
      return;
    }
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const x0 = Math.min(...xs) - 0.05, x1 = Math.max(...xs) + 0.05;
    const yc = ys.reduce((s, y) => s + y, 0) / ys.length, xc = xs.reduce((s, x) => s + x, 0) / xs.length;
    const span = Math.max(Math.max(...ys) - Math.min(...ys), 2 * (x1 - x0)) / 2 + 0.2;
    const y0 = yc - span, y1 = yc + span;
    const X = (x) => P.l + ((x - x0) / (x1 - x0)) * (W - P.l - P.r);
    const Y = (y) => H - P.b - ((y - y0) / (y1 - y0)) * (H - P.t - P.b);
    let g = "";
    for (let i = 0; i <= 4; i++) {
      const yy = y0 + ((y1 - y0) * i) / 4;
      g += `<line class="grid" x1="${P.l}" x2="${W - P.r}" y1="${Y(yy)}" y2="${Y(yy)}"/>`;
      g += `<text x="${P.l - 6}" y="${Y(yy) + 3}" text-anchor="end">${Math.pow(10, yy).toFixed(yy < 1 ? 1 : 0)}</text>`;
    }
    for (const m of [32, 64, 128, 200]) {
      const lx = Math.log10(m);
      if (lx < x0 || lx > x1) continue;
      g += `<text x="${X(lx)}" y="${H - 10}" text-anchor="middle">${m}</text>`;
    }
    g += `<text x="${W - P.r}" y="${H - 10}" text-anchor="end">mean →</text>`;
    // reference slopes through the data centroid
    for (const [s, name] of [[0, "Gaussian"], [1, "Poisson"], [2, "speckle"]]) {
      const ya = yc + s * (x0 - xc), yb = yc + s * (x1 - xc);
      g += `<path class="ref" d="M${X(x0)} ${Y(ya)}L${X(x1)} ${Y(yb)}"/>`;
      const ty = Math.max(P.t + 8, Math.min(H - P.b - 4, Y(yb) - 4));
      g += `<text x="${W - P.r - 2}" y="${ty}" text-anchor="end">${name} (${s})</text>`;
    }
    if (a.slope != null) {
      const ya = yc + a.slope * (x0 - xc), yb = yc + a.slope * (x1 - xc);
      g += `<path class="fit draw-fit" d="M${X(x0)} ${Y(ya)}L${X(x1)} ${Y(yb)}"/>`;
    }
    pts.forEach(([x, y], i) => {
      g += `<circle class="pt" cx="${X(x)}" cy="${Y(y)}" r="3.5" style="animation:cellIn 320ms var(--ease-out) ${i * 40}ms backwards"/>`;
    });
    svg.innerHTML = g;
    $("#loglogNote").textContent = a.slope != null
      ? `Measured slope ${a.slope.toFixed(2)}. Reference slopes: 0 Gaussian · 1 Poisson · 2 speckle.`
      : "Reference slopes: 0 Gaussian · 1 Poisson · 2 speckle.";
  }

  function buildDetect() {
    wireDrop($("#drop2"), $("#fileInput2"), (f) => {
      if (!f.type.startsWith("image/")) { toast("Please choose an image file."); return; }
      $("#drop2Text").textContent = f.name;
      $$("#detSamples button").forEach((b) => b.setAttribute("aria-pressed", "false"));
      runDetect(f, f.name);
    });
    const wrap = $("#detSamples");
    for (const [sample, noise, level, label] of DETECT_SAMPLES) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", async () => {
        $$("#detSamples button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
        const fd = new FormData();
        Object.entries({ sample, noise, level, seed: 7, filter: "median", ksize: 3 }).forEach(([k, v]) => fd.append(k, v));
        try {
          const r = await api("/api/process", fd);
          runDetect(dataURLtoBlob(r.images.noisy), `${sample}-${noise}.png`);
        } catch (e) { toast(e.message); }
      });
      wrap.append(b);
    }
    initCompare($("#compare2"));
  }

  // ------------------------------------------------------------ benchmark
  async function runCompare() {
    const btn = $("#runCompare");
    btn.classList.add("is-loading");
    try {
      state.compare = await api("/api/compare", formData());
      renderCompare();
      $("#exportCsv").disabled = false;
    } catch (e) {
      toast("Comparison failed: " + e.message);
    } finally {
      btn.classList.remove("is-loading");
    }
  }

  function metricBetter(m, a, b) { return m === "mse" ? a < b : a > b; }

  function renderCompare() {
    const c = state.compare;
    if (!c) return;
    const m = state.metric;
    const kernels = state.cfg.kernels;
    const filters = Object.keys(state.cfg.filters);
    const vals = c.rows.map((r) => r[m]);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const best = c.rows.reduce((a, b) => (metricBetter(m, b[m], a[m]) ? b : a));
    const heat = $("#heat");
    let html = `<div class="h"></div>` + kernels.map((k) => `<div class="h">${k}×${k}</div>`).join("");
    let i = 0;
    for (const f of filters) {
      html += `<div class="h rowh">${filterName(f)}</div>`;
      for (const k of kernels) {
        const r = c.rows.find((x) => x.filter === f && x.ksize === k);
        let w = hi === lo ? 1 : (r[m] - lo) / (hi - lo);
        if (m === "mse") w = 1 - w;
        const isBest = r === best;
        html += `<div class="cell in${isBest ? " best" : ""}" style="--w:${w.toFixed(3)};--i:${i++}" title="${filterName(f)} ${k}×${k}: MSE ${fmt.mse(r.mse)}, PSNR ${fmt.psnr(r.psnr)} dB, SSIM ${fmt.ssim(r.ssim)}, ${r.ms} ms">${fmt[m](r[m])}</div>`;
      }
    }
    heat.innerHTML = html;
    $("#benchCtx").textContent = `${noiseName(state.noise)} · ${state.level}`;
    const verdict = $("#verdict");
    verdict.innerHTML = "";
    const s = document.createElement("strong");
    s.textContent = `${filterName(best.filter)} ${best.ksize}×${best.ksize}`;
    verdict.append(
      document.createTextNode("Best by " + m.toUpperCase() + ": "), s,
      document.createTextNode(` (${fmt[m](best[m])}${m === "psnr" ? " dB" : ""}). The noisy input scored ${fmt[m](c.noisy[m])}.`),
    );
    drawKernelChart(c, m);
    renderGallery(c);
  }

  function drawKernelChart(c, m) {
    const svg = $("#kernelChart");
    const W = 400, H = 220, P = { l: 44, r: 14, t: 14, b: 28 };
    const kernels = state.cfg.kernels;
    const vals = c.rows.map((r) => r[m]).concat([c.noisy[m]]);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.08 || 1;
    lo -= pad; hi += pad;
    const X = (i) => P.l + (i / (kernels.length - 1)) * (W - P.l - P.r);
    const Y = (v) => H - P.b - ((v - lo) / (hi - lo)) * (H - P.t - P.b);
    let g = "";
    for (let i = 0; i <= 4; i++) {
      const v = lo + ((hi - lo) * i) / 4;
      g += `<line class="grid" x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${P.l - 6}" y="${Y(v) + 3}" text-anchor="end">${fmt[m](v)}</text>`;
    }
    kernels.forEach((k, i) => { g += `<text x="${X(i)}" y="${H - 10}" text-anchor="middle">${k}×${k}</text>`; });
    g += `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(c.noisy[m])}" y2="${Y(c.noisy[m])}" stroke="var(--noisy)" stroke-dasharray="4 4"/>`;
    g += `<text x="${W - P.r}" y="${Y(c.noisy[m]) - 5}" text-anchor="end" style="fill:var(--noisy)">noisy input</text>`;
    for (const f of Object.keys(state.cfg.filters)) {
      const rows = kernels.map((k) => c.rows.find((r) => r.filter === f && r.ksize === k));
      const d = rows.map((r, i) => (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(r[m]).toFixed(1)).join("");
      g += `<path class="draw" d="${d}" stroke="${FILTER_COLORS[f]}" style="--len:600"/>`;
      rows.forEach((r, i) => { g += `<circle cx="${X(i)}" cy="${Y(r[m])}" r="3" fill="${FILTER_COLORS[f]}"/>`; });
    }
    svg.innerHTML = g;
    $("#kernelLegend").innerHTML = Object.keys(state.cfg.filters).map((f) => `<li class="f-${f}">${filterName(f)}</li>`).join("");
  }

  function renderGallery(c) {
    const gal = $("#gallery");
    const winner = c.overall.filter;
    gal.innerHTML = "";
    const cards = [["noisy", { image: c.noisy_image, ...c.noisy }], ...Object.entries(c.best)];
    cards.forEach(([f, b], i) => {
      if (f === "noisy") return; // keep the gallery to the four filters
      const el = document.createElement("article");
      el.className = "panel g-card" + (f === winner ? " win" : "");
      el.style.setProperty("--i", i);
      el.innerHTML = `<img alt="Best ${filterName(f)} output" src="${b.image}"><div class="g-body"><h4>${filterName(f)} <span>${b.ksize}×${b.ksize}</span></h4><p>SSIM ${fmt.ssim(b.ssim)} · PSNR ${fmt.psnr(b.psnr)} dB</p></div>`;
      gal.append(el);
    });
    if (c.auto) {
      const best = c.best[winner];
      const gain = c.auto.ssim - best.ssim;
      const el = document.createElement("article");
      el.className = "panel g-card auto" + (gain > 0 ? " win" : "");
      el.style.setProperty("--i", cards.length);
      el.innerHTML = `<img alt="Auto (noise-aware) output" src="${c.auto.image}"><div class="g-body"><h4>${c.auto.method.label} <span>auto</span></h4><p>SSIM ${fmt.ssim(c.auto.ssim)} · PSNR ${fmt.psnr(c.auto.psnr)} dB</p><p class="gain ${gain >= 0 ? "up" : "down"}">${gain >= 0 ? "+" : ""}${gain.toFixed(3)} SSIM vs best classic</p></div>`;
      gal.append(el);
    }
  }

  async function runMatrix() {
    const btn = $("#runMatrix");
    btn.classList.add("is-loading");
    try {
      const fd = new FormData();
      if (state.file) fd.append("file", state.file); else fd.append("sample", state.sample);
      fd.append("seed", state.seed);
      const r = await api("/api/matrix", fd);
      renderMatrix(r);
    } catch (e) {
      toast("Matrix failed: " + e.message);
    } finally {
      btn.classList.remove("is-loading");
    }
  }

  // Table 2 of the synopsis — expected best filter(s) per noise type
  const SYNOPSIS = { salt_pepper: ["median"], gaussian: ["gaussian", "bilateral"], speckle: ["median", "bilateral"], poisson: ["gaussian", "bilateral"] };

  function renderMatrix(r) {
    const card = $("#matrixCard");
    card.hidden = false;
    const filters = Object.keys(state.cfg.filters);
    let html = `<div class="h"></div>` + filters.map((f) => `<div class="h">${filterName(f)}</div>`).join("") + `<div class="h h-auto">Auto ✦</div><div class="h">Synopsis</div>`;
    let i = 0;
    for (const [n, row] of Object.entries(r)) {
      const p = state.cfg.noises[n];
      html += `<div class="h rowh">${p.name}<small>noisy SSIM ${fmt.ssim(row.noisy.ssim)}</small></div>`;
      const ss = filters.map((f) => row.filters[f].ssim);
      const lo = Math.min(...ss), hi = Math.max(...ss);
      for (const f of filters) {
        const c = row.filters[f];
        const w = hi === lo ? 1 : (c.ssim - lo) / (hi - lo);
        html += `<div class="cell in${f === row.best ? " best" : ""}" style="--w:${w.toFixed(3)};--i:${i++}">${fmt.ssim(c.ssim)}<small>k=${c.ksize} · ${fmt.psnr(c.psnr)} dB</small></div>`;
      }
      if (row.auto) {
        const gain = row.auto.ssim - row.filters[row.best].ssim;
        html += `<div class="cell in auto${gain > 0 ? " best" : ""}" style="--w:1;--i:${i++}">${fmt.ssim(row.auto.ssim)}<small>${filterName(row.auto.method)} · ${gain >= 0 ? "+" : ""}${gain.toFixed(3)}</small></div>`;
      }
      const exp = SYNOPSIS[n];
      const ok = exp.includes(row.best);
      html += `<div class="syn"><span>${exp.map(filterName).join(" / ")}</span><span class="${ok ? "ok" : "no"}">${ok ? "✓ confirmed" : "✗ " + filterName(row.best) + " won"}</span></div>`;
    }
    $("#matrix").innerHTML = html;
    const src = state.file ? state.file.name : state.cfg.samples[state.sample].split(" - ")[0];
    $("#matrixCtx").textContent = src;
    if (!reduced.matches) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function exportCsv() {
    const c = state.compare;
    if (!c) return;
    const src = state.file ? state.file.name : state.sample;
    const lines = ["image,noise,level,seed,filter,ksize,mse,psnr_db,ssim,time_ms"];
    lines.push([src, state.noise, state.level, state.seed, "none (noisy)", "", c.noisy.mse.toFixed(3), c.noisy.psnr.toFixed(3), c.noisy.ssim.toFixed(5), ""].join(","));
    for (const r of c.rows) {
      lines.push([src, state.noise, state.level, state.seed, r.filter, r.ksize, r.mse.toFixed(3), r.psnr.toFixed(3), r.ssim.toFixed(5), r.ms].join(","));
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    a.download = `comparison_${src.replace(/\W+/g, "_")}_${state.noise}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  function buildBenchmark() {
    $("#runCompare").addEventListener("click", runCompare);
    $("#runMatrix").addEventListener("click", runMatrix);
    $("#exportCsv").addEventListener("click", exportCsv);
    const seg = $("#metricSeg");
    const ind = document.createElement("span");
    ind.className = "ind";
    seg.prepend(ind);
    for (const b of $$("button", seg)) {
      b.type = "button";
      b.addEventListener("click", () => {
        state.metric = b.dataset.m;
        $$("button", seg).forEach((x) => { x.setAttribute("aria-checked", String(x === b)); x.tabIndex = x === b ? 0 : -1; });
        moveInd(seg);
        renderCompare();
      });
    }
    requestAnimationFrame(() => moveInd(seg));
  }

  // ------------------------------------------------------------ theory visuals
  function gauss(rand) { // Box–Muller
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function poissonSample(lambda) {
    if (lambda > 30) return Math.max(0, Math.round(lambda + Math.sqrt(lambda) * gauss(Math.random)));
    const L = Math.exp(-lambda);
    let k = 0, p = 1;
    do { k++; p *= Math.random(); } while (p > L);
    return k - 1;
  }
  function noisyValue(kind, f) {
    switch (kind) {
      case "salt_pepper": { const r = Math.random(); return r < 0.04 ? 255 : r < 0.08 ? 0 : f; }
      case "gaussian": return f + 22 * gauss(Math.random);
      case "speckle": return f + f * 0.3 * gauss(Math.random);
      case "poisson": { const peak = 18; return (poissonSample((f / 255) * peak) / peak) * 255; }
    }
    return f;
  }

  function buildSwatches() {
    const canvases = $$(".swatch");
    const draw = () => {
      for (const c of canvases) {
        const W = 160, H = 28;
        if (c.width !== W) { c.width = W; c.height = H; }
        const ctx = c.getContext("2d");
        const img = ctx.createImageData(W, H);
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const f = (x / (W - 1)) * 255;
            const v = Math.max(0, Math.min(255, noisyValue(c.dataset.noise, f)));
            const i = (y * W + x) * 4;
            img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
            img.data[i + 3] = 255;
          }
        }
        ctx.putImageData(img, 0, 0);
      }
    };
    draw();
    let visible = false, last = 0;
    new IntersectionObserver((es) => { visible = es.some((e) => e.isIntersecting); }).observe($("#noiseCards"));
    const loop = (t) => {
      if (visible && !reduced.matches && t - last > 110) { draw(); last = t; }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);

    // kernel weight glyphs
    for (const k of $$(".kern")) {
      let html = "";
      for (let y = -2; y <= 2; y++) {
        for (let x = -2; x <= 2; x++) {
          let o;
          const g = Math.exp(-(x * x + y * y) / 2.5);
          if (k.dataset.k === "mean") o = 0.55;
          else if (k.dataset.k === "median") o = x === 0 && y === 0 ? 1 : 0.18;
          else if (k.dataset.k === "gaussian") o = 0.12 + g * 0.88;
          else o = x > 0 ? 0.08 : 0.12 + g * 0.88; // bilateral: weights cut off across an edge
          html += `<i style="--o:${o.toFixed(2)};--d:${Math.abs(x) + Math.abs(y)}"></i>`;
        }
      }
      k.innerHTML = html;
    }
  }

  // ------------------------------------------------------------ hero animation
  function buildHero() {
    const canvas = $("#heroCanvas");
    const ctx = canvas.getContext("2d");
    const N = 176; // working resolution; scaled up with crisp pixels
    const off = document.createElement("canvas");
    off.width = off.height = N;
    const octx = off.getContext("2d", { willReadFrequently: true });
    let clean = null;
    const img = new Image();
    img.onload = () => {
      octx.drawImage(img, 0, 0, N, N);
      const d = octx.getImageData(0, 0, N, N).data;
      clean = new Uint8ClampedArray(N * N);
      for (let i = 0; i < N * N; i++) clean[i] = d[i * 4];
      frame(performance.now(), true);
    };
    img.src = "/api/sample/cameraman";

    const noisy = new Uint8ClampedArray(N * N);
    const out = octx.createImageData(N, N);
    const win = new Uint8Array(9);
    const median9 = () => { // insertion sort on 9 values
      for (let i = 1; i < 9; i++) {
        const v = win[i];
        let j = i - 1;
        while (j >= 0 && win[j] > v) { win[j + 1] = win[j]; j--; }
        win[j + 1] = v;
      }
      return win[4];
    };

    let visible = true, last = 0;
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; }).observe(canvas);

    function frame(t, force) {
      if (clean && (force || (visible && t - last > 70))) {
        last = t;
        const scanX = reduced.matches ? N / 2 : (0.5 + 0.38 * Math.sin(t / 1600)) * N;
        for (let i = 0; i < N * N; i++) {
          const r = Math.random();
          noisy[i] = r < 0.07 ? 255 : r < 0.14 ? 0 : clean[i];
        }
        const o = out.data;
        for (let y = 0; y < N; y++) {
          for (let x = 0; x < N; x++) {
            let v;
            if (x < scanX || x === 0 || y === 0 || x === N - 1 || y === N - 1) v = noisy[y * N + x];
            else {
              let n = 0;
              for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) win[n++] = noisy[(y + dy) * N + x + dx];
              v = median9();
            }
            const i = (y * N + x) * 4;
            o[i] = o[i + 1] = o[i + 2] = v;
            o[i + 3] = 255;
          }
        }
        octx.putImageData(out, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
        const sx = (scanX / N) * canvas.width;
        const grad = ctx.createLinearGradient(sx - 40, 0, sx, 0);
        grad.addColorStop(0, "rgba(255,181,71,0)");
        grad.addColorStop(1, "rgba(255,181,71,.35)");
        ctx.fillStyle = grad;
        ctx.fillRect(sx - 40, 0, 40, canvas.height);
        ctx.fillStyle = "#ffb547";
        ctx.fillRect(sx - 1, 0, 2, canvas.height);
      }
      if (!reduced.matches) requestAnimationFrame(frame);
    }
    if (!reduced.matches) requestAnimationFrame(frame);
    reduced.addEventListener?.("change", () => { if (!reduced.matches) requestAnimationFrame(frame); else frame(0, true); });
  }

  // ------------------------------------------------------------ timeline
  function buildGantt() {
    const g = $("#gantt");
    let html = `<div class="g-row g-head" role="row"><div class="g-label" role="columnheader">Phase</div><div class="g-track">${Array.from({ length: 12 }, (_, i) => `<span>W${i + 1}</span>`).join("")}</div></div>`;
    const label = { done: "completed", active: "in progress", todo: "upcoming" };
    TIMELINE.forEach((p, i) => {
      const left = ((p.from - 1) / 12) * 100, width = ((p.to - p.from + 1) / 12) * 100;
      const weeks = p.from === p.to ? `Week ${p.from}` : `Weeks ${p.from}–${p.to}`;
      html += `<div class="g-row" role="row"><div class="g-label" role="cell"><b>P${p.phase}</b>${p.work}</div>` +
        `<div class="g-track" role="cell" aria-label="${weeks}, ${label[p.status]}">${"<span></span>".repeat(12)}` +
        `<div class="g-bar s-${p.status}" style="left:${left}%;width:${width}%;--i:${i}">${weeks}</div></div></div>`;
    });
    g.innerHTML = html;
  }

  // ------------------------------------------------------------ page chrome
  function buildChrome() {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) { e.target.classList.add("is-in"); io.unobserve(e.target); }
      }
    }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
    $$("[data-reveal]").forEach((el) => io.observe(el));

    const links = $$(".nav a");
    const spy = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        links.forEach((a) => a.classList.toggle("is-active", a.getAttribute("href") === "#" + e.target.id));
      }
    }, { rootMargin: "-45% 0px -50% 0px" });
    $$("main section[id]").forEach((s) => spy.observe(s));

    $("#themeToggle").addEventListener("click", () => {
      const root = document.documentElement;
      const isLight = root.dataset.theme
        ? root.dataset.theme === "light"
        : matchMedia("(prefers-color-scheme: light)").matches;
      root.dataset.theme = isLight ? "dark" : "light";
      try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {}
    });
  }

  // ------------------------------------------------------------ boot
  async function boot() {
    buildChrome();
    buildHero();
    buildGantt();
    buildSwatches();
    try {
      state.cfg = await api("/api/config");
    } catch (e) {
      toast("Cannot reach the Python backend — start it with: python run.py");
      return;
    }
    buildControls();
    buildDetect();
    buildBenchmark();
    runProcess();
  }
  boot();
})();
