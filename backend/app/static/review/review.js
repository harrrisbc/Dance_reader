/* global uPlot */
(() => {
  const COCO_EDGES = [
    [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  ];
  const SKEL_COLORS = [
    ["rgba(80, 180, 255, 0.95)", "rgba(80, 180, 255, 0.35)"],
    ["rgba(80, 220, 120, 0.95)", "rgba(80, 220, 120, 0.35)"],
    ["rgba(220, 160, 80, 0.95)", "rgba(220, 160, 80, 0.35)"],
    ["rgba(200, 100, 220, 0.95)", "rgba(200, 100, 220, 0.35)"],
  ];
  const VIS_MIN = 0.3;
  const POSE_MAX_DT = 0.35;

  // kind matches Qt CHART_SERIES / _plot_series_row
  const SERIES = [
    { attr: "energy", label: "Energy", color: "#4aa3ff", unit: "rel. peak", ylim: [0, 1], on: true, kind: "area" },
    { attr: "speed", label: "Speed", color: "#f0a040", unit: "shoulders/s", ylim: [0, 4], on: true, kind: "line" },
    { attr: "reach", label: "Reach", color: "#78b4a0", unit: "× shoulder", ylim: [1, 3.5], on: true, kind: "reach" },
    { attr: "level", label: "Level", color: "#c45c4a", unit: "stance ratio", ylim: [0, 1.05], on: false, kind: "level" },
    { attr: "travel", label: "Travel", color: "#e8c56e", unit: "shoulders/s", ylim: [0, 2], on: false, kind: "area" },
    { attr: "directness", label: "Directness", color: "#a67c52", unit: "", ylim: [0, 1], on: false, kind: "ribbon" },
    { attr: "pos_x", label: "L–R", color: "#8f7cc0", unit: "screen", ylim: [0, 1], on: false, kind: "pos" },
    { attr: "pos_y", label: "F–B", color: "#c07fa8", unit: "screen", ylim: [0, 1], on: false, kind: "pos" },
  ];

  const state = {
    analysis: null,
    actorId: null,
    view: [0, 1],
    plots: [],
    plotHosts: [],
    videoObjectUrl: null,
    _ro: null,
    _skelRaf: 0,
    _skelBusy: false,
  };

  const $ = (id) => document.getElementById(id);
  const video = $("video");
  const skelCanvas = $("skeletonOverlay");
  const skelCtx = skelCanvas.getContext("2d");
  const chartStack = $("chartStack");
  const statusEl = $("status");
  const floor = $("floorMap");
  const floorCtx = floor.getContext("2d");

  function fmt(t) {
    if (!Number.isFinite(t) || t < 0) return "0:00";
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  function hexAlpha(hex, a) {
    const h = hex.replace("#", "");
    const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
    const r = (n >> 16) & 255;
    const g = (n >> 8) & 255;
    const b = n & 255;
    return `rgba(${r},${g},${b},${a})`;
  }

  function selectedActor() {
    if (!state.analysis?.actors?.length) return null;
    return state.analysis.actors.find((a) => a.id === state.actorId) || state.analysis.actors[0];
  }

  function metricArrays(actor) {
    const times = [];
    const byAttr = Object.fromEntries(SERIES.map((s) => [s.attr, []]));
    for (const m of actor.metrics || []) {
      times.push(m.time);
      for (const s of SERIES) {
        let v = Number(m[s.attr]);
        if (!Number.isFinite(v)) v = null;
        byAttr[s.attr].push(v);
      }
    }
    return { times, byAttr };
  }

  /** Locked Y like Qt: expand only when data exceeds design range. */
  function yRangeFor(s, vals) {
    let yMin = s.ylim[0];
    let yMax = s.ylim[1];
    const finite = vals.filter((v) => v != null && Number.isFinite(v));
    if (finite.length) {
      const dataMax = Math.max(...finite);
      const dataMin = Math.min(...finite);
      if (dataMax > yMax) yMax = dataMax * 1.05;
      if (dataMin < yMin && s.kind !== "reach") yMin = Math.min(yMin, dataMin);
    }
    return [yMin, yMax];
  }

  function nearestPose(poses, t) {
    if (!poses?.length) return null;
    let lo = 0;
    let hi = poses.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (poses[mid].time < t) lo = mid + 1;
      else hi = mid;
    }
    let best = poses[lo];
    let bestD = Math.abs(best.time - t);
    if (lo > 0) {
      const prev = poses[lo - 1];
      const d = Math.abs(prev.time - t);
      if (d < bestD) {
        best = prev;
        bestD = d;
      }
    }
    return bestD <= POSE_MAX_DT ? best : null;
  }

  // ---- skeleton: letterbox + rAF ----

  /** Map overlay canvas to the painted video content (object-fit contain). */
  function layoutSkeletonCanvas() {
    const wrap = video.parentElement;
    if (!wrap) return { w: 1, h: 1 };
    const wrapRect = wrap.getBoundingClientRect();
    const vRect = video.getBoundingClientRect();
    const videoW = video.videoWidth || 1;
    const videoH = video.videoHeight || 1;
    const dispW = vRect.width;
    const dispH = vRect.height;
    const scale = Math.min(dispW / videoW, dispH / videoH);
    const drawW = videoW * scale;
    const drawH = videoH * scale;
    const ox = (dispW - drawW) / 2;
    const oy = (dispH - drawH) / 2;
    const left = vRect.left - wrapRect.left + ox;
    const top = vRect.top - wrapRect.top + oy;
    skelCanvas.style.left = `${left}px`;
    skelCanvas.style.top = `${top}px`;
    skelCanvas.style.width = `${drawW}px`;
    skelCanvas.style.height = `${drawH}px`;
    const bw = Math.max(1, Math.round(drawW));
    const bh = Math.max(1, Math.round(drawH));
    if (skelCanvas.width !== bw || skelCanvas.height !== bh) {
      skelCanvas.width = bw;
      skelCanvas.height = bh;
    }
    return { w: bw, h: bh };
  }

  function drawOneSkeleton(kps, w, h, stroke, glow, style, cheap) {
    const pts = kps.map((kp) => {
      if (!kp || kp.length < 3 || kp[2] < VIS_MIN) return null;
      return { x: kp[0] * w, y: kp[1] * h };
    });

    skelCtx.lineCap = "round";
    skelCtx.lineJoin = "round";

    if (style === "skeleton") {
      if (!cheap) {
        skelCtx.strokeStyle = glow;
        skelCtx.lineWidth = 5;
        skelCtx.globalAlpha = 0.45;
        for (const [a, b] of COCO_EDGES) {
          const pa = pts[a];
          const pb = pts[b];
          if (!pa || !pb) continue;
          skelCtx.beginPath();
          skelCtx.moveTo(pa.x, pa.y);
          skelCtx.lineTo(pb.x, pb.y);
          skelCtx.stroke();
        }
      }
      skelCtx.strokeStyle = stroke;
      skelCtx.lineWidth = 2;
      skelCtx.globalAlpha = cheap ? 0.75 : 0.65;
      for (const [a, b] of COCO_EDGES) {
        const pa = pts[a];
        const pb = pts[b];
        if (!pa || !pb) continue;
        skelCtx.beginPath();
        skelCtx.moveTo(pa.x, pa.y);
        skelCtx.lineTo(pb.x, pb.y);
        skelCtx.stroke();
      }
      skelCtx.globalAlpha = 1;
    }

    skelCtx.fillStyle = stroke;
    skelCtx.globalAlpha = 0.95;
    for (const p of pts) {
      if (!p) continue;
      skelCtx.beginPath();
      skelCtx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      skelCtx.fill();
    }
    skelCtx.globalAlpha = 1;
  }

  function drawSkeletonOverlay(opts = {}) {
    const cheap = !!opts.cheap;
    const { w, h } = layoutSkeletonCanvas();
    skelCtx.clearRect(0, 0, w, h);

    const style = $("skelStyle")?.value || "skeleton";
    if (style === "off" || !state.analysis?.actors?.length) return;

    const t = video.currentTime || 0;
    const showAll = $("skeletonAll").checked && !cheap;

    if (showAll) {
      state.analysis.actors.forEach((actor, i) => {
        const pose = nearestPose(actor.poses, t);
        if (!pose) return;
        const [stroke, glow] = SKEL_COLORS[i % SKEL_COLORS.length];
        drawOneSkeleton(pose.keypoints, w, h, stroke, glow, style, false);
      });
    } else {
      const actor = selectedActor();
      if (!actor) return;
      const pose = nearestPose(actor.poses, t);
      if (!pose) return;
      const [stroke, glow] = SKEL_COLORS[0];
      drawOneSkeleton(pose.keypoints, w, h, stroke, glow, style, cheap);
    }
  }

  function skelTick() {
    state._skelRaf = 0;
    if (video.paused || video.ended) return;
    if (!state._skelBusy) {
      state._skelBusy = true;
      try {
        drawSkeletonOverlay({ cheap: true });
        drawPlayheads(video.currentTime || 0);
      } finally {
        state._skelBusy = false;
      }
    }
    state._skelRaf = requestAnimationFrame(skelTick);
  }

  function startSkelLoop() {
    if (state._skelRaf) return;
    state._skelRaf = requestAnimationFrame(skelTick);
  }

  function stopSkelLoop() {
    if (state._skelRaf) cancelAnimationFrame(state._skelRaf);
    state._skelRaf = 0;
  }

  function setStatus(msg) {
    statusEl.textContent = msg;
  }

  async function refreshStateFromServer() {
    const res = await fetch("/api/review/state");
    const data = await res.json();
    applyServerSnapshot(data);
  }

  function applyServerSnapshot(data) {
    const d = data.device || {};
    $("deviceBadge").textContent = d.device === "cuda"
      ? `Device: CUDA · ${d.gpu || ""}`
      : `Device: ${d.device || "cpu"}`;
    if (data.analysis) {
      state.analysis = data.analysis;
      applyAnalysis();
    }
    if (data.hasVideo) {
      video.src = `/api/media?t=${Date.now()}`;
      setStatus(`Loaded: ${data.videoName || "video"}`);
    } else if (data.hasAnalysis) {
      setStatus("Project loaded, but video file was not found. Use Choose video.");
    }
  }

  function applyAnalysis() {
    const actors = state.analysis?.actors || [];
    const sel = $("actorSelect");
    sel.innerHTML = "";
    actors.forEach((a) => {
      const opt = document.createElement("option");
      opt.value = a.id;
      opt.textContent = a.label;
      sel.appendChild(opt);
    });
    state.actorId = actors[0]?.id || null;
    const dur = Number(state.analysis?.duration) || 1;
    state.view = [0, dur];
    buildSeriesToggles();
    buildMotifList();
    rebuildCharts();
    drawFloor();
    drawSkeletonOverlay({ cheap: false });
  }

  function buildSeriesToggles() {
    const row = $("seriesToggles");
    row.innerHTML = "";
    SERIES.forEach((s) => {
      const lab = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = s.on;
      cb.addEventListener("change", () => {
        s.on = cb.checked;
        rebuildCharts();
      });
      lab.append(cb, document.createTextNode(s.label));
      row.appendChild(lab);
    });
  }

  function buildMotifList() {
    const ul = $("motifList");
    ul.innerHTML = "";
    for (const mt of state.analysis?.motifTypes || []) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.innerHTML = `<span>${mt.name}</span><span>${mt.occurrences.length}×</span>`;
      const scored = mt.occurrences.filter((o) => o.score != null);
      if (scored.length) {
        const best = Math.max(...scored.map((o) => o.score));
        btn.innerHTML = `<span>${mt.name}</span><span>${mt.occurrences.length}× · best ${best.toFixed(2)}</span>`;
      }
      btn.addEventListener("click", () => {
        const occ = mt.occurrences[0];
        if (!occ) return;
        const actor = state.analysis.actors.find((a) => a.id === occ.actorId);
        if (actor) {
          state.actorId = actor.id;
          $("actorSelect").value = actor.id;
          rebuildCharts();
        }
        seekTo(occ.start);
        document.querySelectorAll("#motifList button").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
      });
      li.appendChild(btn);
      ul.appendChild(li);
    }
  }

  function destroyPlots() {
    for (const p of state.plots) p.destroy();
    state.plots = [];
    state.plotHosts = [];
    chartStack.innerHTML = "";
  }

  function plotWidth(host) {
    return Math.max(280, host.clientWidth || chartStack.clientWidth || 480);
  }

  let _resizeRaf = 0;
  function resizePlots() {
    if (_resizeRaf) return;
    _resizeRaf = requestAnimationFrame(() => {
      _resizeRaf = 0;
      let changed = false;
      for (let i = 0; i < state.plots.length; i++) {
        const p = state.plots[i];
        const host = state.plotHosts[i];
        if (!p || !host) continue;
        const w = plotWidth(host);
        if (p.width === w && p.height === 90) continue;
        p.setSize({ width: w, height: 90 });
        changed = true;
      }
      if (changed) drawPlayheads(video.currentTime || 0);
    });
  }

  /** Seek only when pointer is inside plot.over (same space as playhead). */
  function seekFromPointer(plot, clientX, clientY) {
    if (plot.select && plot.select.width > 4) return;
    const rect = plot.over.getBoundingClientRect();
    if (
      clientX < rect.left ||
      clientX > rect.right ||
      clientY < rect.top ||
      clientY > rect.bottom
    ) {
      return;
    }
    const left = clientX - rect.left;
    const t = plot.posToVal(left, "x");
    if (Number.isFinite(t)) seekTo(t);
  }

  function zoomAtPointer(plot, clientX, factor) {
    const rect = plot.over.getBoundingClientRect();
    const left = clientX - rect.left;
    if (left < 0 || left > rect.width) return;
    const center = plot.posToVal(left, "x");
    if (!Number.isFinite(center)) return;
    const [a, b] = state.view;
    const span = b - a;
    let next = Math.max(0.5, Math.min(duration(), span * factor));
    let n0 = center - (center - a) * (next / span);
    let n1 = n0 + next;
    if (n0 < 0) {
      n1 -= n0;
      n0 = 0;
    }
    const dur = duration();
    if (n1 > dur) {
      n0 -= n1 - dur;
      n1 = dur;
    }
    state.view = [Math.max(0, n0), Math.min(dur, n1)];
    syncXScales();
    drawFloor();
    drawPlayheads(video.currentTime || 0);
  }

  function seriesOpts(s) {
    if (s.kind === "area") {
      return {
        stroke: s.color,
        fill: hexAlpha(s.color, 0.35),
        width: 1.2,
        spanGaps: false,
        points: { show: false },
      };
    }
    if (s.kind === "ribbon") {
      return {
        stroke: "transparent",
        width: 0,
        spanGaps: false,
        points: { show: false },
      };
    }
    return {
      stroke: s.color,
      width: 1.3,
      spanGaps: false,
      points: { show: false },
    };
  }

  function kindHooks(s, yMin, yMax) {
    const hooks = {
      setSelect: [
        (u) => {
          const left = u.select.left;
          const width = u.select.width;
          if (width < 4) return;
          const from = u.posToVal(left, "x");
          const to = u.posToVal(left + width, "x");
          const a = Math.min(from, to);
          const b = Math.max(from, to);
          if (b - a < 0.2) return;
          state.view = [a, b];
          syncXScales();
          drawFloor();
          drawPlayheads(video.currentTime || 0);
          u.setSelect({ left: 0, width: 0 }, false);
        },
      ],
    };

    if (s.kind === "level") {
      hooks.drawClear = [
        (u) => {
          const { left, top, width, height } = u.bbox;
          const ctx = u.ctx;
          const bands = [
            [0, 0.45, "rgba(196,92,74,0.12)"],
            [0.45, 0.8, "rgba(232,197,110,0.12)"],
            [0.8, 1.05, "rgba(120,180,160,0.12)"],
          ];
          ctx.save();
          for (const [ya, yb, fill] of bands) {
            const y0 = u.valToPos(Math.min(yb, yMax), "y", true);
            const y1 = u.valToPos(Math.max(ya, yMin), "y", true);
            if (!Number.isFinite(y0) || !Number.isFinite(y1)) continue;
            ctx.fillStyle = fill;
            ctx.fillRect(left, y0, width, Math.max(0, y1 - y0));
          }
          ctx.restore();
        },
      ];
    }

    if (s.kind === "reach") {
      hooks.draw = [
        (u) => {
          const ctx = u.ctx;
          const { left, width } = u.bbox;
          ctx.save();
          ctx.strokeStyle = "rgba(153,153,153,0.7)";
          ctx.setLineDash([4, 4]);
          ctx.lineWidth = 1;
          for (const yv of [1.5, 3.0]) {
            const y = u.valToPos(yv, "y", true);
            if (!Number.isFinite(y)) continue;
            ctx.beginPath();
            ctx.moveTo(left, y);
            ctx.lineTo(left + width, y);
            ctx.stroke();
          }
          ctx.setLineDash([]);
          ctx.fillStyle = "#888";
          ctx.font = "10px sans-serif";
          const yClosed = u.valToPos(1.5, "y", true);
          const yOpen = u.valToPos(3.0, "y", true);
          if (Number.isFinite(yClosed)) ctx.fillText("closed", left + width - 40, yClosed - 2);
          if (Number.isFinite(yOpen)) ctx.fillText("open", left + width - 28, yOpen - 2);
          ctx.restore();
        },
      ];
    }

    if (s.kind === "ribbon") {
      hooks.draw = [
        (u) => {
          const ctx = u.ctx;
          const xs = u.data[0];
          const ys = u.data[1];
          if (!xs || !ys) return;
          const { top, height } = u.bbox;
          const y0 = top + height * 0.25;
          const y1 = top + height * 0.75;
          ctx.save();
          for (let i = 1; i < xs.length; i++) {
            const a = ys[i - 1];
            const b = ys[i];
            if (a == null || b == null || !Number.isFinite(a) || !Number.isFinite(b)) continue;
            const x0 = u.valToPos(xs[i - 1], "x", true);
            const x1 = u.valToPos(xs[i], "x", true);
            if (!Number.isFinite(x0) || !Number.isFinite(x1)) continue;
            const t = Math.max(0, Math.min(1, 0.5 * (a + b)));
            const r = Math.round(196 + (80 - 196) * t);
            const g = Math.round(92 + (180 - 92) * t);
            const bl = Math.round(74 + (160 - 74) * t);
            ctx.fillStyle = `rgb(${r},${g},${bl})`;
            ctx.globalAlpha = 0.85;
            ctx.fillRect(x0, y0, Math.max(1, x1 - x0), y1 - y0);
          }
          ctx.restore();
        },
      ];
    }

    return hooks;
  }

  function rebuildCharts() {
    destroyPlots();
    const actor = selectedActor();
    if (!actor?.metrics?.length) {
      chartStack.innerHTML = `<p class="hint">No metric curves yet</p>`;
      return;
    }
    const { times, byAttr } = metricArrays(actor);
    const enabled = SERIES.filter((s) => s.on);
    if (!enabled.length) {
      chartStack.innerHTML = `<p class="hint">Select at least one curve</p>`;
      return;
    }

    const [x0, x1] = state.view;
    enabled.forEach((s) => {
      const wrap = document.createElement("div");
      wrap.className = "panel";
      const unitBit = s.unit ? `<span>${s.unit}</span>` : `<span></span>`;
      wrap.innerHTML = `<div class="panel-head"><strong>${s.label}</strong>${unitBit}</div>`;
      const host = document.createElement("div");
      wrap.appendChild(host);
      chartStack.appendChild(wrap);

      const [yMin, yMax] = yRangeFor(s, byAttr[s.attr]);
      const isRibbon = s.kind === "ribbon";

      const opts = {
        width: plotWidth(host),
        height: 90,
        pxAlign: 0,
        cursor: {
          drag: { x: true, y: false },
          points: { show: false },
        },
        scales: {
          x: { time: false, min: x0, max: x1 },
          y: { auto: false, min: yMin, max: yMax },
        },
        axes: [
          {
            show: s === enabled[enabled.length - 1],
            stroke: "#a89886",
            grid: { show: false },
            ticks: { show: false },
            font: "11px sans-serif",
          },
          {
            show: !isRibbon,
            stroke: "#a89886",
            grid: { stroke: "rgba(255,255,255,0.06)" },
            ticks: { show: false },
            size: isRibbon ? 8 : 42,
            font: "10px sans-serif",
            label: isRibbon ? undefined : s.unit,
            labelSize: 11,
            labelFont: "10px sans-serif",
          },
        ],
        series: [{}, seriesOpts(s)],
        hooks: kindHooks(s, yMin, yMax),
      };

      const data = [times, byAttr[s.attr]];
      const plot = new uPlot(opts, data, host);
      plot.setSize({ width: plotWidth(host), height: 90 });

      // Seek / zoom only on plot area — same coords as playhead
      plot.over.addEventListener("click", (ev) => {
        seekFromPointer(plot, ev.clientX, ev.clientY);
      });
      plot.over.addEventListener(
        "wheel",
        (ev) => {
          ev.preventDefault();
          zoomAtPointer(plot, ev.clientX, ev.deltaY > 0 ? 1.25 : 0.8);
        },
        { passive: false },
      );

      state.plots.push(plot);
      state.plotHosts.push(host);
    });

    requestAnimationFrame(() => {
      resizePlots();
      drawPlayheads(video.currentTime || 0);
    });
  }

  function duration() {
    return Number(state.analysis?.duration) || video.duration || 1;
  }

  function syncXScales() {
    const [a, b] = state.view;
    for (const p of state.plots) {
      p.setScale("x", { min: a, max: b });
    }
    drawPlayheads(video.currentTime || 0);
  }

  function drawPlayheads(t) {
    state.plots.forEach((p) => {
      const over = p.over;
      if (!over) return;
      let line = over.querySelector(".ph-line");
      if (!line) {
        line = document.createElement("div");
        line.className = "ph-line";
        line.style.cssText =
          "position:absolute;top:0;bottom:0;width:2px;background:#c45c4a;pointer-events:none;z-index:5;";
        over.style.position = "relative";
        over.appendChild(line);
      }
      const x = p.valToPos(t, "x");
      if (Number.isFinite(x)) {
        line.style.left = `${x}px`;
        line.style.display = "block";
      }
    });
  }

  function seekTo(t) {
    if (!Number.isFinite(t)) return;
    video.currentTime = Math.max(0, Math.min(t, duration()));
    drawPlayheads(video.currentTime);
    drawFloor();
    drawSkeletonOverlay({ cheap: false });
  }

  function drawFloor() {
    const w = floor.width;
    const h = floor.height;
    floorCtx.fillStyle = "#111";
    floorCtx.fillRect(0, 0, w, h);
    const actor = selectedActor();
    if (!actor?.metrics?.length) return;

    const [t0, t1] = state.view;
    const pts = actor.metrics.filter((m) => m.time >= t0 && m.time <= t1);
    if (pts.length < 1) return;

    const fw = state.analysis.frame_width || 16;
    const fh = state.analysis.frame_height || 9;
    const aspect = fw / fh;
    let rw = w;
    let rh = w / aspect;
    if (rh > h) {
      rh = h;
      rw = h * aspect;
    }
    const ox = (w - rw) / 2;
    const oy = (h - rh) / 2;

    floorCtx.strokeStyle = "#333";
    floorCtx.strokeRect(ox, oy, rw, rh);

    const span = Math.max(pts[pts.length - 1].time - pts[0].time, 1e-6);
    for (let i = 1; i < pts.length; i++) {
      const u = ((pts[i - 1].time + pts[i].time) * 0.5 - pts[0].time) / span;
      const g = Math.round(216 + (51 - 216) * u);
      floorCtx.strokeStyle = `rgb(${g},${g},${g})`;
      floorCtx.lineWidth = 2;
      floorCtx.beginPath();
      floorCtx.moveTo(ox + pts[i - 1].pos_x * rw, oy + pts[i - 1].pos_y * rh);
      floorCtx.lineTo(ox + pts[i].pos_x * rw, oy + pts[i].pos_y * rh);
      floorCtx.stroke();
    }

    const t = video.currentTime || 0;
    let best = pts[0];
    let bestD = Math.abs(best.time - t);
    for (const p of pts) {
      const d = Math.abs(p.time - t);
      if (d < bestD) {
        best = p;
        bestD = d;
      }
    }
    floorCtx.fillStyle = "#c45c4a";
    floorCtx.beginPath();
    floorCtx.arc(ox + best.pos_x * rw, oy + best.pos_y * rh, 5, 0, Math.PI * 2);
    floorCtx.fill();
  }

  // ---- events ----
  $("actorSelect").addEventListener("change", (e) => {
    state.actorId = e.target.value;
    rebuildCharts();
    drawFloor();
    drawSkeletonOverlay({ cheap: false });
  });

  $("skelStyle").addEventListener("change", () => drawSkeletonOverlay({ cheap: false }));
  $("skeletonAll").addEventListener("change", () => drawSkeletonOverlay({ cheap: false }));

  $("btnResetZoom").addEventListener("click", () => {
    state.view = [0, duration()];
    syncXScales();
    drawFloor();
  });

  $("btnPlay").addEventListener("click", () => {
    if (video.paused) video.play();
    else video.pause();
  });

  video.addEventListener("play", () => {
    $("btnPlay").textContent = "Pause";
    startSkelLoop();
  });
  video.addEventListener("pause", () => {
    $("btnPlay").textContent = "Play";
    stopSkelLoop();
    drawFloor();
    drawSkeletonOverlay({ cheap: false });
    drawPlayheads(video.currentTime || 0);
  });

  video.addEventListener("loadedmetadata", () => drawSkeletonOverlay({ cheap: false }));
  video.addEventListener("seeked", () => {
    drawSkeletonOverlay({ cheap: false });
    drawPlayheads(video.currentTime || 0);
  });

  let lastFloor = 0;
  video.addEventListener("timeupdate", () => {
    const t = video.currentTime;
    $("clock").textContent = `${fmt(t)} / ${fmt(duration())}`;
    // Playhead/skeleton driven by rAF while playing; catch up when paused
    if (video.paused) {
      drawPlayheads(t);
    }
    const now = performance.now();
    if (!video.paused && now - lastFloor > 120) {
      lastFloor = now;
      drawFloor();
    }
  });

  $("projectFile").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    setStatus("Loading project…");
    try {
      const res = await fetch("/api/review/open-upload", { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setStatus(data.detail || "Failed to open project");
        return;
      }
      applyServerSnapshot(data);
      if (!data.hasVideo && data.videoPathHint) {
        setStatus(`Project OK. Choose video (expected: ${data.videoPathHint})`);
      }
    } catch (err) {
      setStatus(String(err));
    }
  });

  $("videoFile").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (state.videoObjectUrl) URL.revokeObjectURL(state.videoObjectUrl);
    state.videoObjectUrl = URL.createObjectURL(file);
    video.src = state.videoObjectUrl;
    const fd = new FormData();
    fd.append("file", file);
    await fetch("/api/review/set-video", { method: "POST", body: fd });
    setStatus(`Video: ${file.name}`);
    drawFloor();
    drawSkeletonOverlay({ cheap: false });
  });

  if (typeof ResizeObserver !== "undefined") {
    state._ro = new ResizeObserver(() => {
      resizePlots();
      drawSkeletonOverlay({ cheap: video.paused ? false : true });
    });
    state._ro.observe(chartStack);
    const wrap = video.parentElement;
    if (wrap) state._ro.observe(wrap);
  } else {
    window.addEventListener("resize", () => {
      resizePlots();
      drawSkeletonOverlay({ cheap: false });
    });
  }

  refreshStateFromServer().catch((err) => setStatus(String(err)));
})();
