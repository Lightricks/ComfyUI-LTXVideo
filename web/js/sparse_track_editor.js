import { app } from "../../../scripts/app.js";

// ---------------------------------------------------------------------------
// Catmull-Rom spline interpolation (written from scratch)
// ---------------------------------------------------------------------------

function catmullRom(p0, p1, p2, p3, t) {
  const t2 = t * t;
  const t3 = t2 * t;
  return {
    x:
      0.5 *
      (2 * p1.x +
        (-p0.x + p2.x) * t +
        (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
        (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3),
    y:
      0.5 *
      (2 * p1.y +
        (-p0.y + p2.y) * t +
        (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
        (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3),
  };
}

function interpolateSpline(controlPoints, numSamples) {
  if (controlPoints.length === 0) return [];
  if (controlPoints.length === 1) {
    return Array.from({ length: numSamples }, () => ({ ...controlPoints[0] }));
  }
  if (controlPoints.length === 2) {
    const [a, b] = controlPoints;
    return Array.from({ length: numSamples }, (_, i) => {
      const t = i / (numSamples - 1);
      return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
    });
  }
  // Pad with phantom endpoints for Catmull-Rom
  const pts = [
    controlPoints[0],
    ...controlPoints,
    controlPoints[controlPoints.length - 1],
  ];
  const nSeg = pts.length - 3;
  const result = [];
  for (let i = 0; i < numSamples; i++) {
    const gT = (i / (numSamples - 1)) * nSeg;
    const seg = Math.min(Math.floor(gT), nSeg - 1);
    const lT = gT - seg;
    result.push(catmullRom(pts[seg], pts[seg + 1], pts[seg + 2], pts[seg + 3], lT));
  }
  return result;
}

// ---------------------------------------------------------------------------
// Visual constants
// ---------------------------------------------------------------------------

const SPLINE_COLORS = [
  "#ef4444", "#22c55e", "#3b82f6", "#f59e0b",
  "#a855f7", "#06b6d4", "#f97316", "#84cc16",
];
const POINT_RADIUS = 5;
const ACTIVE_POINT_RADIUS = 7;
const HIT_TOLERANCE = 14;
const CURVE_LINE_WIDTH = 2.5;
const CANVAS_MIN_HEIGHT = 128;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function hideWidget(w) {
  if (!w) return;
  w.hidden = true;
  w._origComputeSize = w.computeSize;
  w.computeSize = () => [0, -3.3];
  w.computedHeight = 0;
}

function closestSegment(pts, p) {
  let best = Infinity;
  let bestIdx = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len2 = dx * dx + dy * dy;
    let t = len2 === 0 ? 0 : ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    const proj = { x: a.x + t * dx, y: a.y + t * dy };
    const d = dist(proj, p);
    if (d < best) {
      best = d;
      bestIdx = i;
    }
  }
  return { dist: best, segIndex: bestIdx };
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "LTXVideo.SparseTrackEditor",

  async nodeCreated(node) {
    if (node.comfyClass !== "LTXVSparseTrackEditor") return;
    initEditor(node);
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "LTXVSparseTrackEditor") return;

    const origExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (data) {
      origExecuted?.apply(this, arguments);
      if (data?.bg_image?.[0]) {
        loadBgImage(this, data.bg_image[0]);
      }
    };

    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      origConfigure?.apply(this, arguments);
      if (this._ed) {
        reloadState(this);
      }
    };

    const origRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      origRemoved?.apply(this, arguments);
      if (this._ed) {
        cancelAnimationFrame(this._ed.rafId);
        if (this._ed._docClickHandler) {
          document.removeEventListener("click", this._ed._docClickHandler);
        }
        this._ed = null;
      }
    };
  },
});

// ---------------------------------------------------------------------------
// Editor initialisation
// ---------------------------------------------------------------------------

function initEditor(node) {
  const pointsW = node.widgets.find((w) => w.name === "points_store");
  const coordsW = node.widgets.find((w) => w.name === "coordinates");
  hideWidget(pointsW);
  hideWidget(coordsW);

  const ed = {
    splines: [],
    active: 0,
    imgW: 1024,
    imgH: 576,
    bgImg: null,
    drag: null,
    hover: null,
    canvas: null,
    ctx: null,
    menu: null,
    dirty: true,
    rafId: null,
  };
  node._ed = ed;

  // Restore saved state
  try {
    const saved = JSON.parse(pointsW.value);
    if (Array.isArray(saved) && saved.length > 0) {
      ed.splines = saved;
    }
  } catch (_) {
    /* first run */
  }
  if (ed.splines.length === 0) {
    ed.splines = [
      [
        { x: ed.imgW * 0.3, y: ed.imgH * 0.5 },
        { x: ed.imgW * 0.7, y: ed.imgH * 0.5 },
      ],
    ];
  }

  // DOM
  const container = document.createElement("div");
  container.style.position = "relative";
  container.style.minHeight = `${CANVAS_MIN_HEIGHT}px`;

  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 576;
  canvas.style.width = "100%";
  canvas.style.display = "block";
  canvas.style.background = "#1a1a2e";
  canvas.style.borderRadius = "4px";
  canvas.style.cursor = "default";
  canvas.style.pointerEvents = "auto";
  canvas.style.touchAction = "none";
  container.appendChild(canvas);
  ed.canvas = canvas;
  ed.ctx = canvas.getContext("2d");

  const menu = buildContextMenu();
  container.appendChild(menu);
  ed.menu = menu;

  const widget = node.addDOMWidget("track_editor", "TrackEditorWidget", container, {
    serialize: false,
    hideOnZoom: false,
  });
  widget.computeSize = () => {
    const w = node.size[0];
    const aspect = node._ed.imgH / node._ed.imgW;
    const h = Math.max(CANVAS_MIN_HEIGHT, Math.round(w * aspect));
    return [w, h + 10];
  };

  setupEvents(node);
  syncWidgets(node);
  startRenderLoop(node);

  requestAnimationFrame(() => {
    node.setSize?.(node.computeSize());
    app.graph?.setDirtyCanvas(true, true);
  });
}

// ---------------------------------------------------------------------------
// State reload (fires on onConfigure, i.e. workflow load)
// ---------------------------------------------------------------------------

function reloadState(node) {
  const ed = node._ed;
  const pointsW = node.widgets.find((w) => w.name === "points_store");
  try {
    const saved = JSON.parse(pointsW?.value);
    if (Array.isArray(saved) && saved.length > 0) {
      ed.splines = saved;
    }
  } catch (_) {
    /* keep current splines */
  }
  ed.dirty = true;
  syncWidgets(node);
}

// ---------------------------------------------------------------------------
// Background image loading
// ---------------------------------------------------------------------------

function loadBgImage(node, b64) {
  const ed = node._ed;
  const img = new window.Image();
  img.onload = () => {
    const firstLoad = ed.bgImg === null;
    ed.bgImg = img;
    ed.imgW = img.width;
    ed.imgH = img.height;
    if (firstLoad && ed.splines.length === 1 && ed.splines[0].length === 2) {
      ed.splines[0][0] = { x: ed.imgW * 0.3, y: ed.imgH * 0.5 };
      ed.splines[0][1] = { x: ed.imgW * 0.7, y: ed.imgH * 0.5 };
    }
    ed.dirty = true;
    syncWidgets(node);
    requestAnimationFrame(() => {
      node.setSize?.(node.computeSize());
      app.graph?.setDirtyCanvas(true, true);
    });
  };
  img.src = `data:image/jpeg;base64,${b64}`;
}

// ---------------------------------------------------------------------------
// Coordinate transforms  (image space <-> canvas pixel space)
// ---------------------------------------------------------------------------

function getScale(node) {
  const ed = node._ed;
  const c = ed.canvas;
  const scaleX = c.width / ed.imgW;
  const scaleY = c.height / ed.imgH;
  return Math.min(scaleX, scaleY);
}

function imgToCanvas(node, p) {
  const s = getScale(node);
  return { x: p.x * s, y: p.y * s };
}

function canvasToImg(node, p) {
  const s = getScale(node);
  return { x: p.x / s, y: p.y / s };
}

function mouseToCanvas(node, e) {
  const rect = node._ed.canvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * (node._ed.canvas.width / rect.width),
    y: (e.clientY - rect.top) * (node._ed.canvas.height / rect.height),
  };
}

// ---------------------------------------------------------------------------
// Hit-testing
// ---------------------------------------------------------------------------

function hitTestPoint(node, canvasPos) {
  const ed = node._ed;
  const dpr = window.devicePixelRatio || 1;
  const tol = HIT_TOLERANCE * dpr;
  for (let si = 0; si < ed.splines.length; si++) {
    for (let pi = 0; pi < ed.splines[si].length; pi++) {
      const cp = imgToCanvas(node, ed.splines[si][pi]);
      if (dist(cp, canvasPos) < tol) {
        return { si, pi };
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Context menu
// ---------------------------------------------------------------------------

function buildContextMenu() {
  const menu = document.createElement("div");
  Object.assign(menu.style, {
    display: "none",
    position: "absolute",
    background: "#252530",
    border: "1px solid #555",
    borderRadius: "6px",
    padding: "4px 0",
    zIndex: "1000",
    boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
    minWidth: "160px",
    fontFamily: "system-ui, sans-serif",
    fontSize: "13px",
  });
  return menu;
}

function showMenu(node, canvasEvt, imgPos) {
  const ed = node._ed;
  const menu = ed.menu;
  menu.innerHTML = "";

  const hit = hitTestPoint(node, mouseToCanvas(node, canvasEvt));

  const items = [];
  items.push({
    label: "Add Point at Cursor",
    action: () => {
      ed.splines[ed.active].push({ x: imgPos.x, y: imgPos.y });
      ed.dirty = true;
      syncWidgets(node);
    },
  });
  items.push({
    label: "Subdivide Nearest Segment",
    action: () => {
      const spline = ed.splines[ed.active];
      if (spline.length < 2) return;
      const { segIndex } = closestSegment(spline, imgPos);
      const a = spline[segIndex];
      const b = spline[segIndex + 1];
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      spline.splice(segIndex + 1, 0, mid);
      ed.dirty = true;
      syncWidgets(node);
    },
  });
  items.push({ separator: true });
  items.push({
    label: "New Spline",
    action: () => {
      ed.splines.push([
        { x: imgPos.x - 40, y: imgPos.y },
        { x: imgPos.x + 40, y: imgPos.y },
      ]);
      ed.active = ed.splines.length - 1;
      ed.dirty = true;
      syncWidgets(node);
    },
  });
  items.push({
    label: "New Static Spline",
    action: () => {
      ed.splines.push([{ x: imgPos.x, y: imgPos.y }]);
      ed.active = ed.splines.length - 1;
      ed.dirty = true;
      syncWidgets(node);
    },
  });
  items.push({
    label: "Delete Spline",
    action: () => {
      if (ed.splines.length <= 1) return;
      ed.splines.splice(ed.active, 1);
      ed.active = Math.min(ed.active, ed.splines.length - 1);
      ed.dirty = true;
      syncWidgets(node);
    },
    disabled: ed.splines.length <= 1,
  });
  if (hit) {
    const spline = ed.splines[hit.si];
    if (spline.length > 1) {
      items.push({ separator: true });
      items.push({
        label: "Delete Point",
        action: () => {
          spline.splice(hit.pi, 1);
          ed.dirty = true;
          syncWidgets(node);
        },
      });
    }
  }

  for (const item of items) {
    if (item.separator) {
      const hr = document.createElement("div");
      hr.style.borderTop = "1px solid #444";
      hr.style.margin = "4px 8px";
      menu.appendChild(hr);
      continue;
    }
    const el = document.createElement("div");
    el.textContent = item.label;
    const isDisabled = item.disabled;
    Object.assign(el.style, {
      padding: "6px 14px",
      cursor: isDisabled ? "default" : "pointer",
      color: isDisabled ? "#666" : "#ddd",
      whiteSpace: "nowrap",
    });
    if (!isDisabled) {
      el.addEventListener("mouseenter", () => (el.style.background = "#3a3a4a"));
      el.addEventListener("mouseleave", () => (el.style.background = "none"));
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        menu.style.display = "none";
        item.action();
      });
    }
    menu.appendChild(el);
  }

  const rect = node._ed.canvas.getBoundingClientRect();
  const containerRect = node._ed.canvas.parentElement.getBoundingClientRect();
  menu.style.left = `${canvasEvt.clientX - containerRect.left}px`;
  menu.style.top = `${canvasEvt.clientY - containerRect.top}px`;
  menu.style.display = "block";
}

// ---------------------------------------------------------------------------
// Event handling
// ---------------------------------------------------------------------------

function setupEvents(node) {
  const ed = node._ed;
  const canvas = ed.canvas;

  canvas.addEventListener("pointerdown", (e) => {
    ed.menu.style.display = "none";
    if (e.button === 2) return; // right-click handled by contextmenu
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    canvas.setPointerCapture(e.pointerId);

    const cp = mouseToCanvas(node, e);
    const hit = hitTestPoint(node, cp);
    if (hit) {
      ed.active = hit.si;
      ed.drag = { ...hit, pointerId: e.pointerId };
      canvas.style.cursor = "grabbing";
      ed.dirty = true;
    }
  });

  canvas.addEventListener("pointermove", (e) => {
    const cp = mouseToCanvas(node, e);

    if (ed.drag) {
      e.preventDefault();
      e.stopPropagation();
      const imgP = canvasToImg(node, cp);
      imgP.x = Math.max(0, Math.min(ed.imgW, imgP.x));
      imgP.y = Math.max(0, Math.min(ed.imgH, imgP.y));
      ed.splines[ed.drag.si][ed.drag.pi] = imgP;
      ed.dirty = true;
      syncWidgets(node);
      return;
    }

    const hit = hitTestPoint(node, cp);
    if (hit) {
      canvas.style.cursor = "grab";
      ed.hover = hit;
    } else {
      canvas.style.cursor = "default";
      ed.hover = null;
    }
    ed.dirty = true;
  });

  canvas.addEventListener("pointerup", (e) => {
    if (ed.drag) {
      e.preventDefault();
      e.stopPropagation();
      canvas.releasePointerCapture(e.pointerId);
      ed.drag = null;
      canvas.style.cursor = "default";
      ed.dirty = true;
      syncWidgets(node);
    }
  });

  canvas.addEventListener("pointerleave", () => {
    ed.hover = null;
    ed.dirty = true;
  });

  canvas.addEventListener("lostpointercapture", () => {
    if (ed.drag) {
      ed.drag = null;
      canvas.style.cursor = "default";
      ed.dirty = true;
      syncWidgets(node);
    }
  });

  canvas.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const cp = mouseToCanvas(node, e);
    const imgP = canvasToImg(node, cp);
    showMenu(node, e, imgP);
  });

  const docClickHandler = (e) => {
    if (!ed.menu.contains(e.target)) {
      ed.menu.style.display = "none";
    }
  };
  document.addEventListener("click", docClickHandler);
  ed._docClickHandler = docClickHandler;
}

// ---------------------------------------------------------------------------
// Widget synchronisation
// ---------------------------------------------------------------------------

function syncWidgets(node) {
  const ed = node._ed;
  const pointsW = node.widgets.find((w) => w.name === "points_store");
  const coordsW = node.widgets.find((w) => w.name === "coordinates");
  const samplesW = node.widgets.find((w) => w.name === "points_to_sample");

  if (pointsW) pointsW.value = JSON.stringify(ed.splines);

  const numSamples = samplesW ? samplesW.value : 121;
  const interpolated = ed.splines.map((sp) => interpolateSpline(sp, numSamples));
  // Round to integers for pixel-level coordinates
  const rounded = interpolated.map((track) =>
    track.map((p) => ({ x: Math.round(p.x), y: Math.round(p.y) }))
  );
  if (coordsW) coordsW.value = JSON.stringify(rounded);
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function startRenderLoop(node) {
  const ed = node._ed;

  function loop() {
    ed.rafId = requestAnimationFrame(loop);
    resizeCanvas(node);
    if (!ed.dirty) return;
    ed.dirty = false;
    render(node);
  }

  loop();
}

function resizeCanvas(node) {
  const ed = node._ed;
  const canvas = ed.canvas;
  const dispW = canvas.clientWidth || node.size?.[0] || 400;

  const aspect = ed.imgH / ed.imgW;
  const dispH = Math.max(CANVAS_MIN_HEIGHT, Math.round(dispW * aspect));

  canvas.style.height = `${dispH}px`;

  const dpr = window.devicePixelRatio || 1;
  const bufW = Math.round(dispW * dpr);
  const bufH = Math.round(dispH * dpr);

  if (canvas.width !== bufW || canvas.height !== bufH) {
    canvas.width = bufW;
    canvas.height = bufH;
    ed.dirty = true;
  }
}

function render(node) {
  const ed = node._ed;
  const ctx = ed.ctx;
  const c = ed.canvas;

  ctx.clearRect(0, 0, c.width, c.height);

  // Background image
  if (ed.bgImg) {
    const s = getScale(node);
    ctx.globalAlpha = 0.85;
    ctx.drawImage(ed.bgImg, 0, 0, ed.imgW * s, ed.imgH * s);
    ctx.globalAlpha = 1.0;
  }

  const samplesW = node.widgets.find((w) => w.name === "points_to_sample");
  const numSamples = samplesW ? samplesW.value : 121;
  const dpr = window.devicePixelRatio || 1;

  // Draw each spline
  for (let si = 0; si < ed.splines.length; si++) {
    const isActive = si === ed.active;
    const color = SPLINE_COLORS[si % SPLINE_COLORS.length];
    const spline = ed.splines[si];

    // Interpolated curve
    if (spline.length >= 2) {
      const curve = interpolateSpline(spline, Math.max(numSamples, 60));
      ctx.beginPath();
      const p0 = imgToCanvas(node, curve[0]);
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < curve.length; i++) {
        const p = imgToCanvas(node, curve[i]);
        ctx.lineTo(p.x, p.y);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = (isActive ? CURVE_LINE_WIDTH * 1.4 : CURVE_LINE_WIDTH) * dpr;
      ctx.globalAlpha = isActive ? 1.0 : 0.5;
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }

    // Control points
    for (let pi = 0; pi < spline.length; pi++) {
      const cp = imgToCanvas(node, spline[pi]);
      const isHov = ed.hover && ed.hover.si === si && ed.hover.pi === pi;
      const isDrag = ed.drag && ed.drag.si === si && ed.drag.pi === pi;
      const baseR =
        isHov || isDrag
          ? ACTIVE_POINT_RADIUS
          : isActive
            ? POINT_RADIUS
            : POINT_RADIUS * 0.8;
      const r = baseR * dpr;

      // Outer ring
      ctx.beginPath();
      ctx.arc(cp.x, cp.y, r + 2 * dpr, 0, Math.PI * 2);
      ctx.fillStyle = isActive ? "#fff" : "rgba(255,255,255,0.5)";
      ctx.fill();

      // Inner fill
      ctx.beginPath();
      ctx.arc(cp.x, cp.y, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // Point number label
      if (isActive) {
        ctx.fillStyle = "#fff";
        ctx.font = `bold ${Math.round(11 * dpr)}px system-ui`;
        ctx.textAlign = "center";
        ctx.fillText(String(pi), cp.x, cp.y - r - 4 * dpr);
      }
    }
  }

  // Active spline indicator
  if (ed.splines.length > 1) {
    const dpr = window.devicePixelRatio || 1;
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(4 * dpr, 4 * dpr, 130 * dpr, 20 * dpr);
    ctx.fillStyle = SPLINE_COLORS[ed.active % SPLINE_COLORS.length];
    ctx.font = `bold ${11 * dpr}px system-ui`;
    ctx.textAlign = "left";
    ctx.fillText(
      `Spline ${ed.active + 1} / ${ed.splines.length}`,
      10 * dpr,
      18 * dpr
    );
  }
}
