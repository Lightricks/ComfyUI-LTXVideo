import { app } from "../../../scripts/app.js";

const MIN_W = 560;
const CHROME_H = 136;
const PREVIEW_MIN_H = 180;
const PREVIEW_MAX_H = 360;

function livePreviewHeight(node) {
  const w = Math.max(node.size?.[0] || MIN_W, MIN_W);
  const inner = Math.max(w - 24, 160);
  const live = node._live;
  const aspect = live?.w > 0 && live?.h > 0 ? live.w / live.h : 16 / 9;
  return Math.round(Math.min(PREVIEW_MAX_H, Math.max(PREVIEW_MIN_H, inner / aspect)));
}

function liveWidgetSize(node) {
  return [Math.max(node.size?.[0] || 0, MIN_W), CHROME_H + livePreviewHeight(node)];
}

function fitNodeToLive(node) {
  const sz = liveWidgetSize(node);
  const computed = node.computeSize?.() || sz;
  node.setSize?.([
    Math.max(node.size[0], computed[0], sz[0], MIN_W),
    Math.max(node.size[1], computed[1], sz[1]),
  ]);
  app.graph?.setDirtyCanvas(true, true);
}

const BTN =
  "font-size:11px;padding:2px 8px;border-radius:4px;border:1px solid #555;background:#2a2a2a;color:#ddd;cursor:pointer;";

function widgetValue(node, name, fallback) {
  const w = node.widgets?.find((x) => x.name === name);
  if (!w) return fallback;
  return w.value;
}

function scaledPx(value, scale, lo, hi) {
  const v = Number(value) * scale;
  if (!Number.isFinite(v) || v === 0) return 0;
  const r = v > 0 ? Math.max(1, Math.round(v)) : Math.min(-1, Math.round(v));
  return Math.max(lo, Math.min(hi, r));
}

function clamp01(v) {
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

function readParams(node, scale) {
  const invert = Boolean(widgetValue(node, "invert", false));
  const expand = scaledPx(widgetValue(node, "expand", 0), scale, -128, 128);
  const feather = Math.min(64, Math.max(0, Number(widgetValue(node, "feather", 0)) * scale));
  const blackClip = Math.min(0.5, Math.max(0, Number(widgetValue(node, "black_clip", 0))));
  const whiteClip = Math.min(0.5, Math.max(0, Number(widgetValue(node, "white_clip", 0))));
  const mix = clamp01(Number(widgetValue(node, "mix", 1)));
  const unmix = clamp01(Number(widgetValue(node, "unmix", 0)));
  const unmixRadius = Math.min(64, Math.max(0.5, Number(widgetValue(node, "unmix_radius", 12)) * scale));
  const background = String(widgetValue(node, "background", "none") || "none");
  const fit = String(widgetValue(node, "fit", "cover") || "cover");
  const length = String(widgetValue(node, "length", "hold") || "hold");
  return {
    invert,
    expand,
    feather,
    blackClip,
    whiteClip,
    mix,
    unmix,
    unmixRadius,
    background,
    fit,
    length,
  };
}

function morph(src, w, h, radius) {
  if (radius === 0) return src;
  const dilate = radius > 0;
  const r = Math.abs(radius) | 0;
  const tmp = new Float32Array(w * h);
  const dst = new Float32Array(w * h);
  const pick = dilate ? (a, b) => (a > b ? a : b) : (a, b) => (a < b ? a : b);
  const seed = dilate ? -1 : 2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let m = seed;
      for (let i = -r; i <= r; i++) {
        const xx = Math.min(w - 1, Math.max(0, x + i));
        m = pick(m, src[y * w + xx]);
      }
      tmp[y * w + x] = m;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let m = seed;
      for (let i = -r; i <= r; i++) {
        const yy = Math.min(h - 1, Math.max(0, y + i));
        m = pick(m, tmp[yy * w + x]);
      }
      dst[y * w + x] = m;
    }
  }
  return dst;
}

function gaussian(src, w, h, sigma) {
  if (sigma <= 0) return src;
  const radius = Math.max(1, Math.round(sigma * 3));
  const k = radius * 2 + 1;
  const kernel = new Float32Array(k);
  let sum = 0;
  for (let i = 0; i < k; i++) {
    const x = i - radius;
    const v = Math.exp(-(x * x) / (2 * sigma * sigma));
    kernel[i] = v;
    sum += v;
  }
  for (let i = 0; i < k; i++) kernel[i] /= sum;
  const tmp = new Float32Array(w * h);
  const dst = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -radius; i <= radius; i++) {
        const xx = Math.min(w - 1, Math.max(0, x + i));
        acc += src[y * w + xx] * kernel[i + radius];
      }
      tmp[y * w + x] = acc;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -radius; i <= radius; i++) {
        const yy = Math.min(h - 1, Math.max(0, y + i));
        acc += tmp[yy * w + x] * kernel[i + radius];
      }
      dst[y * w + x] = acc;
    }
  }
  return dst;
}

function clipLevels(src, blackClip, whiteClip) {
  const lo = blackClip;
  let hi = 1 - whiteClip;
  if (lo <= 0 && hi >= 1) return src;
  if (hi <= lo) hi = lo + 1e-4;
  const dst = new Float32Array(src.length);
  const s = 1 / (hi - lo);
  for (let i = 0; i < src.length; i++) {
    let v = (src[i] - lo) * s;
    if (v < 0) v = 0;
    else if (v > 1) v = 1;
    dst[i] = v;
  }
  return dst;
}

function refineMask(raw, w, h, p) {
  let work = raw;
  if (p.invert) {
    work = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) work[i] = 1 - raw[i];
  }
  const afterInvert = work;
  if (p.mix <= 0) {
    return { work, afterInvert };
  }
  work = morph(work, w, h, p.expand);
  work = clipLevels(work, p.blackClip, p.whiteClip);
  work = gaussian(work, w, h, p.feather);
  if (p.mix < 1) {
    const mixed = new Float32Array(work.length);
    for (let i = 0; i < work.length; i++) {
      mixed[i] = afterInvert[i] * (1 - p.mix) + work[i] * p.mix;
    }
    work = mixed;
  }
  for (let i = 0; i < work.length; i++) {
    const v = work[i];
    if (v < 0) work[i] = 0;
    else if (v > 1) work[i] = 1;
  }
  return { work, afterInvert };
}

function unmixRgb(rgb, alpha, w, h, amount, sigma) {
  if (amount <= 0) return rgb;
  const n = w * h;
  const weight = new Float32Array(n);
  const wr = new Float32Array(n);
  const wg = new Float32Array(n);
  const wb = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = 1 - alpha[i];
    const ww = t * t;
    weight[i] = ww;
    wr[i] = rgb[i * 3] * ww;
    wg[i] = rgb[i * 3 + 1] * ww;
    wb[i] = rgb[i * 3 + 2] * ww;
  }
  const br = gaussian(wr, w, h, sigma);
  const bg = gaussian(wg, w, h, sigma);
  const bb = gaussian(wb, w, h, sigma);
  const den = gaussian(weight, w, h, sigma);
  const out = new Float32Array(rgb.length);
  const floor = 0.1;
  const gainFloor = 0.25;
  for (let i = 0; i < n; i++) {
    const a = alpha[i];
    let gate = a / floor;
    if (gate < 0) gate = 0;
    else if (gate > 1) gate = 1;
    const m = amount * gate;
    const r0 = rgb[i * 3];
    const g0 = rgb[i * 3 + 1];
    const b0 = rgb[i * 3 + 2];
    if (m <= 0) {
      out[i * 3] = r0;
      out[i * 3 + 1] = g0;
      out[i * 3 + 2] = b0;
      continue;
    }
    const d = den[i] < 1e-4 ? 1e-4 : den[i];
    const sa = a < gainFloor ? gainFloor : a;
    const dr = (r0 - (1 - sa) * (br[i] / d)) / sa - r0;
    const dg = (g0 - (1 - sa) * (bg[i] / d)) / sa - g0;
    const db = (b0 - (1 - sa) * (bb[i] / d)) / sa - b0;
    // scale all channels together so the pull stays in gamut without shifting hue
    let t = 1;
    if (dr > 0) t = Math.min(t, (1 - r0) / dr);
    else if (dr < 0) t = Math.min(t, r0 / -dr);
    if (dg > 0) t = Math.min(t, (1 - g0) / dg);
    else if (dg < 0) t = Math.min(t, g0 / -dg);
    if (db > 0) t = Math.min(t, (1 - b0) / db);
    else if (db < 0) t = Math.min(t, b0 / -db);
    if (t < 0) t = 0;
    const s = m * t;
    out[i * 3] = r0 + s * dr;
    out[i * 3 + 1] = g0 + s * dg;
    out[i * 3 + 2] = b0 + s * db;
  }
  return out;
}

function pixelsFromImage(img) {
  const c = document.createElement("canvas");
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  return ctx.getImageData(0, 0, img.width, img.height);
}

function maskFromImageData(data) {
  const n = data.width * data.height;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = data.data[i * 4];
  return out;
}

function rgbFromImageData(data) {
  const n = data.width * data.height;
  const out = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    out[i * 3] = data.data[i * 4];
    out[i * 3 + 1] = data.data[i * 4 + 1];
    out[i * 3 + 2] = data.data[i * 4 + 2];
  }
  return out;
}

function maskToFloat(src) {
  const out = new Float32Array(src.length);
  if (src instanceof Float32Array) {
    out.set(src);
    return out;
  }
  for (let i = 0; i < src.length; i++) out[i] = src[i] / 255;
  return out;
}

function rgbToFloat(src) {
  const out = new Float32Array(src.length);
  if (src instanceof Float32Array) {
    out.set(src);
    return out;
  }
  for (let i = 0; i < src.length; i++) out[i] = src[i] / 255;
  return out;
}

function fitPlate(source, dstW, dstH, fit) {
  const sw = source.width;
  const sh = source.height;
  const c = document.createElement("canvas");
  c.width = dstW;
  c.height = dstH;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, dstW, dstH);
  if (fit === "stretch") {
    ctx.drawImage(source, 0, 0, dstW, dstH);
  } else if (fit === "contain") {
    const s = Math.min(dstW / sw, dstH / sh);
    const nw = sw * s;
    const nh = sh * s;
    ctx.drawImage(source, (dstW - nw) / 2, (dstH - nh) / 2, nw, nh);
  } else {
    const s = Math.max(dstW / sw, dstH / sh);
    const nw = sw * s;
    const nh = sh * s;
    ctx.drawImage(source, (dstW - nw) / 2, (dstH - nh) / 2, nw, nh);
  }
  return rgbFromImageData(ctx.getImageData(0, 0, dstW, dstH));
}

function splitCutX(w, splitX) {
  const t = Number(splitX);
  const x = Math.round(((Number.isFinite(t) ? t : 0.5) * w));
  return Math.max(0, Math.min(w, x));
}

function splitPosFromEvent(canvas, e) {
  const rect = canvas.getBoundingClientRect();
  const cssW = rect.width || 1;
  const cssH = rect.height || 1;
  const imgW = canvas.width || 1;
  const imgH = canvas.height || 1;
  const scale = Math.min(cssW / imgW, cssH / imgH) || 1;
  const dispW = imgW * scale;
  const pad = (cssW - dispW) / 2;
  return Math.max(0, Math.min(1, (e.clientX - rect.left - pad) / dispW));
}

function paint(ctx, w, h, mask, rgb, view, split, splitX, splitMask, bgRgb, splitRgb) {
  const img = ctx.createImageData(w, h);
  const d = img.data;
  const useOver = view === "over";
  const useSplit = Boolean(split);
  const cut = useSplit ? splitCutX(w, splitX) : w;
  for (let i = 0; i < w * h; i++) {
    const x = i % w;
    const y = (i / w) | 0;
    const left = useSplit && x < cut;
    let a = left ? splitMask[i] : mask[i];
    if (a < 0) a = 0;
    else if (a > 1) a = 1;
    let r, g, b;
    if (view === "matte") {
      r = g = b = a;
    } else {
      const src = left && splitRgb ? splitRgb : rgb;
      const chk = ((y >> 4) + (x >> 4)) & 1 ? 0.9 : 0.62;
      const fr = src[i * 3];
      const fg = src[i * 3 + 1];
      const fb = src[i * 3 + 2];
      if (view === "green") {
        r = fr * a;
        g = fg * a + 1 * (1 - a);
        b = fb * a;
      } else if (useOver && bgRgb) {
        r = fr * a + bgRgb[i * 3] * (1 - a);
        g = fg * a + bgRgb[i * 3 + 1] * (1 - a);
        b = fb * a + bgRgb[i * 3 + 2] * (1 - a);
      } else if (useOver) {
        r = fr;
        g = fg;
        b = fb;
      } else {
        r = fr * a + chk * (1 - a);
        g = fg * a + chk * (1 - a);
        b = fb * a + chk * (1 - a);
      }
    }
    d[i * 4] = r * 255;
    d[i * 4 + 1] = g * 255;
    d[i * 4 + 2] = b * 255;
    d[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  if (!useSplit) return;
  const hx = cut + 0.5;
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = Math.max(2, Math.round(w / 320));
  ctx.beginPath();
  ctx.moveTo(hx, 0);
  ctx.lineTo(hx, h);
  ctx.stroke();
  const r = Math.max(6, Math.round(Math.min(w, h) / 36));
  ctx.beginPath();
  ctx.arc(cut, h / 2, r, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.92)";
  ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.45)";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.strokeStyle = "rgba(0,0,0,0.55)";
  ctx.beginPath();
  ctx.moveTo(cut - r * 0.2, h / 2 - r * 0.32);
  ctx.lineTo(cut - r * 0.48, h / 2);
  ctx.lineTo(cut - r * 0.2, h / 2 + r * 0.32);
  ctx.moveTo(cut + r * 0.2, h / 2 - r * 0.32);
  ctx.lineTo(cut + r * 0.48, h / 2);
  ctx.lineTo(cut + r * 0.2, h / 2 + r * 0.32);
  ctx.stroke();
  ctx.font = `${Math.max(11, Math.round(w / 55))}px sans-serif`;
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  const lw = 62;
  const rw = 52;
  const ly = 6;
  const lx = Math.max(6, Math.min(cut - lw - 6, w - lw - rw - 18));
  const rx = Math.max(cut + 6, Math.min(w - rw - 6, cut + 6));
  ctx.fillRect(lx, ly, lw, 18);
  ctx.fillRect(rx, ly, rw, 18);
  ctx.fillStyle = "#fff";
  ctx.fillText("WITHOUT", lx + 4, ly + 13);
  ctx.fillText("WITH", rx + 8, ly + 13);
}

function loadDataUrl(url, { revoke = false } = {}) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      if (revoke) URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      if (revoke) URL.revokeObjectURL(url);
      reject(new Error("Could not read image"));
    };
    img.src = url;
  });
}

function firstVideoFrame(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    let done = false;
    let timer = 0;
    const cleanup = () => {
      clearTimeout(timer);
      URL.revokeObjectURL(url);
    };
    const fail = (err) => {
      if (done) return;
      done = true;
      cleanup();
      reject(err);
    };
    const grab = () => {
      if (done) return;
      done = true;
      const c = document.createElement("canvas");
      c.width = video.videoWidth || 1;
      c.height = video.videoHeight || 1;
      c.getContext("2d").drawImage(video, 0, 0);
      cleanup();
      resolve(c);
    };
    video.onerror = () => fail(new Error("Could not read video"));
    video.onloadeddata = () => {
      if (!video.videoWidth) return;
      if (video.currentTime > 0.01) video.currentTime = 0;
      else setTimeout(grab, 30);
    };
    video.onseeked = grab;
    timer = setTimeout(() => fail(new Error("Timed out reading video")), 8000);
    video.src = url;
    video.load();
  });
}

function findNode(pred) {
  const nodes = app.graph?._nodes || app.graph?.nodes || [];
  return nodes.find(pred);
}

function connectedNode(node, name) {
  const idx = node.inputs?.findIndex((i) => i.name === name);
  if (idx == null || idx < 0) return null;
  if (typeof node.getInputNode === "function") return node.getInputNode(idx);
  const input = node.inputs[idx];
  if (input?.link == null) return null;
  const link = app.graph?.links?.[input.link];
  if (!link) return null;
  const originId = link.origin_id ?? link.originId;
  return (app.graph?.getNodeById?.(originId) || findNode((n) => n.id === originId)) ?? null;
}

function setComboValue(node, names, value) {
  if (!node?.widgets) return false;
  const w = node.widgets.find((x) => names.includes(x.name));
  if (!w) return false;
  const opts = w.options?.values;
  if (Array.isArray(opts) && !opts.includes(value)) {
    opts.push(value);
  }
  w.value = value;
  w.callback?.(value);
  return true;
}

function setWidgetValue(node, name, value) {
  const w = node.widgets?.find((x) => x.name === name);
  if (!w) return false;
  w.value = value;
  w.callback?.(value);
  return true;
}

function hidePlateFileWidget(node) {
  const w = node.widgets?.find((x) => x.name === "plate_file");
  if (!w) return;
  w.hidden = true;
  w.computeSize = () => [0, -4];
}

async function uploadToInput(file) {
  const form = new FormData();
  form.append("image", file);
  form.append("overwrite", "true");
  const resp = await fetch("/upload/image", { method: "POST", body: form });
  if (!resp.ok) throw new Error("Upload failed");
  return resp.json();
}

function applyPlateFile(node, filename, isVideo) {
  setWidgetValue(node, "plate_file", filename);
  setComboValue(node, ["background"], isVideo ? "video" : "image");
  const loader = connectedNode(node, isVideo ? "background_video" : "background_image");
  if (loader) setComboValue(loader, ["file", "video", "image"], filename);
}

function setBarButtons(bar, live) {
  for (const b of bar.querySelectorAll("button[data-view]")) {
    b.style.background = b.dataset.view === live.view ? "#3d5c4a" : "#2a2a2a";
  }
  const splitBtn = bar.querySelector("button[data-split]");
  if (splitBtn) splitBtn.style.background = live.split ? "#3d5c4a" : "#2a2a2a";
}

function graphFps() {
  const nodes = app.graph?._nodes || app.graph?.nodes || [];
  for (const n of nodes) {
    if (n.comfyClass !== "CreateVideo" && n.comfyClass !== "SaveVideo") continue;
    const w = n.widgets?.find((x) => x.name === "fps");
    const v = Number(w?.value);
    if (Number.isFinite(v) && v > 0) return v;
  }
  return 0;
}

function playbackFps(node) {
  const live = node._live;
  const fromGraph = graphFps();
  const base = fromGraph > 0 ? fromGraph : Number(live?.fps) > 0 ? Number(live.fps) : 24;
  const sampled = live?.frameCount || 0;
  const source = live?.sourceFrames || sampled;
  if (sampled > 0 && source > sampled) return base * (sampled / source);
  return base;
}

function setPlaying(node, playing) {
  const live = node._live;
  if (!live) return;
  live.playing = Boolean(playing) && (live.frameCount || 0) > 1;
  live.playAnchor = null;
  if (live.playBtn) live.playBtn.style.background = live.playing ? "#3d5c4a" : "#2a2a2a";
  if (live.stopBtn) live.stopBtn.style.background = live.playing ? "#2a2a2a" : "#5c3d3d";
  updateTransport(node);
  updateStatus(node);
}

function currentFrame(live) {
  const frames = live?.frames;
  if (!frames?.length) return null;
  const n = frames.length;
  const i = Math.max(0, Math.min(live.frameIndex | 0, n - 1));
  if (frames[i]) return frames[i];
  for (let j = 1; j < n; j++) {
    if (frames[(i + j) % n]) return frames[(i + j) % n];
  }
  return null;
}

function updateTransport(node) {
  const live = node._live;
  if (!live) return;
  const n = live.frameCount || 0;
  const i = n ? Math.min(live.frameIndex | 0, n - 1) + 1 : 0;
  if (live.scrub) {
    live.scrub.max = String(Math.max(0, n - 1));
    live.scrub.value = String(Math.max(0, live.frameIndex | 0));
    live.scrub.disabled = n <= 1;
  }
  if (live.frameLabel) {
    live.frameLabel.textContent = n ? `${i} / ${n}` : "—";
  }
  if (live.playBtn) live.playBtn.disabled = n <= 1;
  if (live.stopBtn) live.stopBtn.disabled = n <= 1;
}

function showFrame(node, index, { stop = false } = {}) {
  const live = node._live;
  if (!live?.frameCount) return;
  const next = Math.max(0, Math.min(index | 0, live.frameCount - 1));
  if (stop) setPlaying(node, false);
  if (next !== live.frameIndex) {
    live.frameIndex = next;
    live.dirty = true;
  }
  updateTransport(node);
}

function initLive(node) {
  if (node._live) return;
  const live = {
    view: "checker",
    split: false,
    splitX: 0.5,
    draggingSplit: false,
    fit: "cover",
    scale: 1,
    frames: [],
    frameIndex: 0,
    frameCount: 0,
    playing: false,
    playAnchor: null,
    fps: 24,
    sourceFrames: 0,
    loadGen: 0,
    payloadSig: "",
    bgSource: null,
    bgRgb: null,
    bgKind: null,
    w: 0,
    h: 0,
    dirty: true,
    raf: 0,
    status: null,
    canvas: null,
    ctx: null,
    playBtn: null,
    stopBtn: null,
    scrub: null,
    frameLabel: null,
  };
  node._live = live;

  const wrap = document.createElement("div");
  wrap.style.cssText =
    "display:flex;flex-direction:column;gap:6px;padding:2px 0 4px;height:100%;box-sizing:border-box;";

  const bar = document.createElement("div");
  bar.style.cssText = "display:flex;gap:4px;flex-wrap:wrap;flex-shrink:0;";
  for (const [id, label] of [
    ["checker", "Checker"],
    ["matte", "Matte"],
    ["green", "Green"],
    ["over", "Over BG"],
  ]) {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.dataset.view = id;
    btn.style.cssText = BTN;
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      live.view = id;
      live.dirty = true;
      setBarButtons(bar, live);
    });
    bar.appendChild(btn);
  }
  const splitBtn = document.createElement("button");
  splitBtn.textContent = "Without | with";
  splitBtn.dataset.split = "1";
  splitBtn.style.cssText = BTN;
  splitBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    live.split = !live.split;
    live.dirty = true;
    setBarButtons(bar, live);
    if (live.canvas) live.canvas.style.cursor = live.split ? "ew-resize" : "default";
  });
  bar.appendChild(splitBtn);
  setBarButtons(bar, live);
  wrap.appendChild(bar);

  const plateRow = document.createElement("div");
  plateRow.style.cssText =
    "display:flex;gap:6px;align-items:center;flex-wrap:wrap;flex-shrink:0;";
  const fileBtn = document.createElement("label");
  fileBtn.textContent = "Load plate";
  fileBtn.style.cssText = BTN;
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "image/*,video/*";
  fileInput.style.display = "none";
  fileBtn.appendChild(fileInput);

  const clearBtn = document.createElement("button");
  clearBtn.textContent = "Clear plate";
  clearBtn.style.cssText = BTN;
  clearBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    live.bgSource = null;
    live.bgRgb = null;
    live.bgKind = null;
    live.dirty = true;
    setWidgetValue(node, "plate_file", "");
    setComboValue(node, ["background"], "none");
    updateStatus(node);
  });

  plateRow.appendChild(fileBtn);
  plateRow.appendChild(clearBtn);
  wrap.appendChild(plateRow);

  fileInput.addEventListener("click", (e) => e.stopPropagation());
  fileInput.addEventListener("change", async (e) => {
    e.stopPropagation();
    const file = fileInput.files?.[0];
    fileInput.value = "";
    if (!file) return;
    try {
      const isVideo = file.type.startsWith("video/") || /\.(mp4|webm|mov|mkv|m4v)$/i.test(file.name);
      updateStatus(node, "Uploading plate into Comfy input…");
      const uploaded = await uploadToInput(file);
      const name = uploaded?.name || file.name;
      applyPlateFile(node, name, isVideo);
      live.bgSource = isVideo
        ? await firstVideoFrame(file)
        : await loadDataUrl(URL.createObjectURL(file), { revoke: true });
      live.bgRgb = null;
      live.bgKind = isVideo ? "video" : "image";
      live.view = "over";
      setBarButtons(bar, live);
      live.dirty = true;
      updateStatus(node);
    } catch (err) {
      updateStatus(node, "Could not load plate: " + (err?.message || err));
    }
  });

  const preview = document.createElement("div");
  preview.style.cssText =
    "flex:1 1 auto;min-height:0;position:relative;background:#1a1a1a;border-radius:4px;overflow:hidden;";
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 360;
  canvas.style.cssText =
    "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;cursor:default;touch-action:none;";
  preview.appendChild(canvas);
  wrap.appendChild(preview);
  live.canvas = canvas;
  live.ctx = canvas.getContext("2d");

  const onSplitMove = (e) => {
    if (!node._live?.draggingSplit) return;
    e.preventDefault();
    e.stopPropagation();
    live.splitX = splitPosFromEvent(canvas, e);
    live.dirty = true;
  };
  const onSplitUp = (e) => {
    if (!node._live) return;
    live.draggingSplit = false;
    try {
      canvas.releasePointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    window.removeEventListener("pointermove", onSplitMove, true);
    window.removeEventListener("pointerup", onSplitUp, true);
  };
  canvas.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    if (!live.split || e.button !== 0) return;
    e.preventDefault();
    live.draggingSplit = true;
    live.splitX = splitPosFromEvent(canvas, e);
    live.dirty = true;
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    window.addEventListener("pointermove", onSplitMove, true);
    window.addEventListener("pointerup", onSplitUp, true);
  });
  canvas.addEventListener("pointermove", (e) => e.stopPropagation());
  canvas.addEventListener("wheel", (e) => e.stopPropagation());

  const transport = document.createElement("div");
  transport.style.cssText = "display:flex;gap:6px;align-items:center;flex-shrink:0;";
  const playBtn = document.createElement("button");
  playBtn.textContent = "Play";
  playBtn.style.cssText = BTN;
  playBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    setPlaying(node, true);
  });
  const stopBtn = document.createElement("button");
  stopBtn.textContent = "Stop";
  stopBtn.style.cssText = BTN;
  stopBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    setPlaying(node, false);
  });
  const scrub = document.createElement("input");
  scrub.type = "range";
  scrub.min = "0";
  scrub.max = "0";
  scrub.value = "0";
  scrub.step = "1";
  scrub.style.cssText = "flex:1;min-width:80px;accent-color:#6a9;";
  const onScrub = (e) => {
    e.stopPropagation();
    showFrame(node, Number(scrub.value) || 0, { stop: true });
  };
  scrub.addEventListener("input", onScrub);
  scrub.addEventListener("change", onScrub);
  scrub.addEventListener("pointerdown", (e) => e.stopPropagation());
  const frameLabel = document.createElement("span");
  frameLabel.style.cssText = "font-size:11px;color:#ccc;min-width:52px;text-align:right;";
  frameLabel.textContent = "—";
  transport.appendChild(playBtn);
  transport.appendChild(stopBtn);
  transport.appendChild(scrub);
  transport.appendChild(frameLabel);
  wrap.appendChild(transport);
  live.playBtn = playBtn;
  live.stopBtn = stopBtn;
  live.scrub = scrub;
  live.frameLabel = frameLabel;
  setPlaying(node, false);

  const status = document.createElement("div");
  status.style.cssText = "font-size:11px;color:#aaa;line-height:1.3;flex-shrink:0;";
  wrap.appendChild(status);
  live.status = status;
  updateStatus(node);

  const widget = node.addDOMWidget("live_matte", "LiveMattePreview", wrap, {
    serialize: false,
    hideOnZoom: false,
  });
  widget.computeSize = () => liveWidgetSize(node);

  const tick = () => {
    if (!node._live) return;
    live.raf = requestAnimationFrame(tick);
    const n = live.frameCount || 0;
    if (live.playing && n > 1) {
      const now = performance.now();
      if (live.playAnchor == null) {
        live.playAnchor = now;
        live.playStartFrame = live.frameIndex | 0;
      }
      const elapsed = (now - live.playAnchor) / 1000;
      const idx = (live.playStartFrame + Math.floor(elapsed * playbackFps(node))) % n;
      if (idx !== live.frameIndex) {
        if (live.frames[idx]) {
          live.frameIndex = idx;
          live.dirty = true;
          updateTransport(node);
          updateStatus(node);
        } else {
          for (let j = 1; j < n; j++) {
            const k = (idx + j) % n;
            if (live.frames[k]) {
              live.frameIndex = k;
              live.dirty = true;
              updateTransport(node);
              updateStatus(node);
              break;
            }
          }
        }
      }
    }
    const fr = currentFrame(live);
    if (!fr) return;
    const p = readParams(node, live.scale);
    if (p.fit !== live.fit) {
      live.fit = p.fit;
      live.bgRgb = null;
    }
    if (live.bgSource && (!live.bgRgb || live.w !== fr.w || live.h !== fr.h)) {
      live.w = fr.w;
      live.h = fr.h;
      live.bgRgb = fitPlate(live.bgSource, live.w, live.h, live.fit);
    }
    live.w = fr.w;
    live.h = fr.h;
    const plateRgb = p.background === "none" ? null : fr.bg || live.bgRgb;
    const processKey = JSON.stringify(p) + ":" + (live.frameIndex | 0);
    const key =
      processKey +
      live.view +
      live.split +
      live.splitX +
      (plateRgb ? "p" : "") +
      (live.bgKind || "");
    if (!live.dirty && key === live.lastKey) return;
    if (live.processKey !== processKey || !live.processed) {
      live.processKey = processKey;
      const maskF = maskToFloat(fr.mask);
      const rgbF = rgbToFloat(fr.rgb);
      live.processed = refineMask(maskF, fr.w, fr.h, p);
      live.processedRgb = unmixRgb(
        rgbF,
        live.processed.work,
        fr.w,
        fr.h,
        p.unmix,
        p.unmixRadius
      );
      live.splitRgb = rgbF;
    }
    live.lastKey = key;
    live.dirty = false;
    live.canvas.width = fr.w;
    live.canvas.height = fr.h;
    const plateF = plateRgb
      ? plateRgb instanceof Float32Array
        ? plateRgb
        : rgbToFloat(plateRgb)
      : null;
    paint(
      live.ctx,
      fr.w,
      fr.h,
      live.processed.work,
      live.processedRgb,
      live.view,
      live.split,
      live.splitX,
      live.processed.afterInvert,
      plateF,
      live.splitRgb
    );
  };
  live.raf = requestAnimationFrame(tick);

  requestAnimationFrame(() => fitNodeToLive(node));
}

function updateStatus(node, extra) {
  const live = node._live;
  if (!live?.status) return;
  if (extra) {
    live.status.textContent = extra;
    return;
  }
  const n = live.frameCount || 0;
  if (!n) {
    live.status.textContent =
      "Queue once to load the clip. Then Play to loop, Stop to freeze a frame, and drag sliders.";
    return;
  }
  const i = Math.min(live.frameIndex | 0, n - 1) + 1;
  const motion = live.playing ? "Playing" : "Stopped";
  const p = readParams(node, live.scale || 1);
  const plate =
    p.background !== "none" && (live.bgKind || live.bgRgb || currentFrame(live)?.bg)
      ? ` over ${p.background} (${p.fit})`
      : "";
  live.status.textContent =
    `${motion} · ${i} / ${n}${plate}. Sliders update this frame immediately. Queue Prompt writes the full clip.`;
}

function decodeFramePixels(maskB64, rgbB64, bgB64) {
  return Promise.all([
    loadDataUrl("data:image/png;base64," + maskB64),
    rgbB64 ? loadDataUrl("data:image/jpeg;base64," + rgbB64) : null,
    bgB64 ? loadDataUrl("data:image/jpeg;base64," + bgB64) : null,
  ]).then(([maskImg, rgbImg, bgImg]) => {
    const maskPixels = pixelsFromImage(maskImg);
    const rgbPixels = pixelsFromImage(rgbImg || maskImg);
    return {
      mask: maskFromImageData(maskPixels),
      rgb: rgbFromImageData(rgbPixels),
      bg: bgImg ? rgbFromImageData(pixelsFromImage(bgImg)) : null,
      w: maskPixels.width,
      h: maskPixels.height,
    };
  });
}

function asStringList(value) {
  if (typeof value === "string") return [value];
  if (!Array.isArray(value) || !value.length) return [];
  if (typeof value[0] === "string") return value;
  if (Array.isArray(value[0])) return asStringList(value[0]);
  return [];
}

async function applyLivePayload(node, data) {
  const masks = asStringList(data?.live_mask);
  if (!masks.length || !node._live) return;
  const live = node._live;
  const rgbs = asStringList(data?.live_rgb);
  const bgs = asStringList(data?.live_bg);
  const scale = Number(data?.live_scale?.[0] ?? 1) || 1;
  const fps = Number(data?.live_fps?.[0] ?? graphFps() ?? 24) || 24;
  const sourceFrames = Number(data?.live_source_frames?.[0] ?? masks.length) || masks.length;
  const mid = masks[Math.floor(masks.length / 2)] || "";
  const last = masks[masks.length - 1] || "";
  const sig = `${masks.length}:${masks[0].slice(0, 24)}:${mid.slice(0, 24)}:${last.slice(0, 24)}:${bgs.length}:${(bgs[0] || "").slice(0, 24)}`;
  if (live.payloadSig === sig) return;
  live.payloadSig = sig;
  live.loadGen += 1;
  const gen = live.loadGen;
  live.fps = fps;
  live.sourceFrames = sourceFrames;
  live.scale = scale;
  live.frameCount = masks.length;
  live.frames = new Array(masks.length);
  live.frameIndex = 0;
  live.bgRgb = null;
  live.processed = null;
  live.processKey = "";
  live.dirty = true;
  if (bgs.length) {
    live.bgKind = live.bgKind || "image";
    live.bgSource = null;
  }
  try {
    live.frames[0] = await decodeFramePixels(masks[0], rgbs[0], bgs[0]);
  } catch (err) {
    if (gen !== live.loadGen) return;
    updateStatus(node, "Could not decode preview: " + (err?.message || err));
    return;
  }
  if (gen !== live.loadGen) return;
  live.w = live.frames[0].w;
  live.h = live.frames[0].h;
  live.dirty = true;
  updateTransport(node);
  updateStatus(node);
  if (masks.length > 1) setPlaying(node, true);
  else setPlaying(node, false);

  for (let i = 1; i < masks.length; i++) {
    if (gen !== live.loadGen) return;
    try {
      live.frames[i] = await decodeFramePixels(masks[i], rgbs[i], bgs[i] || bgs[0]);
    } catch {
      continue;
    }
    if (i % 6 === 0) {
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  }
}

const LEGACY_INPUTS = new Set(["fps", "compare_frame", "compare_demo"]);
const COMBO_FALLBACK = {
  background: { ok: ["none", "image", "video"], fallback: "none" },
  fit: { ok: ["cover", "contain", "stretch"], fallback: "cover" },
  length: { ok: ["hold", "loop"], fallback: "hold" },
};

function normalizeCombos(node) {
  if (!node?.widgets) return;
  for (const w of node.widgets) {
    const spec = COMBO_FALLBACK[w.name];
    if (!spec) continue;
    if (!spec.ok.includes(w.value)) w.value = spec.fallback;
  }
}

function stripLegacyInputs(node) {
  if (!node) return;
  if (Array.isArray(node.widgets)) {
    node.widgets = node.widgets.filter((w) => !LEGACY_INPUTS.has(w.name));
  }
  if (!Array.isArray(node.inputs)) return;
  for (let i = node.inputs.length - 1; i >= 0; i--) {
    if (!LEGACY_INPUTS.has(node.inputs[i]?.name)) continue;
    const linkId = node.inputs[i].link;
    if (linkId != null) app.graph?.removeLink?.(linkId);
    node.removeInput(i);
  }
}

app.registerExtension({
  name: "LTXVideo.RefineAlphaMatteLive",

  async nodeCreated(node) {
    if (node.comfyClass !== "LTXRefineAlphaMatte") return;
    stripLegacyInputs(node);
    hidePlateFileWidget(node);
    normalizeCombos(node);
    initLive(node);
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "LTXRefineAlphaMatte") return;
    const origExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (data) {
      origExecuted?.apply(this, arguments);
      if (!this._live) initLive(this);
      const payload = data?.live_mask ? data : data?.ui;
      if (payload) applyLivePayload(this, payload);
    };
    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      origConfigure?.apply(this, arguments);
      stripLegacyInputs(this);
      hidePlateFileWidget(this);
      normalizeCombos(this);
    };
    const origRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      origRemoved?.apply(this, arguments);
      if (this._live?.raf) cancelAnimationFrame(this._live.raf);
      if (this._live) this._live.loadGen += 1;
      this._live = null;
    };
  },
});
