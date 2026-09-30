import { app } from "../../../scripts/app.js";

const NODE_NAME = "LTXVGetTilingSizes";

function linesOf(text) {
  return String(text || "").split("\n").length;
}

function ensurePreview(node, text) {
  const value = Array.isArray(text) ? text.join("\n") : String(text ?? "");
  if (node._tilingPreviewEl) {
    node._tilingPreviewEl.textContent = value;
    const widget = (node.widgets || []).find((w) => w.name === "tiling_preview");
    if (widget?.computeSize) {
      const lines = linesOf(value);
      widget.computedHeight = Math.max(88, 14 + lines * 16);
    }
    return;
  }

  const el = document.createElement("pre");
  el.className = "ltx-tiling-sizes-preview";
  el.textContent = value || "Queue prompt to preview tiling sizes.";
  Object.assign(el.style, {
    margin: "0",
    padding: "6px 8px",
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    fontSize: "11px",
    lineHeight: "1.4",
    whiteSpace: "pre-wrap",
    color: "var(--fg-color, #ddd)",
    background: "var(--bg-color, transparent)",
    border: "1px solid var(--border-color, rgba(255,255,255,0.12))",
    borderRadius: "6px",
    width: "100%",
    boxSizing: "border-box",
    pointerEvents: "none",
  });

  const widget = node.addDOMWidget("tiling_preview", "preview", el, {
    serialize: false,
    hideOnZoom: false,
    getValue() {
      return el.textContent;
    },
    setValue(v) {
      el.textContent = v ?? "";
    },
  });
  widget.computeSize = function (width) {
    const lines = linesOf(el.textContent);
    const height = Math.max(88, 14 + lines * 16);
    this.computedHeight = height;
    return [width, height];
  };

  node._tilingPreviewEl = el;
}

app.registerExtension({
  name: "LTXVideo.GetTilingSizesPreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      ensurePreview(this, "Queue prompt to preview tiling sizes.");
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const text = message?.text;
      if (text == null) {
        return;
      }
      ensurePreview(this, text);
      requestAnimationFrame(() => {
        const size = this.computeSize?.() || this.size;
        if (size && this.onResize) {
          const next = [
            Math.max(size[0], this.size?.[0] || 0),
            Math.max(size[1], this.size?.[1] || 0),
          ];
          this.onResize(next);
        }
        app.graph?.setDirtyCanvas?.(true, false);
      });
    };
  },
});
