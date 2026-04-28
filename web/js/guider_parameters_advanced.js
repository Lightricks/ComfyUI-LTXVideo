import { app } from "../../../scripts/app.js";

const PREFIXES = ["video", "audio"];
const STRING_FIELDS = [
    "sigmas", "cfg", "stg", "rescale", "modality_scale",
    "perturb_attn", "skip_step", "cross_attn",
    "cfg_zero_star",
];
const FIELD_LABELS = {
    sigmas: "Sigmas",
    cfg: "CFG",
    stg: "STG",
    rescale: "Rescale",
    modality_scale: "Modality Scale",
    perturb_attn: "Perturb Attn",
    skip_step: "Skip Step",
    cross_attn: "Cross Attn",
    cfg_zero_star: "CFG Zero*",
};

const _origProps = {};

function toggleWidget(node, widget, show) {
    if (!widget) return;
    if (!_origProps[widget.name]) {
        _origProps[widget.name] = {
            origComputeSize: widget.computeSize,
        };
    }
    const origSize = node.size;

    widget.hidden = !show;
    widget.computeSize = show
        ? _origProps[widget.name].origComputeSize
        : () => [0, -3.3];

    widget.linkedWidgets?.forEach((w) => toggleWidget(node, w, show));

    const height = show
        ? Math.max(node.computeSize()[1], origSize[1])
        : node.computeSize()[1];
    node.setSize([node.size[0], height]);

    if (show) delete widget.computedHeight;
    else widget.computedHeight = 0;
}

function toggleModalityWidgets(node, prefix, enabled) {
    for (const w of node.widgets) {
        if (
            w.name.startsWith(prefix + "_") &&
            w.name !== prefix + "_enabled"
        ) {
            toggleWidget(node, w, enabled);
        }
    }
}

function injectMultilineLabels(node) {
    const labelMap = {};
    for (const prefix of PREFIXES) {
        for (const field of STRING_FIELDS) {
            const key = prefix + "_" + field;
            labelMap[key] = prefix.toUpperCase() + " " + FIELD_LABELS[field];
        }
    }

    function applyLabels() {
        let allDone = true;
        for (const w of node.widgets) {
            if (!(w.name in labelMap)) continue;
            // w.element is the <textarea> itself, not the wrapper div
            const textarea = w.element;
            if (!textarea || textarea.tagName !== "TEXTAREA") {
                allDone = false;
                continue;
            }
            const container = textarea.parentElement;
            if (!container) {
                allDone = false;
                continue;
            }
            if (container.querySelector(".guider-param-label")) continue;

            textarea.placeholder = labelMap[w.name];

            const label = document.createElement("div");
            label.className = "guider-param-label";
            label.textContent = labelMap[w.name];
            label.style.cssText =
                "font-size:10px;font-weight:600;color:#bbb;" +
                "padding:0 4px;pointer-events:none;" +
                "position:absolute;top:-14px;left:0;z-index:1;";
            container.style.overflow = "visible";
            container.insertBefore(label, textarea);
        }
        return allDone;
    }

    let retries = 0;
    function tryApply() {
        if (applyLabels() || retries > 30) return;
        retries++;
        requestAnimationFrame(tryApply);
    }
    requestAnimationFrame(tryApply);
}

const MOD_PREFIX_MAP = { VIDEO: "video", AUDIO: "audio" };

function populateWidgets(node, data) {
    for (const [modName, fields] of Object.entries(data)) {
        const prefix = MOD_PREFIX_MAP[modName];
        if (!prefix) continue;
        for (const [field, value] of Object.entries(fields)) {
            const w = node.widgets.find(
                (w) => w.name === prefix + "_" + field
            );
            if (w) w.value = value;
        }
    }
}

async function importXlsxValues(node, filename) {
    try {
        const resp = await fetch("/guider_params/import", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename }),
        });
        if (!resp.ok) return;
        const data = await resp.json();
        populateWidgets(node, data);
    } catch (err) {
        console.error("XLSX import error:", err);
    }
}

function setupUploadButton(node) {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".xlsx";
    fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    fileInput.addEventListener("change", async () => {
        if (!fileInput.files || fileInput.files.length === 0) return;
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append("image", file, file.name);
        formData.append("type", "input");
        formData.append("overwrite", "true");

        try {
            const resp = await fetch("/upload/image", {
                method: "POST",
                body: formData,
            });
            if (!resp.ok) {
                console.error("XLSX upload failed:", resp.statusText);
                return;
            }
            const result = await resp.json();
            const uploadedName = result.name;

            await importXlsxValues(node, uploadedName);
        } catch (err) {
            console.error("XLSX upload error:", err);
        }
        fileInput.value = "";
    });

    node.addWidget("button", "Upload XLSX", "Upload XLSX", () => {
        fileInput.click();
    });
}

function hasConnectedInput(node, widgetName) {
    if (!node.inputs) return false;
    for (const inp of node.inputs) {
        if (inp.widget && inp.widget.name === widgetName && inp.link != null) {
            return true;
        }
    }
    return false;
}

function collectModalityParams(node) {
    const resolved = node._resolvedValues || {};
    const modalities = {};
    for (const prefix of PREFIXES) {
        const enabledWidget = node.widgets.find(
            (w) => w.name === prefix + "_enabled"
        );
        if (!enabledWidget || !enabledWidget.value) continue;
        const fields = {};
        const resolvedMod = resolved[prefix] || {};
        for (const field of STRING_FIELDS) {
            const widgetName = prefix + "_" + field;
            if (hasConnectedInput(node, widgetName) && field in resolvedMod) {
                fields[field] = resolvedMod[field];
            } else {
                const w = node.widgets.find(
                    (w) => w.name === widgetName
                );
                if (w) fields[field] = w.value;
            }
        }
        const zisName = prefix + "_zero_init_sigma";
        if (hasConnectedInput(node, zisName) && "zero_init_sigma" in resolvedMod) {
            fields["zero_init_sigma"] = resolvedMod["zero_init_sigma"];
        } else {
            const zisWidget = node.widgets.find(
                (w) => w.name === zisName
            );
            if (zisWidget) fields["zero_init_sigma"] = String(zisWidget.value);
        }
        modalities[prefix === "video" ? "VIDEO" : "AUDIO"] = fields;
    }
    return modalities;
}

function setupExportButton(node) {
    node.addWidget("button", "Export XLSX", "Export XLSX", async () => {
        const filename = prompt("Enter filename for XLSX export:", "preset");
        if (!filename) return;

        const modalities = collectModalityParams(node);
        if (Object.keys(modalities).length === 0) {
            alert("No modalities enabled. Enable at least one.");
            return;
        }

        try {
            const resp = await fetch("/guider_params/export", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename, modalities }),
            });
            const result = await resp.json();
            if (!resp.ok) {
                alert("Export failed: " + (result.error || resp.statusText));
                return;
            }

            const url =
                "/view?filename=" +
                encodeURIComponent(result.filename) +
                "&type=" +
                encodeURIComponent(result.type);
            window.open(url, "_blank");
        } catch (err) {
            console.error("Export error:", err);
            alert("Export failed: " + err.message);
        }
    });
}

app.registerExtension({
    name: "ltx.guider_parameters_advanced",

    async nodeCreated(node) {
        if (node.comfyClass !== "GuiderParametersAdvanced") {
            return;
        }

        // Hide internal widgets that buttons manage
        const saveWidget = node.widgets.find(
            (w) => w.name === "save_filename"
        );
        if (saveWidget) toggleWidget(node, saveWidget, false);

        setupUploadButton(node);
        setupExportButton(node);
        injectMultilineLabels(node);

        for (const prefix of PREFIXES) {
            const enabledWidget = node.widgets.find(
                (w) => w.name === prefix + "_enabled"
            );
            if (!enabledWidget) continue;

            const origCallback = enabledWidget.callback;
            enabledWidget.callback = function (...args) {
                if (origCallback) origCallback.apply(this, args);
                toggleModalityWidgets(node, prefix, enabledWidget.value);
            };

            toggleModalityWidgets(node, prefix, enabledWidget.value);
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GuiderParametersAdvanced") {
            return;
        }

        const origOnExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onExecuted = function (message) {
            if (origOnExecuted) {
                origOnExecuted.apply(this, arguments);
            }

            if (
                Array.isArray(message.resolved_values) &&
                message.resolved_values.length > 0
            ) {
                this._resolvedValues = message.resolved_values[0];
            }

            if (
                Array.isArray(message.xlsx_download) &&
                message.xlsx_download.length > 0
            ) {
                const info = message.xlsx_download[0];
                const url =
                    "/view?filename=" +
                    encodeURIComponent(info.filename) +
                    "&type=" +
                    encodeURIComponent(info.type);

                let linkWidget = this.widgets.find(
                    (w) => w.name === "_download_link"
                );
                if (!linkWidget) {
                    linkWidget = this.addWidget(
                        "button",
                        "_download_link",
                        "Download XLSX",
                        () => {
                            window.open(url, "_blank");
                        }
                    );
                }
                linkWidget.label = "Download: " + info.filename;
                linkWidget.callback = () => {
                    window.open(url, "_blank");
                };

                this.setSize(this.computeSize());
            }

        };
    },
});
