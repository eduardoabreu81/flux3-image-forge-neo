// FLUX 3 Image Klein support for Forge Neo
// The "FLUX 3 Layout" panel: draw the boxes of a FLUX 3 layout or box edit on a frame shaped like the picture, and
// write them into the prompt as the JSON list FLUX 3 reads (Black Forest Labs' bounding box guide):
//   layout:    {"id", "bbox": [top, left, bottom, right], "desc"}
//   box edit:  {"id", "from": "target" | "ref_image_N", "src_bbox", "tgt_bbox" (null removes), "desc"}
// on a 0-1000 grid. The pure functions at the top are shared with the tests (tests/js/layout_test.js).

(function () {
    "use strict";

    const KINDS = {
        place: "place",       // layout: where it goes in a new picture
        add: "add",           // box edit: something new, at tgt_bbox
        ref: "ref",           // box edit: from a reference, src_bbox -> tgt_bbox (move, keep, replace)
        remove: "remove",     // box edit: from a reference, src_bbox -> gone
    };

    function clamp(v) {
        return Math.max(0, Math.min(1000, Math.round(v)));
    }

    function normBox(box) {
        if (!box) return null;
        const [a, b, c, d] = box.map(clamp);
        return [Math.min(a, c), Math.min(b, d), Math.max(a, c), Math.max(b, d)];
    }

    function elementJson(el) {
        const id = String(el.id || "").trim();
        const desc = String(el.desc || "").trim();
        if (el.kind === KINDS.place) return {id, bbox: normBox(el.tgt), desc};
        const from = el.kind === KINDS.add ? "target" : `ref_image_${el.ref || 0}`;
        const src = el.kind === KINDS.add ? null : normBox(el.src);
        const tgt = el.kind === KINDS.remove ? null : normBox(el.tgt);
        return {id, from, src_bbox: src, tgt_bbox: tgt, desc};
    }

    function isComplete(el) {
        if (!String(el.id || "").trim()) return false;
        if (el.kind === KINDS.place || el.kind === KINDS.add) return !!el.tgt;
        if (el.kind === KINDS.remove) return !!el.src;
        return !!el.src && !!el.tgt;
    }

    // Black Forest Labs' own style: {"id": "obj_1", "bbox": [80, 60, 480, 380], "desc": "..."}
    function formatValue(v) {
        if (Array.isArray(v)) return `[${v.join(", ")}]`;
        return JSON.stringify(v);
    }

    function buildLayout(elements) {
        const items = elements.filter(isComplete).map(elementJson);
        if (!items.length) return "";
        return `[${items.map((o) => `{${Object.entries(o).map(([k, v]) => `${JSON.stringify(k)}: ${formatValue(v)}`).join(", ")}}`).join(", ")}]`;
    }

    // the [ ... ] span of a layout list in the prompt (intact, or with the braces sd-dynamic-prompts strips)
    function findLayoutSpan(text) {
        const re = /\[\s*\{?\s*"id"\s*:/g;
        let match, last = null;
        while ((match = re.exec(text)) !== null) last = match.index;
        if (last === null) return null;
        let depth = 0, inString = false, escaped = false;
        for (let i = last; i < text.length; i++) {
            const c = text[i];
            if (inString) {
                if (escaped) escaped = false;
                else if (c === "\\") escaped = true;
                else if (c === '"') inString = false;
            } else if (c === '"') inString = true;
            else if (c === "[") depth++;
            else if (c === "]" && --depth === 0) return [last, i + 1];
        }
        return null;
    }

    function writeLayout(prompt, json) {
        const span = findLayoutSpan(prompt);
        if (span) {
            const before = prompt.slice(0, span[0]).replace(/\s+$/, "");
            const after = prompt.slice(span[1]);
            return json ? `${before} ${json}${after}` : `${before}${after}`;
        }
        if (!json) return prompt;
        return prompt.replace(/\s+$/, "") + (prompt.trim() ? " " : "") + json;
    }

    function readLayout(prompt) {
        const span = findLayoutSpan(prompt);
        if (!span) return [];
        let list;
        try {
            list = JSON.parse(prompt.slice(span[0], span[1]));
        } catch (e) {
            return [];
        }
        return list.filter((o) => o && typeof o === "object").map((o) => {
            if ("bbox" in o) return {id: o.id, desc: o.desc || "", kind: KINDS.place, tgt: o.bbox, src: null, ref: 0};
            const from = String(o.from || "target");
            if (from === "target") return {id: o.id, desc: o.desc || "", kind: KINDS.add, tgt: o.tgt_bbox, src: null, ref: 0};
            const ref = parseInt(from.replace("ref_image_", ""), 10) || 0;
            const kind = o.tgt_bbox ? KINDS.ref : KINDS.remove;
            return {id: o.id, desc: o.desc || "", kind, tgt: o.tgt_bbox || null, src: o.src_bbox || null, ref};
        });
    }

    const api = {KINDS, buildLayout, writeLayout, readLayout, findLayoutSpan, elementJson, isComplete};
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
        return;
    }
    window.Flux3Layout = api;

    // ---- the panel ----

    const COLORS = ["#ff4d4d", "#4da3ff", "#ffb84d", "#57d957", "#d24dff", "#4dffe1", "#ff4da6", "#c8ff4d"];
    const LABELS = {place: "place", add: "add new", ref: "from reference", remove: "remove"};

    function el(tag, attrs, ...children) {
        const node = document.createElement(tag);
        for (const [k, v] of Object.entries(attrs || {})) {
            if (k === "class") node.className = v;
            else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
            else if (v !== undefined && v !== null) node.setAttribute(k, v);
        }
        for (const c of children) node.append(c);
        return node;
    }

    function setPrompt(textarea, value) {
        textarea.value = value;
        textarea.dispatchEvent(new Event("input", {bubbles: true}));
    }

    function numberOf(root, id, fallback) {
        const input = root.querySelector(`#${id} input[type=number]`);
        const v = input ? parseFloat(input.value) : NaN;
        return v > 0 ? v : fallback;
    }

    function initPanel(host) {
        const tab = host.dataset.tab;
        const app = gradioApp();
        const state = {elements: [], selected: -1, armed: null, counter: 0};  // armed: {i, which} for the next drag

        const frame = el("div", {class: "flux3-layout-frame"});
        const background = el("img", {class: "flux3-layout-bg", draggable: "false", alt: ""});
        const layer = el("div", {class: "flux3-layout-boxes"});
        frame.append(background, layer);
        const hint = el("div", {class: "flux3-layout-hint"});
        const list = el("div", {class: "flux3-layout-list"});
        const buttons = el("div", {class: "flux3-layout-buttons"},
            el("button", {class: "lg secondary gradio-button", onclick: () => write()}, "Write to prompt"),
            el("button", {class: "lg secondary gradio-button", onclick: () => read()}, "Read from prompt"),
            el("button", {class: "lg secondary gradio-button", onclick: () => clear()}, "Clear"));
        host.append(hint, frame, list, buttons);

        const prompt = () => app.querySelector(`#${tab}_prompt textarea`);

        function fitFrame() {
            const w = numberOf(app, `${tab}_width`, 1024), h = numberOf(app, `${tab}_height`, 1024);
            frame.style.aspectRatio = `${w} / ${h}`;
            const img = tab === "img2img" ? app.querySelector("#img2img_image img.forge-image") : null;
            const src = img && img.getAttribute("src");
            if (src) {
                background.src = src;
                background.style.display = "";
            } else {
                background.removeAttribute("src");
                background.style.display = "none";
            }
        }

        function color(i) {
            return COLORS[i % COLORS.length];
        }

        function updateHint() {
            const armed = state.armed && state.elements[state.armed.i];
            if (!armed) {
                hint.textContent = "Drag on the frame to draw a box: each one becomes an element below. Name it, describe it, then Write to prompt.";
            } else if (state.armed.which === "src") {
                hint.textContent = `Drag where "${armed.id}" is now, in reference image ${Number(armed.ref) + 1}.`;
            } else {
                hint.textContent = `Drag where "${armed.id}" should go.`;
            }
        }

        function renderBoxes() {
            layer.replaceChildren();
            state.elements.forEach((e, i) => {
                for (const which of ["src", "tgt"]) {
                    const b = e[which];
                    if (!b || (which === "tgt" && e.kind === KINDS.remove) || (which === "src" && (e.kind === KINDS.place || e.kind === KINDS.add))) continue;
                    const [t, l, bo, r] = normBox(b);
                    const box = el("div", {class: `flux3-layout-box ${which}${i === state.selected ? " selected" : ""}`},
                        el("span", {}, which === "src" ? `${e.id} (from)` : e.id));
                    Object.assign(box.style, {top: `${t / 10}%`, left: `${l / 10}%`, height: `${(bo - t) / 10}%`, width: `${(r - l) / 10}%`, borderColor: color(i)});
                    box.querySelector("span").style.background = color(i);
                    layer.append(box);
                }
            });
        }

        function renderList() {
            list.replaceChildren();
            state.elements.forEach((e, i) => {
                const kind = el("select", {onchange: (ev) => {
                    e.kind = ev.target.value;
                    if ((e.kind === KINDS.ref || e.kind === KINDS.remove) && !e.src) arm(i, "src");
                    render();
                }}, ...Object.keys(LABELS).map((k) => el("option", {value: k}, LABELS[k])));
                kind.value = e.kind;
                const ref = el("select", {onchange: (ev) => { e.ref = parseInt(ev.target.value, 10); render(); }},
                    ...Array.from({length: 10}, (_, n) => el("option", {value: String(n)}, `ref ${n + 1}`)));
                ref.value = String(e.ref || 0);
                ref.style.display = e.kind === KINDS.ref || e.kind === KINDS.remove ? "" : "none";
                const needsSrc = e.kind === KINDS.ref || e.kind === KINDS.remove;
                const tagButton = el("button", {class: "secondary gradio-button", title: "Insert <name> at the cursor in the prompt", onclick: () => insertTag(e.id)}, `<${e.id}>`);
                const row = el("div", {class: `flux3-layout-row${i === state.selected ? " selected" : ""}${isComplete(e) ? "" : " incomplete"}`},
                    el("span", {class: "flux3-layout-swatch", onclick: () => { state.selected = i; render(); }}),
                    // typing updates the state in place: rebuilding the list would drop the focus and the next click
                    el("input", {type: "text", value: e.id, title: "Name: the caption mentions it as <name>", oninput: (ev) => {
                        e.id = ev.target.value.trim().replace(/[\s<>"]+/g, "_");
                        tagButton.textContent = `<${e.id}>`;
                        row.classList.toggle("incomplete", !isComplete(e));
                        renderBoxes();
                    }}),
                    el("input", {type: "text", class: "flux3-layout-desc", value: e.desc, placeholder: "Short description: look, colors, material", oninput: (ev) => { e.desc = ev.target.value; }}),
                    kind, ref,
                    needsSrc ? el("button", {class: "secondary gradio-button", title: "Draw where it is in the reference", onclick: () => arm(i, "src")}, e.src ? "Redraw from" : "Draw from") : "",
                    e.kind !== KINDS.remove ? el("button", {class: "secondary gradio-button", title: "Draw where it goes", onclick: () => arm(i, "tgt")}, e.tgt ? "Redraw" : "Draw") : "",
                    tagButton,
                    el("button", {class: "secondary gradio-button", title: "Delete", onclick: () => remove(i)}, "✕"));
                row.querySelector(".flux3-layout-swatch").style.background = color(i);
                list.append(row);
            });
        }

        function render() {
            renderBoxes();
            renderList();
            updateHint();
        }

        function arm(i, which) {
            state.selected = i;
            state.armed = {i, which};
            render();
        }

        function remove(i) {
            state.elements.splice(i, 1);
            state.selected = -1;
            state.armed = null;
            render();
        }

        function clear() {
            state.elements = [];
            state.selected = -1;
            state.armed = null;
            render();
        }

        function insertTag(id) {
            const ta = prompt();
            if (!ta) return;
            const tag = `<${id}>`;
            const at = ta.selectionStart ?? ta.value.length;
            setPrompt(ta, ta.value.slice(0, at) + tag + ta.value.slice(ta.selectionEnd ?? at));
            ta.focus();
            ta.selectionStart = ta.selectionEnd = at + tag.length;
        }

        function write() {
            const ta = prompt();
            if (ta) setPrompt(ta, writeLayout(ta.value, buildLayout(state.elements)));
        }

        function read() {
            const ta = prompt();
            if (!ta) return;
            state.elements = readLayout(ta.value);
            state.counter = state.elements.length;
            state.selected = -1;
            state.armed = null;
            render();
        }

        function point(ev) {
            const r = frame.getBoundingClientRect();
            return [clamp(((ev.clientY - r.top) / r.height) * 1000), clamp(((ev.clientX - r.left) / r.width) * 1000)];
        }

        let start = null, ghost = null;
        frame.addEventListener("mousedown", (ev) => {
            if (ev.button !== 0) return;
            ev.preventDefault();
            fitFrame();
            start = point(ev);
            ghost = el("div", {class: "flux3-layout-box ghost"});
            layer.append(ghost);
        });
        window.addEventListener("mousemove", (ev) => {
            if (!start || !ghost) return;
            const [t, l, b, r] = normBox([...start, ...point(ev)]);
            Object.assign(ghost.style, {top: `${t / 10}%`, left: `${l / 10}%`, height: `${(b - t) / 10}%`, width: `${(r - l) / 10}%`});
        });
        window.addEventListener("mouseup", (ev) => {
            if (!start) return;
            const box = normBox([...start, ...point(ev)]);
            start = null;
            if (ghost) ghost.remove();
            ghost = null;
            if (box[2] - box[0] < 10 || box[3] - box[1] < 10) {
                // a click, not a drag: select the smallest box under it (or nothing), and disarm
                const [y, x] = [box[0], box[1]];
                let best = -1, area = Infinity;
                state.elements.forEach((e, i) => {
                    for (const b of [e.tgt, e.src]) {
                        if (!b) continue;
                        const [t, l, bo, r] = normBox(b);
                        if (y >= t && y <= bo && x >= l && x <= r && (bo - t) * (r - l) < area) {
                            best = i;
                            area = (bo - t) * (r - l);
                        }
                    }
                });
                state.selected = best;
                state.armed = null;
                render();
                return;
            }
            const armed = state.armed && state.elements[state.armed.i];
            if (armed) {
                armed[state.armed.which] = box;
                // an element from a reference still needs where it goes, unless it is removed
                const next = state.armed.which === "src" && armed.kind === KINDS.ref && !armed.tgt ? "tgt" : null;
                state.armed = next ? {i: state.armed.i, which: next} : null;
            } else {
                state.counter += 1;
                state.elements.push({id: `obj_${state.counter}`, desc: "", kind: tab === "img2img" ? KINDS.add : KINDS.place, tgt: box, src: null, ref: 0});
                state.selected = state.elements.length - 1;
            }
            render();
        });

        for (const id of [`${tab}_width`, `${tab}_height`]) {
            const node = app.querySelector(`#${id}`);
            if (node) node.addEventListener("input", fitFrame);
        }
        host.closest(".gradio-accordion, .block")?.addEventListener("click", () => setTimeout(fitFrame, 50));
        fitFrame();
        render();
    }

    onUiLoaded(() => {
        gradioApp().querySelectorAll(".flux3-layout[data-tab]").forEach((host) => {
            if (!host.dataset.ready) {
                host.dataset.ready = "1";
                initPanel(host);
            }
        });
    });
})();
