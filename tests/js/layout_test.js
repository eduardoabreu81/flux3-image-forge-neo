// FLUX 3 Image Klein support for Forge Neo
// The FLUX 3 Layout panel's JSON: Black Forest Labs' exact format, written into and read back from the prompt.
// Run by tests/test_layout_panel.py (node tests/js/layout_test.js).

const assert = require("node:assert/strict");
const path = require("node:path");
const L = require(path.join(__dirname, "..", "..", "javascript", "flux3_layout.js"));

const tests = {
    "layout element in Black Forest Labs' style"() {
        const json = L.buildLayout([{id: "obj_1", desc: "a red balloon", kind: "place", tgt: [80, 60, 480, 380]}]);
        assert.equal(json, '[{"id": "obj_1", "bbox": [80, 60, 480, 380], "desc": "a red balloon"}]');
    },

    "box edit as the official cactus example"() {
        const json = L.buildLayout([
            {id: "obj_1", desc: "a small potted cactus", kind: "add", tgt: [550, 620, 850, 850]},
            {id: "mug_1", desc: "the white mug", kind: "ref", ref: 0, src: [354, 283, 768, 755], tgt: [354, 283, 768, 755]},
        ]);
        assert.equal(json, '[{"id": "obj_1", "from": "target", "src_bbox": null, "tgt_bbox": [550, 620, 850, 850], ' +
            '"desc": "a small potted cactus"}, {"id": "mug_1", "from": "ref_image_0", "src_bbox": [354, 283, 768, 755], ' +
            '"tgt_bbox": [354, 283, 768, 755], "desc": "the white mug"}]');
    },

    "remove has a null target and keeps the reference index"() {
        const [o] = JSON.parse(L.buildLayout([{id: "lamp", desc: "", kind: "remove", ref: 2, src: [1, 2, 300, 400], tgt: [5, 5, 50, 50]}]));
        assert.deepEqual(o, {id: "lamp", from: "ref_image_2", src_bbox: [1, 2, 300, 400], tgt_bbox: null, desc: ""});
    },

    "incomplete elements are left out, and drags drawn backwards are normalized"() {
        const json = L.buildLayout([
            {id: "a", desc: "", kind: "place", tgt: null},
            {id: "b", desc: "", kind: "ref", ref: 0, src: [1, 1, 9, 9], tgt: null},
            {id: "c", desc: "", kind: "place", tgt: [900, 800, 100, 1200]},
        ]);
        assert.deepEqual(JSON.parse(json), [{id: "c", bbox: [100, 800, 900, 1000], desc: ""}]);
        assert.equal(L.buildLayout([]), "");
    },

    "descriptions keep quotes, commas and brackets"() {
        const desc = 'a sign reading "OPEN, 24h" [neon]';
        const [o] = JSON.parse(L.buildLayout([{id: "sign", desc, kind: "place", tgt: [0, 0, 10, 10]}]));
        assert.equal(o.desc, desc);
    },

    "writes at the end, then replaces the same list"() {
        const caption = "A still life <a_1>.";
        const once = L.writeLayout(caption, '[{"id": "a_1", "bbox": [0, 0, 500, 500], "desc": ""}]');
        assert.equal(once, 'A still life <a_1>. [{"id": "a_1", "bbox": [0, 0, 500, 500], "desc": ""}]');
        const twice = L.writeLayout(once, '[{"id": "a_1", "bbox": [1, 1, 2, 2], "desc": ""}]');
        assert.equal(twice, 'A still life <a_1>. [{"id": "a_1", "bbox": [1, 1, 2, 2], "desc": ""}]');
        assert.equal(L.writeLayout(twice, ""), caption);
    },

    "replaces a list sd-dynamic-prompts stripped"() {
        const stripped = 'Caption <a_1>. ["id": "a_1", "bbox": [0, 0, 500, 500], "desc": "x"] and more';
        assert.equal(L.writeLayout(stripped, '[{"id": "a_1", "bbox": [0, 0, 9, 9], "desc": "x"}]'),
            'Caption <a_1>. [{"id": "a_1", "bbox": [0, 0, 9, 9], "desc": "x"}] and more');
    },

    "leaves prompt editing syntax alone"() {
        assert.equal(L.findLayoutSpan("a [red:blue:0.5] car"), null);
        assert.equal(L.writeLayout("a [red:blue:0.5] car", ""), "a [red:blue:0.5] car");
    },

    "reads the official cactus prompt back"() {
        const prompt = 'Add a small potted cactus <obj_1>. [{"id": "obj_1", "from": "target", "src_bbox": null, ' +
            '"tgt_bbox": [550, 620, 850, 850], "desc": "A small potted cactus."}, {"id": "mug_1", "from": "ref_image_0", ' +
            '"src_bbox": [354, 283, 768, 755], "tgt_bbox": null, "desc": "A mug."}]';
        const els = L.readLayout(prompt);
        assert.equal(els.length, 2);
        assert.equal(els[0].kind, "add");
        assert.deepEqual(els[0].tgt, [550, 620, 850, 850]);
        assert.equal(els[1].kind, "remove");
        assert.equal(els[1].ref, 0);
        assert.deepEqual(JSON.parse(L.buildLayout(els)), JSON.parse(prompt.slice(prompt.indexOf("["))));
    },

    "reads a layout back to the same JSON"() {
        const json = '[{"id": "teapot_1", "bbox": [80, 60, 420, 400], "desc": "A red teapot."}, ' +
            '{"id": "vase_1", "bbox": [480, 600, 950, 940], "desc": "A blue vase."}]';
        assert.equal(L.buildLayout(L.readLayout("Still life. " + json)), json);
    },
};

let failed = 0;
for (const [name, fn] of Object.entries(tests)) {
    try {
        fn();
        console.log(`ok   ${name}`);
    } catch (e) {
        failed++;
        console.log(`FAIL ${name}\n${e.message}`);
    }
}
console.log(`${Object.keys(tests).length - failed}/${Object.keys(tests).length} passed`);
process.exit(failed ? 1 : 0);
