# FLUX 3 Image Klein support for Forge Neo
# The layout JSON rebuilt after sd-dynamic-prompts strips its braces (the infotext of a real run on Forge Neo).

import json
import unittest

from forge_flux3.native.layout import restore_layout_json

CAPTION = "Minimalist illustration of a runner <silhouette_1> on a chartreuse field <background_1>. "
ORIGINAL = ('[{"id": "background_1", "bbox": [0, 0, 1000, 1000], "desc": "A flat field, fine grain."}, '
            '{"id": "silhouette_1", "bbox": [150, 150, 850, 850], "desc": "A black silhouette, stippled ink."}]')
STRIPPED = ('["id": "background_1", "bbox": [0, 0, 1000, 1000], "desc": "A flat field, fine grain.", '
            '"id": "silhouette_1", "bbox": [150, 150, 850, 850], "desc": "A black silhouette, stippled ink."]')


class TestLayout(unittest.TestCase):
    def test_rebuilds_the_stripped_list(self):
        restored = restore_layout_json(CAPTION + STRIPPED)
        self.assertTrue(restored.startswith(CAPTION))
        self.assertEqual(json.loads(restored[len(CAPTION):]), json.loads(ORIGINAL))

    def test_intact_json_is_unchanged(self):
        self.assertEqual(restore_layout_json(CAPTION + ORIGINAL), CAPTION + ORIGINAL)

    def test_edit_by_box_keys(self):
        stripped = ('Move the lamp <lamp_1> from <ref_image_1>. ["id": "lamp_1", "from": "ref_image_1", '
                    '"src_bbox": [100, 100, 300, 300], "tgt_bbox": null, "desc": "a brass lamp, [old]"]')
        restored = restore_layout_json(stripped)
        element = json.loads(restored[restored.index("["):])[0]
        self.assertEqual(element["tgt_bbox"], None)
        self.assertEqual(element["desc"], "a brass lamp, [old]")

    def test_plain_prompts_are_unchanged(self):
        for prompt in ("a cat", "a [red:blue:0.5] car", '["id" is a word]', 'list ["id": broken'):
            self.assertEqual(restore_layout_json(prompt), prompt)


if __name__ == "__main__":
    unittest.main()
