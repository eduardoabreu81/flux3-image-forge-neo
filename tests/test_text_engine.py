# FLUX 3 Image Klein support for Forge Neo
# The text engine: upstream's template, right padding to a multiple of 80, the padding's rope positions, and the
# (T, 8 x 2560) per-token layout.

import importlib
import unittest

import torch

from tests import forge_stubs


class TestTextEngine(unittest.TestCase):
    def setUp(self):
        self.ctx = forge_stubs.installed(forge_stubs.text_processing())
        self.ctx.__enter__()
        self.te = importlib.import_module("forge_flux3.native.text_engine")
        model = torch.nn.Module()
        model.num_layers = 36
        self.engine = self.te.Flux3TextEngine(model, forge_stubs.FakeHFTokenizer())

    def tearDown(self):
        self.ctx.__exit__(None, None, None)

    def test_encoder_settings(self):
        kwargs = self.engine.text_encoder.kwargs
        self.assertEqual(kwargs["layer"], [4, 8, 12, 16, 20, 24, 28, 32])
        self.assertFalse(kwargs["layer_norm_hidden_state"])
        self.assertTrue(kwargs["enable_attention_masks"])
        self.assertIsInstance(self.engine.text_encoder.transformer, self.te.PadPositions)

    def test_template_and_padding(self):
        self.engine(["a cat"])
        tokens = self.engine.text_encoder.calls[-1]
        text = "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(tokens[: len(text)], [1000 + ord(c) for c in text])
        self.assertEqual(len(tokens), 80)
        self.assertTrue(all(t == self.te.PAD for t in tokens[len(text):]))

    def test_whitespace_is_the_empty_prompt(self):
        self.assertEqual(self.engine.token_ids("  "), self.engine.token_ids(""))

    def test_lengths_round_up_to_80_and_cap_at_4096(self):
        self.assertEqual(self.te.padded_length(1), 80)
        self.assertEqual(self.te.padded_length(80), 80)
        self.assertEqual(self.te.padded_length(81), 160)
        self.assertEqual(self.te.padded_length(5000), 4096)

    def test_per_token_layout(self):
        cond = self.engine(["x"])[0]
        layers, dim = forge_stubs.FakeSDClipModel.layers, forge_stubs.FakeSDClipModel.dim
        self.assertEqual(tuple(cond.shape), (80, layers * dim))
        # token 5: layer 0's features, then layer 1's (100 + 5), ... as upstream's permute(0, 2, 1, 3)
        expected = torch.tensor([[100.0 * layer + 5] * dim for layer in range(layers)]).flatten()
        torch.testing.assert_close(cond[5], expected)

    def test_identical_prompts_encode_once(self):
        out = self.engine(["same", "same"])
        self.assertEqual(len(self.engine.text_encoder.calls), 1)
        self.assertIs(out[0], out[1])

    def test_pad_positions(self):
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        self.assertEqual(self.te.pad_positions(mask).tolist(), [[0, 1, 2, 1, 1]])

    def test_pad_positions_reach_the_model(self):
        seen = {}

        class Model(torch.nn.Module):
            num_layers = 36

            def forward(self, x, attention_mask=None, **kwargs):
                seen.update(kwargs)
                return x

        wrapped = self.te.PadPositions(Model())
        wrapped(None, torch.tensor([[1, 1, 0]]), embeds="e")
        self.assertEqual(seen["position_ids"].tolist(), [[0, 1, 1]])
        self.assertEqual(seen["embeds"], "e")


if __name__ == "__main__":
    unittest.main()
