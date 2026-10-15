# FLUX 3 Image Klein support for Forge Neo
# detect() on the real checkpoints' headers (key names and shapes of black-forest-labs/flux-3-image-klein-comfy,
# in tests/fixtures), and the module Forge builds from it carrying exactly their weights.

import json
import unittest
from pathlib import Path

import torch

from forge_flux3.native.checkpoints import DROPPED_SUFFIXES
from forge_flux3.native.detect import detect
from forge_flux3.native.transformer import Flux3DiT

FIXTURES = Path(__file__).parent / "fixtures"
QUANT_SUFFIXES = (".comfy_quant", ".weight_scale", ".weight_scale_2", ".input_scale")


def header(name):
    shapes = json.loads((FIXTURES / f"{name}.shapes.json").read_text())
    return {k: {"shape": v} for k, v in shapes.items()}


def built(config):
    with torch.device("meta"):
        return Flux3DiT(**config).state_dict()


class TestDetect(unittest.TestCase):
    def check(self, name, flow_map, fused):
        state = header(name)
        config = detect(state)
        self.assertEqual(config["image_model"], "flux3")
        self.assertEqual((config["hidden_size"], config["num_heads"]), (3072, 24))
        self.assertEqual((config["depth"], config["depth_single_blocks"]), (5, 28))
        self.assertEqual((config["mlp_hidden"], config["in_channels"]), (9216, 128))
        self.assertEqual(config["flow_map"], flow_map)
        self.assertEqual(config["fused_qkv"], fused)
        module = built(config)
        weights = {k for k in state if not k.endswith(QUANT_SUFFIXES + DROPPED_SUFFIXES)}
        self.assertEqual(set(module), weights)
        return state, module

    def test_base_bf16(self):
        state, module = self.check("flux3_image_klein_bf16", flow_map=False, fused=False)
        for key, tensor in module.items():
            self.assertEqual(tuple(tensor.shape), tuple(state[key]["shape"]), key)

    def test_distilled_bf16(self):
        state, module = self.check("flux3_image_klein_sd_bf16", flow_map=True, fused=False)
        for key, tensor in module.items():
            self.assertEqual(tuple(tensor.shape), tuple(state[key]["shape"]), key)

    def test_distilled_int8_convrot(self):
        self.check("flux3_image_klein_sd_int8_convrot", flow_map=True, fused=True)

    def test_distilled_nvfp4(self):
        self.check("flux3_image_klein_sd_nvfp4", flow_map=True, fused=False)


if __name__ == "__main__":
    unittest.main()
