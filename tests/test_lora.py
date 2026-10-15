# FLUX 3 Image Klein support for Forge Neo
# The LoRA key map: trainer prefixes on the split projections, and q/k/v LoRAs onto their third of the INT8
# build's fused qkv.

import unittest

import torch

from forge_flux3.native.lora import flux3_lora_keys


class TestLoraKeys(unittest.TestCase):
    def test_split_projections(self):
        keys = flux3_lora_keys({"diffusion_model.single_blocks.0.q_proj.weight": torch.zeros(8, 8)})
        target = "diffusion_model.single_blocks.0.q_proj.weight"
        for alias in ("single_blocks.0.q_proj", "transformer.single_blocks.0.q_proj",
                      "diffusion_model.single_blocks.0.q_proj", "lora_unet_single_blocks_0_q_proj"):
            self.assertEqual(keys[alias], target)

    def test_fused_qkv_thirds(self):
        keys = flux3_lora_keys({"diffusion_model.txt_mode_blocks.2.qkv.weight": torch.zeros(9, 4)})
        target = "diffusion_model.txt_mode_blocks.2.qkv.weight"
        self.assertEqual(keys["transformer.txt_mode_blocks.2.q_proj"], (target, (0, 0, 3)))
        self.assertEqual(keys["txt_mode_blocks.2.k_proj"], (target, (0, 3, 3)))
        self.assertEqual(keys["lora_unet_txt_mode_blocks_2_v_proj"], (target, (0, 6, 3)))
        self.assertNotIn("txt_mode_blocks.2.qkv", keys)

    def test_ignores_non_weights(self):
        self.assertEqual(flux3_lora_keys({"diffusion_model.single_blocks.0.norm.query_norm.scale": torch.ones(2),
                                          "first_stage_model.x.weight": torch.ones(2)}), {})


if __name__ == "__main__":
    unittest.main()
