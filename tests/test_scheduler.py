# FLUX 3 Image Klein support for Forge Neo
# The FLUX.3 scheduler against upstream's schedule, the shift read back from Forge's sigma grid, and the flow
# map's next-sigma lookup.

import types
import unittest

import torch

from forge_flux3.native import flowmap, scheduler
from forge_flux3.native.checkpoints import MISSING_TE, MISSING_VAE, check_text_encoder, check_vae


def upstream_timesteps(steps, alpha):
    # src/flux3/sampling.timesteps
    t = torch.linspace(1.0, 0.0, steps + 1)
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def forge_grid(shift, multiplier=1.0, timesteps=1000):
    # backend/modules/k_prediction.PredictionDiscreteFlow.set_parameters with time_snr_shift
    t = (torch.arange(1, timesteps + 1) / timesteps) * multiplier / multiplier
    return shift * t / (1 + (shift - 1) * t)


class TestScheduler(unittest.TestCase):
    def test_matches_upstream(self):
        for steps in (16, 50, 4):
            torch.testing.assert_close(scheduler.schedule(steps), upstream_timesteps(steps, 6.93))

    def test_ends_at_zero(self):
        self.assertEqual(float(scheduler.sigmas(16)[-1]), 0.0)
        self.assertEqual(float(scheduler.sigmas(16)[0]), 1.0)

    def test_shift_read_back_from_the_model_grid(self):
        for shift in (6.93, 3.0, 1.0):
            model = types.SimpleNamespace(sigmas=forge_grid(shift))
            self.assertAlmostEqual(scheduler.shift_of(model), shift, places=4)
        self.assertEqual(scheduler.shift_of(None), scheduler.SHIFT)


class TestFlowMap(unittest.TestCase):
    def tearDown(self):
        flowmap.clear()

    def test_next_sigma(self):
        flowmap.record(scheduler.schedule(16))
        sched = scheduler.schedule(16).tolist()
        for i in range(16):
            self.assertEqual(flowmap.target(sched[i]), sched[i + 1])
        # Forge hands the model the sigma in the computation dtype
        self.assertEqual(flowmap.target(float(torch.tensor(sched[3]).half())), sched[4])

    def test_img2img_truncation_keeps_the_lookup(self):
        sched = scheduler.schedule(16)
        flowmap.record(sched)
        self.assertAlmostEqual(flowmap.target(float(sched[10])), float(sched[11]))

    def test_without_schedule_jumps_to_zero(self):
        flowmap.clear()
        self.assertEqual(flowmap.target(0.5), 0.0)


class TestChecks(unittest.TestCase):
    def test_text_encoder(self):
        keys = {f"model.language_model.layers.{i}.self_attn.q_proj.weight": {"shape": [4096, 2560]} for i in range(36)}
        keys["model.language_model.layers.0.self_attn.q_norm.weight"] = {"shape": [128]}
        keys["model.language_model.layers.0.input_layernorm.weight"] = {"shape": [2560]}
        check_text_encoder(keys)
        keys["model.language_model.layers.0.input_layernorm.weight"] = {"shape": [4096]}
        self.assertRaises(ValueError, check_text_encoder, keys)
        with self.assertRaisesRegex(ValueError, MISSING_TE[:20]):
            check_text_encoder({})

    def test_vae(self):
        with self.assertRaisesRegex(ValueError, MISSING_VAE[:20]):
            check_vae(None)
        check_vae({str(i): None for i in range(20)})


if __name__ == "__main__":
    unittest.main()
