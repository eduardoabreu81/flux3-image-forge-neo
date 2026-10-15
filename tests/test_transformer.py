# FLUX 3 Image Klein support for Forge Neo
# The DiT against upstream's own (tests/reference/flux3_dit.py) at a small width: same weights, same inputs,
# same velocity. Also the Forge build path: Flux3DiT(**detect(checkpoint)) loads the checkpoint strict.

import unittest

import torch

from forge_flux3.native import flowmap
from forge_flux3.native import transformer as ours
from forge_flux3.native.detect import detect
from tests.reference import flux3_dit as upstream

HIDDEN, HEADS, DEPTH, SINGLE = 256, 2, 2, 3  # head_dim 128: upstream's rope axes (32, 32, 32, 32)


def upstream_dit(flow_map):
    upstream.HIDDEN, upstream.HEADS, upstream.DEPTH, upstream.DEPTH_SINGLE = HIDDEN, HEADS, DEPTH, SINGLE
    torch.manual_seed(0)
    dit = upstream.DiT(flow_map)
    with torch.no_grad():
        for p in dit.parameters():
            p.normal_(0.0, 0.05)
        for name, p in dit.named_parameters():
            if name.endswith(".scale"):
                p.uniform_(0.5, 1.5)
    return dit.eval()


def ours_from(reference):
    dit = ours.Flux3DiT(**detect(reference.state_dict()))
    dit.load_state_dict(reference.state_dict(), strict=True)
    return dit.eval()


def fuse_qkv(state):
    fused = {}
    for k, v in state.items():
        if k.endswith(".q_proj.weight"):
            base = k[: -len(".q_proj.weight")]
            fused[f"{base}.qkv.weight"] = torch.cat(
                [state[f"{base}.{p}.weight"] for p in ("q_proj", "k_proj", "v_proj")])
        elif not k.endswith((".k_proj.weight", ".v_proj.weight")):
            fused[k] = v
    return fused


def upstream_velocity(dit, latent, context, sigma, refs=(), target=None):
    """Upstream's packing for one sample: [text, image_cond, image] tokens with their ids."""
    _, c, h, w = latent.shape
    image = latent[0].flatten(1).T
    ids = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), -1).reshape(-1, 2)

    def scaled(rows_cols, index):
        out = torch.zeros(rows_cols.shape[0], 4)
        out[:, 1:3] = rows_cols.float() * 0.5
        out[:, 3] = index
        return out

    cond_tokens, cond_ids = [], []
    for i, ref in enumerate(refs):
        rh, rw = ref.shape[-2:]
        cond_tokens.append(ref[0].flatten(1).T)
        rc = torch.stack(torch.meshgrid(torch.arange(rh), torch.arange(rw), indexing="ij"), -1).reshape(-1, 2)
        cond_ids.append(scaled(rc, -(1 + i)))
    text = context[0]
    text_ids = torch.zeros(text.shape[0], 4)
    text_ids[:, 3] = torch.arange(text.shape[0])
    data = {"image": image, "text": text,
            "image_cond": torch.cat(cond_tokens) if refs else torch.zeros(0, c)}
    pack_ids = {"image": scaled(ids, 0), "text": text_ids,
                "image_cond": torch.cat(cond_ids) if refs else torch.zeros(0, 4)}
    timesteps = {m: torch.zeros(data[m].shape[0]) for m in upstream.MODES}
    timesteps["image"].fill_(sigma)
    packed = upstream.Packed(data=data, ids=pack_ids, timesteps=timesteps,
                             mode_indices={m: [(0, data[m].shape[0])] for m in upstream.MODES})
    with torch.no_grad():
        out = dit(packed, timesteps_ref=None if target is None else torch.tensor([target]))
    return out.T.reshape(1, c, h, w)


class TestTransformer(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.latent = torch.randn(1, 128, 4, 6)
        self.context = torch.randn(1, 10, 20480)
        self.refs = [torch.randn(1, 128, 3, 4), torch.randn(1, 128, 2, 2)]
        flowmap.clear()

    def assert_close(self, a, b):
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4)

    def test_detect_reads_the_configuration(self):
        config = detect(upstream_dit(False).state_dict())
        self.assertEqual(config["hidden_size"], HIDDEN)
        self.assertEqual(config["num_heads"], HEADS)
        self.assertEqual((config["depth"], config["depth_single_blocks"]), (DEPTH, SINGLE))
        self.assertEqual(config["mlp_hidden"], HIDDEN * 3)
        self.assertFalse(config["flow_map"])
        self.assertFalse(config["fused_qkv"])
        self.assertTrue(detect(upstream_dit(True).state_dict())["flow_map"])
        self.assertIsNone(detect({"double_blocks.0.img_attn.norm.key_norm.scale": torch.zeros(1)}))

    def test_base_matches_upstream(self):
        reference = upstream_dit(False)
        mine = ours_from(reference)
        # 0.75 and 0.5 are exact in bfloat16, the precision upstream embeds the time in
        expected = upstream_velocity(reference, self.latent, self.context, 0.75)
        with torch.no_grad():
            got = mine.run(self.latent, torch.tensor([0.75]), self.context)
        self.assert_close(got, expected)

    def test_references_match_upstream(self):
        reference = upstream_dit(False)
        mine = ours_from(reference)
        expected = upstream_velocity(reference, self.latent, self.context, 0.5, refs=self.refs)
        with torch.no_grad():
            got = mine.run(self.latent, torch.tensor([0.5]), self.context, self.refs)
        self.assert_close(got, expected)

    def test_flow_map_matches_upstream(self):
        reference = upstream_dit(True)
        mine = ours_from(reference)
        flowmap.record(torch.tensor([1.0, 0.75, 0.5, 0.0]))
        expected = upstream_velocity(reference, self.latent, self.context, 0.75, refs=self.refs[:1], target=0.5)
        with torch.no_grad():
            got = mine.run(self.latent, torch.tensor([0.75]), self.context, self.refs[:1])
        self.assert_close(got, expected)

    def test_fused_qkv_build_matches_split(self):
        reference = upstream_dit(False)
        split = ours_from(reference)
        state = fuse_qkv(reference.state_dict())
        config = detect(state)
        self.assertTrue(config["fused_qkv"])
        fused = ours.Flux3DiT(**config)
        fused.load_state_dict(state, strict=True)
        with torch.no_grad():
            a = split.run(self.latent, torch.tensor([0.5]), self.context)
            b = fused.eval().run(self.latent, torch.tensor([0.5]), self.context)
        self.assert_close(a, b)

    def test_batch_samples_stay_independent(self):
        mine = ours_from(upstream_dit(False))
        latent = torch.cat([self.latent, torch.randn_like(self.latent)])
        context = torch.cat([self.context, torch.randn_like(self.context)])
        with torch.no_grad():
            both = mine.run(latent, torch.tensor([0.5, 0.5]), context, self.refs)
            first = mine.run(latent[:1], torch.tensor([0.5]), context[:1], self.refs)
        self.assert_close(both[:1], first)


if __name__ == "__main__":
    unittest.main()
