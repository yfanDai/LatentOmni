import importlib.util
import math
import os
import sys
import types
import unittest


IMPORT_ERROR = None

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - dependency-gated
    torch = None
    nn = None
    IMPORT_ERROR = exc

MODULE = None
if IMPORT_ERROR is None:
    module_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "latent_model",
        "modeling_qwen2_5_omni_latent.py",
    )
    try:
        spec = importlib.util.spec_from_file_location("latent_modeling_qwen2_5_omni_latent", module_path)
        MODULE = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = MODULE
        spec.loader.exec_module(MODULE)
    except Exception as exc:  # pragma: no cover - dependency-gated
        IMPORT_ERROR = exc
        MODULE = None


@unittest.skipIf(IMPORT_ERROR is not None, f"Missing runtime dependencies: {IMPORT_ERROR}")
class TemporalSyncLossTests(unittest.TestCase):
    def _make_proj(self, hidden_size: int, sync_proj_dim: int):
        return nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, sync_proj_dim),
            nn.GELU(),
            nn.Linear(sync_proj_dim, sync_proj_dim),
        )

    def test_random_inputs_backward_and_projection_grads(self):
        hidden_size = 16
        sync_proj_dim = 12
        video_proj = self._make_proj(hidden_size, sync_proj_dim)
        audio_proj = self._make_proj(hidden_size, sync_proj_dim)
        logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        video_emb = torch.randn(10, hidden_size, requires_grad=True)
        audio_emb = torch.randn(8, hidden_size, requires_grad=True)

        loss = MODULE.temporal_sync_loss(
            video_emb=video_emb,
            audio_emb=audio_emb,
            video_proj=video_proj,
            audio_proj=audio_proj,
            logit_scale=logit_scale,
            max_target_len=6,
        )

        self.assertIsNotNone(loss)
        self.assertEqual(loss.dim(), 0)
        self.assertFalse(torch.isnan(loss))

        loss.backward()

        self.assertIsNotNone(video_emb.grad)
        self.assertIsNotNone(audio_emb.grad)
        for param in video_proj.parameters():
            self.assertIsNotNone(param.grad)
        for param in audio_proj.parameters():
            self.assertIsNotNone(param.grad)
        self.assertIsNotNone(logit_scale.grad)

    def test_detached_inputs_block_raw_embedding_grads(self):
        hidden_size = 16
        sync_proj_dim = 12
        video_proj = self._make_proj(hidden_size, sync_proj_dim)
        audio_proj = self._make_proj(hidden_size, sync_proj_dim)
        logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        video_emb = torch.randn(10, hidden_size, requires_grad=True)
        audio_emb = torch.randn(8, hidden_size, requires_grad=True)

        loss = MODULE.temporal_sync_loss(
            video_emb=video_emb.detach(),
            audio_emb=audio_emb.detach(),
            video_proj=video_proj,
            audio_proj=audio_proj,
            logit_scale=logit_scale,
            max_target_len=6,
        )

        self.assertIsNotNone(loss)
        loss.backward()

        self.assertIsNone(video_emb.grad)
        self.assertIsNone(audio_emb.grad)
        for param in video_proj.parameters():
            self.assertIsNotNone(param.grad)
        for param in audio_proj.parameters():
            self.assertIsNotNone(param.grad)
        self.assertIsNotNone(logit_scale.grad)

    def test_non_detached_inputs_receive_grads(self):
        hidden_size = 16
        sync_proj_dim = 12
        video_proj = self._make_proj(hidden_size, sync_proj_dim)
        audio_proj = self._make_proj(hidden_size, sync_proj_dim)
        logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        video_emb = torch.randn(10, hidden_size, requires_grad=True)
        audio_emb = torch.randn(8, hidden_size, requires_grad=True)

        loss = MODULE.temporal_sync_loss(
            video_emb=video_emb,
            audio_emb=audio_emb,
            video_proj=video_proj,
            audio_proj=audio_proj,
            logit_scale=logit_scale,
            max_target_len=6,
        )

        self.assertIsNotNone(loss)
        loss.backward()

        self.assertIsNotNone(video_emb.grad)
        self.assertIsNotNone(audio_emb.grad)

    def test_positive_pairs_have_lower_loss_than_random_pairs(self):
        hidden_size = 16
        logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        video_emb = torch.randn(8, hidden_size)
        audio_emb_pos = video_emb.clone() + 0.01 * torch.randn_like(video_emb)
        audio_emb_rand = torch.randn_like(audio_emb_pos)

        loss_pos = MODULE.temporal_sync_loss(
            video_emb=video_emb,
            audio_emb=audio_emb_pos,
            video_proj=nn.Identity(),
            audio_proj=nn.Identity(),
            logit_scale=logit_scale,
            max_target_len=8,
        )
        loss_rand = MODULE.temporal_sync_loss(
            video_emb=video_emb,
            audio_emb=audio_emb_rand,
            video_proj=nn.Identity(),
            audio_proj=nn.Identity(),
            logit_scale=logit_scale,
            max_target_len=8,
        )

        self.assertIsNotNone(loss_pos)
        self.assertIsNotNone(loss_rand)
        self.assertLess(loss_pos.item(), loss_rand.item())

    def test_invalid_inputs_return_none(self):
        hidden_size = 16
        logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        invalid_cases = [
            (torch.empty(0, hidden_size), torch.randn(8, hidden_size)),
            (torch.randn(1, hidden_size), torch.randn(8, hidden_size)),
            (torch.randn(8, hidden_size), torch.randn(1, hidden_size)),
            (torch.randn(8, hidden_size), torch.randn(8, hidden_size + 1)),
            (torch.randn(2, 3, hidden_size), torch.randn(8, hidden_size)),
            ("not-a-tensor", torch.randn(8, hidden_size)),
            (torch.randn(8, hidden_size), None),
        ]

        for video_emb, audio_emb in invalid_cases:
            loss = MODULE.temporal_sync_loss(
                video_emb=video_emb,
                audio_emb=audio_emb,
                video_proj=nn.Identity(),
                audio_proj=nn.Identity(),
                logit_scale=logit_scale,
                max_target_len=8,
            )
            self.assertIsNone(loss)


if MODULE is not None:
    class _DummyTextBackbone(nn.Module):
        def __init__(self, hidden_size: int, vocab_size: int):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        def get_input_embeddings(self):
            return self.embed_tokens

        def set_input_embeddings(self, value):
            self.embed_tokens = value

        def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
            return types.SimpleNamespace(
                last_hidden_state=inputs_embeds,
                past_key_values=None,
                attentions=None,
            )


    class _DummyConfig:
        def __init__(self, hidden_size: int, vocab_size: int):
            self.text_config = types.SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
            self.video_start_id = 90
            self.audio_start_id = 91
            self.audio_end_id = 92
            self.video_end_id = 93
            self.video_token_id = 94
            self.audio_token_id = 95
            self.output_attentions = False
            self.output_hidden_states = False
            self.use_return_dict = True
            self.pad_token_id = 0
            self.lambda_sync = 1.0
            self.lambda_alignment = 1.0
            self.detach_sync_inputs = True
            self.sync_max_target_len = 6

        def get_text_config(self):
            return self.text_config


    class _ThinkerShell(nn.Module):
        forward = MODULE.Qwen2_5OmniThinkerForConditionalGeneration.forward
        get_input_embeddings = MODULE.Qwen2_5OmniThinkerForConditionalGeneration.get_input_embeddings
        set_input_embeddings = MODULE.Qwen2_5OmniThinkerForConditionalGeneration.set_input_embeddings
        _ensure_sync_trainable = MODULE.Qwen2_5OmniThinkerForConditionalGeneration._ensure_sync_trainable

        def __init__(self, hidden_size: int = 16, vocab_size: int = 128):
            super().__init__()
            self.config = _DummyConfig(hidden_size, vocab_size)
            self.model = _DummyTextBackbone(hidden_size, vocab_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            self.sync_video_proj = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.sync_audio_proj = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.sync_logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
            self.rope_deltas = None
            self.pad_token_id = 0
            self.spatial_merge_size = 1


@unittest.skipIf(IMPORT_ERROR is not None, f"Missing runtime dependencies: {IMPORT_ERROR}")
class ForwardSyncIntegrationTests(unittest.TestCase):
    def test_forward_sync_loss_populates_loss_dict(self):
        model = _ThinkerShell()
        hidden_size = model.config.text_config.hidden_size

        input_ids = torch.tensor(
            [[
                5,
                90, 91, 94, 94, 95, 95, 92, 93,
                6,
                90, 91, 94, 94, 94, 95, 95, 95, 92, 93,
                7,
            ]],
            dtype=torch.long,
        )
        seq_len = input_ids.shape[1]
        inputs_embeds = torch.randn(1, seq_len, hidden_size)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(seq_len, dtype=torch.long).view(1, 1, seq_len).expand(3, 1, seq_len).clone()

        outputs = model.forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            loss_type=["sync"],
            return_dict=True,
        )

        self.assertIsNotNone(outputs.loss)
        self.assertIn("sync", outputs.loss_dict)
        self.assertIn("sync_logit_scale", outputs.loss_dict)
        self.assertIn("sync_valid_pairs", outputs.loss_dict)
        self.assertEqual(outputs.loss_dict["sync_valid_pairs"].item(), 1)
        self.assertTrue(torch.allclose(outputs.loss, outputs.loss_dict["sync"]))


if __name__ == "__main__":
    unittest.main()
