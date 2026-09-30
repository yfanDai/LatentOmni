from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


def _to_cpu_clone(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach().cpu().clone()


def _get_cache_seq_len(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    try:
        return int(past_key_values[0][0].shape[2])
    except Exception:
        return 0


def resolve_special_token_ids(tokenizer) -> dict[str, int]:
    return {
        "latent_start_id": tokenizer.convert_tokens_to_ids("<Unified_Latent>"),
        "latent_end_id": tokenizer.convert_tokens_to_ids("</Unified_Latent>"),
        "latent_pad_id": tokenizer.convert_tokens_to_ids("<Unified_Latent_pad>"),
        "video_start_id": tokenizer.convert_tokens_to_ids("<|vision_bos|>"),
        "video_end_id": tokenizer.convert_tokens_to_ids("<|vision_eos|>"),
        "video_token_id": tokenizer.convert_tokens_to_ids("<|VIDEO|>"),
        "audio_start_id": tokenizer.convert_tokens_to_ids("<|audio_bos|>"),
        "audio_end_id": tokenizer.convert_tokens_to_ids("<|audio_eos|>"),
        "audio_token_id": tokenizer.convert_tokens_to_ids("<|AUDIO|>"),
    }


def configure_model_special_ids(model, token_ids: dict[str, int]) -> None:
    model.config.latent_token_id = int(token_ids["latent_pad_id"])
    model.config.latent_start_id = int(token_ids["latent_start_id"])
    model.config.latent_end_id = int(token_ids["latent_end_id"])
    model.config.video_token_id = int(token_ids["video_token_id"])
    model.config.video_start_id = int(token_ids["video_start_id"])
    model.config.video_end_id = int(token_ids["video_end_id"])
    model.config.audio_token_id = int(token_ids["audio_token_id"])
    model.config.audio_start_id = int(token_ids["audio_start_id"])
    model.config.audio_end_id = int(token_ids["audio_end_id"])


def build_excluded_token_id_set(tokenizer, token_ids: dict[str, int]) -> set[int]:
    excluded = set(int(x) for x in getattr(tokenizer, "all_special_ids", []))
    # Keep the actual AV content tokens in the denominator/numerator.
    excluded.discard(int(token_ids["video_token_id"]))
    excluded.discard(int(token_ids["audio_token_id"]))
    return excluded


def extract_media_blocks(
    input_ids: torch.Tensor,
    token_ids: dict[str, int],
) -> list[dict[str, Any]]:
    seq = input_ids.tolist()
    blocks: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None

    for idx, token in enumerate(seq):
        if token == token_ids["video_start_id"]:
            current = {
                "start_pos": idx,
                "video_positions": [],
                "audio_positions": [],
            }
            continue

        if current is None:
            continue

        if token == token_ids["video_token_id"]:
            current["video_positions"].append(idx)
        elif token == token_ids["audio_token_id"]:
            current["audio_positions"].append(idx)
        elif token == token_ids["video_end_id"]:
            current["end_pos"] = idx
            blocks.append(current)
            current = None

    return blocks


@dataclass
class AttentionStepRecord:
    step_index: int
    q_len: int
    past_len: int
    attention_len: Optional[int]
    input_ids: Optional[torch.Tensor]
    attention: torch.Tensor
    stage: str


class GenerationAttentionRecorder:
    """
    Capture single-step attention during generation.

    attention_source:
    - `latent`: latent reasoning steps only
    - `text`: ordinary decode text steps only
    - `all`: latent reasoning + ordinary decode text steps
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        latent_start_id: int,
        latent_size: int,
        attention_source: str = "all",
    ):
        self.model = model
        self.latent_start_id = int(latent_start_id)
        self.latent_size = int(latent_size)
        self.attention_source = attention_source

        num_layers = model.model.config.num_hidden_layers
        self.layer_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx

        self._self_attn = model.model.layers[self.layer_idx].self_attn
        self._inner_model = model.model
        self._original_self_attn_forward = self._self_attn.forward
        self._original_inner_forward = self._inner_model.forward

        self._current_meta: Optional[dict[str, Any]] = None
        self._state = "idle"
        self._remaining_steps = 0

        self.call_records: list[dict[str, Any]] = []
        self.step_records: list[AttentionStepRecord] = []

    def install(self) -> None:
        def _patched_self_attn_forward(*args, **kwargs):
            kwargs["output_attentions"] = True
            result = self._original_self_attn_forward(*args, **kwargs)
            attn_weights = result[1]
            if (
                self._current_meta is not None
                and self._current_meta.get("stage") in {"latent_step", "text_step"}
                and attn_weights is not None
            ):
                self._current_meta["attention"] = attn_weights.detach().cpu().float()
            return result

        def _patched_inner_forward(*args, **kwargs):
            input_ids = kwargs.get("input_ids")
            inputs_embeds = kwargs.get("inputs_embeds")
            q_len = 0
            if input_ids is not None:
                q_len = int(input_ids.shape[1])
            elif inputs_embeds is not None:
                q_len = int(inputs_embeds.shape[1])

            maybe_token = None
            if input_ids is not None and input_ids.numel() == 1:
                maybe_token = int(input_ids.reshape(-1)[0].item())

            stage = "other"
            allow_latent = self.attention_source in {"latent", "all"}
            allow_text = self.attention_source in {"text", "all"}

            if allow_latent and maybe_token == self.latent_start_id and q_len == 1:
                stage = "latent_start"
            elif allow_latent and self._state == "collect_steps" and input_ids is None and q_len == 1:
                stage = "latent_step"
            elif allow_latent and self._state == "expect_final":
                stage = "latent_final"
            elif allow_text and q_len == 1 and maybe_token != self.latent_start_id:
                stage = "text_step"

            meta = {
                "stage": stage,
                "q_len": q_len,
                "past_len": _get_cache_seq_len(kwargs.get("past_key_values")),
                "attention_len": int(kwargs["attention_mask"].shape[-1])
                if isinstance(kwargs.get("attention_mask"), torch.Tensor)
                else None,
                "input_ids": _to_cpu_clone(input_ids),
            }

            if stage in {"latent_step", "text_step"}:
                meta["step_index"] = len(self.step_records)

            self._current_meta = meta
            outputs = self._original_inner_forward(*args, **kwargs)
            self._current_meta = None

            self.call_records.append(meta)

            if stage == "latent_start":
                self._state = "collect_steps"
                self._remaining_steps = self.latent_size
            elif stage == "latent_step":
                attention = meta.get("attention")
                if attention is None:
                    raise RuntimeError("Latent step attention was not captured.")
                self.step_records.append(
                    AttentionStepRecord(
                        step_index=int(meta["step_index"]),
                        q_len=int(meta["q_len"]),
                        past_len=int(meta["past_len"]),
                        attention_len=meta["attention_len"],
                        input_ids=meta["input_ids"],
                        attention=attention,
                        stage=stage,
                    )
                )
                self._remaining_steps -= 1
                if self._remaining_steps == 0:
                    self._state = "expect_final"
            elif stage == "text_step":
                attention = meta.get("attention")
                if attention is None:
                    raise RuntimeError("Text decode attention was not captured.")
                self.step_records.append(
                    AttentionStepRecord(
                        step_index=int(meta["step_index"]),
                        q_len=int(meta["q_len"]),
                        past_len=int(meta["past_len"]),
                        attention_len=meta["attention_len"],
                        input_ids=meta["input_ids"],
                        attention=attention,
                        stage=stage,
                    )
                )
            elif stage == "latent_final":
                self._state = "idle"
                self._remaining_steps = 0

            return outputs

        self._self_attn.forward = _patched_self_attn_forward
        self._inner_model.forward = _patched_inner_forward

    def remove(self) -> None:
        self._self_attn.forward = self._original_self_attn_forward
        self._inner_model.forward = self._original_inner_forward


def _flatten_token_ids(x: Optional[torch.Tensor]) -> list[int]:
    if x is None:
        return []
    return [int(v) for v in x.reshape(-1).tolist()]


def summarize_attention_ratio(
    recorder: GenerationAttentionRecorder,
    prefill_input_ids: torch.Tensor,
    tokenizer,
    token_ids: dict[str, int],
    media_index: int = 0,
    excluded_token_ids: Optional[set[int]] = None,
    include_step_details: bool = False,
    include_text_to_latent_as_av: bool = True,
) -> dict[str, Any]:
    if recorder.attention_source not in {"latent", "text", "all"}:
        raise ValueError(f"Unsupported attention_source: {recorder.attention_source}")

    prefill_list = [int(x) for x in prefill_input_ids.reshape(-1).tolist()]
    media_blocks = extract_media_blocks(prefill_input_ids, token_ids)
    if media_index < 0 or media_index >= len(media_blocks):
        raise IndexError(
            f"media_index={media_index} is out of range. Found {len(media_blocks)} media block(s)."
        )
    media_block = media_blocks[media_index]

    av_positions = sorted(media_block["video_positions"] + media_block["audio_positions"])
    excluded_token_ids = (
        build_excluded_token_id_set(tokenizer, token_ids)
        if excluded_token_ids is None
        else set(int(x) for x in excluded_token_ids)
    )

    history: list[Optional[int]] = list(prefill_list)
    total_numerator = 0.0
    total_denominator = 0.0
    total_excluded = 0.0
    total_direct_av = 0.0
    total_text_to_latent = 0.0

    stage_buckets: dict[str, dict[str, float]] = {
        "latent_step": {
            "num": 0.0,
            "den": 0.0,
            "count": 0.0,
            "direct_num": 0.0,
            "text_to_latent_num": 0.0,
        },
        "text_step": {
            "num": 0.0,
            "den": 0.0,
            "count": 0.0,
            "direct_num": 0.0,
            "text_to_latent_num": 0.0,
        },
    }
    step_details: list[dict[str, Any]] = []

    captured_by_index = {int(step.step_index): step for step in recorder.step_records}

    for meta in recorder.call_records:
        stage = meta["stage"]
        if stage == "other":
            continue

        if stage == "latent_start":
            history.extend(_flatten_token_ids(meta["input_ids"]))
            continue

        if stage == "latent_step":
            history.append(None)
        elif stage == "latent_final":
            history.extend(_flatten_token_ids(meta["input_ids"]))
            continue
        elif stage == "text_step":
            history.extend(_flatten_token_ids(meta["input_ids"]))

        if stage not in {"latent_step", "text_step"}:
            continue

        step = captured_by_index.get(int(meta["step_index"]))
        if step is None:
            continue

        row = step.attention[0].mean(dim=0).squeeze(0).cpu().numpy().astype(np.float32)
        key_len = row.shape[0]
        visible_history = history[:key_len]

        valid_mask = np.ones((key_len,), dtype=bool)
        for idx, tok in enumerate(visible_history):
            if tok is not None and int(tok) in excluded_token_ids:
                valid_mask[idx] = False

        av_mask = np.zeros((key_len,), dtype=bool)
        for pos in av_positions:
            if pos < key_len:
                av_mask[pos] = True

        latent_proxy_mask = np.zeros((key_len,), dtype=bool)
        if include_text_to_latent_as_av and stage == "text_step":
            for idx, tok in enumerate(visible_history):
                if tok is None:
                    latent_proxy_mask[idx] = True

        target_mask = av_mask | latent_proxy_mask

        denominator = float(row[valid_mask].sum())
        direct_av_numerator = float(row[av_mask].sum())
        text_to_latent_numerator = float(row[latent_proxy_mask].sum())
        numerator = float(row[target_mask].sum())
        excluded_mass = float(row[~valid_mask].sum())
        ratio = numerator / denominator if denominator > 1e-12 else 0.0

        total_numerator += numerator
        total_denominator += denominator
        total_excluded += excluded_mass
        total_direct_av += direct_av_numerator
        total_text_to_latent += text_to_latent_numerator

        stage_buckets[stage]["num"] += numerator
        stage_buckets[stage]["den"] += denominator
        stage_buckets[stage]["count"] += 1.0
        stage_buckets[stage]["direct_num"] += direct_av_numerator
        stage_buckets[stage]["text_to_latent_num"] += text_to_latent_numerator

        if include_step_details:
            step_details.append(
                {
                    "step_index": int(step.step_index),
                    "stage": stage,
                    "key_len": key_len,
                    "av_attention_mass": numerator,
                    "direct_av_attention_mass": direct_av_numerator,
                    "text_to_latent_attention_mass": text_to_latent_numerator,
                    "valid_attention_mass": denominator,
                    "excluded_attention_mass": excluded_mass,
                    "av_attention_ratio": ratio,
                }
            )

    def _bucket_ratio(stage: str, key: str = "num") -> tuple[float, int]:
        den = float(stage_buckets[stage]["den"])
        count = int(stage_buckets[stage]["count"])
        if den <= 1e-12:
            return 0.0, count
        return float(stage_buckets[stage][key] / den), count

    latent_ratio, latent_count = _bucket_ratio("latent_step")
    text_ratio, text_count = _bucket_ratio("text_step")
    direct_latent_ratio, _ = _bucket_ratio("latent_step", "direct_num")
    direct_text_ratio, _ = _bucket_ratio("text_step", "direct_num")
    text_to_latent_latent_ratio, _ = _bucket_ratio("latent_step", "text_to_latent_num")
    text_to_latent_text_ratio, _ = _bucket_ratio("text_step", "text_to_latent_num")
    all_ratio = total_numerator / total_denominator if total_denominator > 1e-12 else 0.0
    direct_all_ratio = total_direct_av / total_denominator if total_denominator > 1e-12 else 0.0
    text_to_latent_all_ratio = (
        total_text_to_latent / total_denominator if total_denominator > 1e-12 else 0.0
    )

    result = {
        "attention_layer": int(recorder.layer_idx),
        "attention_source": recorder.attention_source,
        "media_index": int(media_index),
        "include_text_to_latent_as_av": bool(include_text_to_latent_as_av),
        "num_total_steps": int(len(recorder.step_records)),
        "num_latent_steps": int(latent_count),
        "num_text_steps": int(text_count),
        "av_attention_ratio_all": float(all_ratio),
        "av_attention_ratio_latent": float(latent_ratio),
        "av_attention_ratio_text": float(text_ratio),
        "av_attention_mass_total": float(total_numerator),
        "direct_av_attention_ratio_all": float(direct_all_ratio),
        "direct_av_attention_ratio_latent": float(direct_latent_ratio),
        "direct_av_attention_ratio_text": float(direct_text_ratio),
        "direct_av_attention_mass_total": float(total_direct_av),
        "text_to_latent_attention_ratio_all": float(text_to_latent_all_ratio),
        "text_to_latent_attention_ratio_latent": float(text_to_latent_latent_ratio),
        "text_to_latent_attention_ratio_text": float(text_to_latent_text_ratio),
        "text_to_latent_attention_mass_total": float(total_text_to_latent),
        "valid_attention_mass_total": float(total_denominator),
        "excluded_attention_mass_total": float(total_excluded),
        "video_token_count": int(len(media_block["video_positions"])),
        "audio_token_count": int(len(media_block["audio_positions"])),
        "excluded_token_ids": sorted(int(x) for x in excluded_token_ids),
    }
    if include_step_details:
        result["step_details"] = step_details
    return result
