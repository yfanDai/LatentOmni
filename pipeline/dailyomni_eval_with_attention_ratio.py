import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import math
import multiprocessing as mp
import re
import sys
import time
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from latent_model import apply_latent_omni  # noqa: F401 - patch local HF module
from transformers import (
    AutoProcessor,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from pipeline.attention_ratio_utils import (
    GenerationAttentionRecorder,
    build_excluded_token_id_set,
    configure_model_special_ids,
    resolve_special_token_ids,
    summarize_attention_ratio,
)
from src.omni_utils import process_mm_info


def extract_boxed_content(text: str) -> str:
    boxed_pattern = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
    matches = boxed_pattern.findall(text)
    if matches:
        return "".join(matches)
    return text


def parse_single_choice_response(text: str) -> str:
    """
    Lightweight local fallback for multiple-choice parsing.

    Priority:
    1. boxed answer
    2. explicit "answer is X" style patterns
    3. first standalone A/B/C/D option token
    """
    if text is None:
        return ""

    candidate = extract_boxed_content(text).strip()
    upper = candidate.upper()

    explicit_patterns = [
        r"\bANSWER\s*(?:IS|:)?\s*\(?([A-D])\)?\b",
        r"\bOPTION\s*([A-D])\b",
        r"\bCHOICE\s*([A-D])\b",
        r"^\(?([A-D])\)?[\.\s,:-]",
        r"\(([A-D])\)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)

    standalone = re.findall(r"\b([A-D])\b", upper)
    if standalone:
        return standalone[0]

    return candidate.strip()


def _build_conversation(input_modality: str, file_path: str, prompt: str, sys_prompt: str):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": sys_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": input_modality, input_modality: file_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _prepare_inputs(conversation, processor, model):
    use_audio_in_video = True
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    return inputs, videos, use_audio_in_video


def _configure_worker_runtime(args):
    os.environ["OMP_NUM_THREADS"] = str(args.torch_num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.torch_num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.torch_num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.torch_num_threads)
    os.environ["TORCHCODEC_NUM_THREADS"] = str(args.torchcodec_num_threads)

    torch.set_num_threads(max(1, int(args.torch_num_threads)))
    try:
        torch.set_num_interop_threads(max(1, int(args.torch_num_interop_threads)))
    except RuntimeError:
        # `set_num_interop_threads` can only be called once per process.
        pass


def chat_and_measure(
    input_modality,
    file_path,
    prompt,
    sys_prompt,
    model,
    processor,
    latent_size,
    attention_layer,
    attention_source,
    attention_media_index,
    attention_include_step_details,
    token_ids,
    excluded_token_ids,
    preprocess_semaphore,
):
    conversation = _build_conversation(input_modality, file_path, prompt, sys_prompt)
    if preprocess_semaphore is None:
        inputs, videos, use_audio_in_video = _prepare_inputs(conversation, processor, model)
    else:
        preprocess_semaphore.acquire()
        try:
            inputs, videos, use_audio_in_video = _prepare_inputs(conversation, processor, model)
        finally:
            preprocess_semaphore.release()

    # Keep the original latent evaluation behavior. If latent_size == 0, the recorder
    # will still capture text steps and latent steps will naturally be absent.
    inputs["latent_mode"] = True
    inputs["latent_size"] = latent_size

    recorder = GenerationAttentionRecorder(
        model=model,
        layer_idx=attention_layer,
        latent_start_id=token_ids["latent_start_id"],
        latent_size=latent_size,
        attention_source=attention_source,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    e2e_start = time.perf_counter()
    recorder.install()
    try:
        with torch.no_grad():
            text_ids = model.generate(
                **inputs,
                use_audio_in_video=use_audio_in_video,
                temperature=0.4,
                do_sample=False,
                max_new_tokens=512,
                use_cache=True,
                eos_token_id=[
                    processor.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                    processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                ],
            )
    finally:
        recorder.remove()

    torch.cuda.synchronize()
    e2e_end = time.perf_counter()
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    e2e_latency_ms = (e2e_end - e2e_start) * 1000

    try:
        generated_text = processor.batch_decode(
            text_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
    except Exception as e:
        print(f"[Decode Error] Data: {text_ids}")
        raise e

    print("-------test--------\n")
    print(generated_text)
    model_generation = extract_boxed_content(generated_text)

    attention_stats = summarize_attention_ratio(
        recorder=recorder,
        prefill_input_ids=inputs["input_ids"][0].detach().cpu(),
        tokenizer=processor.tokenizer,
        token_ids=token_ids,
        media_index=attention_media_index,
        excluded_token_ids=excluded_token_ids,
        include_step_details=attention_include_step_details,
        include_text_to_latent_as_av=True,
    )

    metrics = {
        "e2e_latency_ms": round(e2e_latency_ms, 2),
        "peak_memory_gb": round(peak_memory_gb, 4),
        "n_frames": int(videos[0].shape[0]) if videos else 0,
    }

    return model_generation, generated_text, metrics, attention_stats


def _load_model_and_processor(args, gpu_id):
    device_map = {"": f"cuda:{gpu_id}"}
    torch.cuda.set_device(gpu_id)

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    processor.tokenizer.add_tokens("<Unified_Latent_pad>", special_tokens=True)
    processor.tokenizer.add_tokens("<Unified_Latent>", special_tokens=True)
    processor.tokenizer.add_tokens("</Unified_Latent>", special_tokens=True)

    config = Qwen2_5OmniThinkerConfig.from_pretrained(args.model_path)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()

    token_ids = resolve_special_token_ids(processor.tokenizer)
    configure_model_special_ids(model, token_ids)
    answer_start_pattern = processor.tokenizer(
        "<|im_start|>assistant", return_tensors="pt"
    )["input_ids"][0]
    model.config.answer_start_pattern = answer_start_pattern.tolist()
    excluded_token_ids = build_excluded_token_id_set(processor.tokenizer, token_ids)

    return model, processor, token_ids, excluded_token_ids


def worker_proc(rank, gpu_id, args, task_chunk, out_path, preprocess_semaphore):
    sys_prompt = "You are a helpful AVQA expert."
    _configure_worker_runtime(args)

    print(f"[Worker-{rank}] Loading Latent Omni model and processor...")
    try:
        model, processor, token_ids, excluded_token_ids = _load_model_and_processor(args, gpu_id)
    except Exception as e:
        print(f"[Worker-{rank}] Model load failed: {e}")
        traceback.print_exc()
        return

    with open(out_path, "w", encoding="utf-8") as fout:
        for item in tqdm(task_chunk, desc=f"Worker-{rank}[GPU-{gpu_id}]"):
            video_path = item["video_path"]
            prompt = item["raw_data"].get("prompt", "")
            video_id = item["video_id"]
            ground_truth = item["raw_data"].get("Answer", "")

            try:
                response, raw_response, metrics, attention_stats = chat_and_measure(
                    "video",
                    video_path,
                    prompt,
                    sys_prompt,
                    model,
                    processor,
                    args.latent_size,
                    args.attention_layer,
                    args.attention_source,
                    args.attention_media_index,
                    args.attention_include_step_details,
                    token_ids,
                    excluded_token_ids,
                    preprocess_semaphore,
                )

                response_normalized = parse_single_choice_response(response)
                is_correct = response_normalized.lower() == ground_truth.lower()

                out_data = dict(item["raw_data"])
                out_data["output"] = response
                out_data["raw_output"] = raw_response
                out_data["judge"] = is_correct
                out_data["video_path"] = video_path
                out_data["metrics"] = metrics
                out_data["attention_stats"] = attention_stats

                fout.write(json.dumps(out_data, ensure_ascii=False) + "\n")
                fout.flush()
            except Exception as e:
                print(f"[Worker-{rank}] Error on {video_id}: {e}")
                traceback.print_exc()

    print(f"[Worker-{rank}] Done. Results -> {out_path}")


def run_multi_gpu(args):
    print(f"Loading data from {args.qa_path}...")
    try:
        with open(args.qa_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    task_list = []
    missing_count = 0

    for item in qa_data:
        vid = item["video_id"]
        video_path = os.path.join(args.video_base_dir, vid, f"{vid}_video.mp4")

        if os.path.exists(video_path):
            task_list.append(
                {
                    "video_id": vid,
                    "video_path": video_path,
                    "raw_data": item,
                }
            )
        else:
            missing_count += 1

    print(f"Total tasks loaded: {len(task_list)}. Missing videos: {missing_count}")

    num_gpus = args.num_gpus
    chunk_size = math.ceil(len(task_list) / num_gpus) if num_gpus > 0 else len(task_list)
    chunk_size = max(chunk_size, 1)
    chunks = [task_list[i : i + chunk_size] for i in range(0, len(task_list), chunk_size)]

    processes = []
    tmp_files = []
    preprocess_limit = args.max_parallel_preprocess
    if preprocess_limit is not None and preprocess_limit > 0:
        preprocess_semaphore = mp.Semaphore(preprocess_limit)
    else:
        preprocess_semaphore = None

    out_dir = os.path.dirname(args.fout_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for rank, chunk in enumerate(chunks):
        if not chunk:
            continue
        if rank >= num_gpus:
            break

        gpu_id = rank % num_gpus
        tmp_out = args.fout_path.replace(".jsonl", f".part{rank}.jsonl")
        tmp_files.append(tmp_out)

        process = mp.Process(
            target=worker_proc,
            args=(rank, gpu_id, args, chunk, tmp_out, preprocess_semaphore),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    print("Merging results...")
    with open(args.fout_path, "w", encoding="utf-8") as fout:
        for tmp in tmp_files:
            if os.path.exists(tmp):
                with open(tmp, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
                os.remove(tmp)

    print(f"All done. Saved to {args.fout_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="DailyOmni evaluation with AV attention-ratio tracing for Latent Omni"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="path/to/model",
    )
    parser.add_argument(
        "--video_base_dir",
        type=str,
        default="path/to/videos",
    )
    parser.add_argument(
        "--qa_path",
        type=str,
        default="path/to/data.json",
    )
    parser.add_argument("--fout_path", type=str, required=True, help="Path to save output jsonl")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--latent_size", type=int, default=0, help="Latent size to inject into the inputs")
    parser.add_argument("--attention_layer", type=int, default=-1, help="Decoder layer used for attention tracing")
    parser.add_argument(
        "--attention_source",
        type=str,
        choices=["latent", "text", "all"],
        default="all",
        help="Which generation steps to include in attention tracing",
    )
    parser.add_argument(
        "--attention_media_index",
        type=int,
        default=0,
        help="Which input media block to measure when multiple videos exist",
    )
    parser.add_argument(
        "--attention_include_step_details",
        action="store_true",
        help="Whether to dump per-step ratios into the output JSONL",
    )
    parser.add_argument(
        "--torch_num_threads",
        type=int,
        default=1,
        help="Per-process intra-op CPU threads for PyTorch preprocessing",
    )
    parser.add_argument(
        "--torch_num_interop_threads",
        type=int,
        default=1,
        help="Per-process inter-op CPU threads for PyTorch preprocessing",
    )
    parser.add_argument(
        "--torchcodec_num_threads",
        type=int,
        default=1,
        help="FFmpeg/torchcodec worker threads used during video decoding",
    )
    parser.add_argument(
        "--max_parallel_preprocess",
        type=int,
        default=2,
        help="Upper bound for how many worker processes can run CPU video preprocessing simultaneously",
    )

    args = parser.parse_args()
    run_multi_gpu(args)
