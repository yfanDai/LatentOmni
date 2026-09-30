# sitecustomize.py
# -----------------------------------------------------------------------
# This patch is only for RL codes (vllm==0.8.5), not for SFT and inference.
# So use `MONET_RL_PATCH=1` only in the RL script.
# -----------------------------------------------------------------------

import os, sys, importlib, inspect

print(f"[sitecustomize] imported from {__file__}", file=sys.stderr)

def patch_qwen_monet():
    # Import official and Monet implementations
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as q_official
    import monet_models.transformers.monet_modeling_qwen2_5_vl as q_monet

    off_cls = q_official.Qwen2_5_VLForConditionalGeneration
    mon_cls = q_monet.Qwen2_5_VLForConditionalGeneration

    # Debug: check signatures before patch
    try:
        print(
            "[Monet RL patch] official forward sig before:",
            inspect.signature(off_cls.forward),
            file=sys.stderr,
        )
        print(
            "[Monet RL patch] monet    forward sig:",
            inspect.signature(mon_cls.forward),
            file=sys.stderr,
        )
    except Exception:
        pass

    # In-place monkey patch: copy methods from Monet class to official class
    off_cls.forward = mon_cls.forward
    #off_cls.__init__ = mon_cls.__init__

    # Debug: check signature after patch
    try:
        print(
            "[Monet RL patch] official forward sig after:",
            inspect.signature(off_cls.forward),
            file=sys.stderr,
        )
    except Exception:
        pass

    print(
        "[Monet RL patch] Patched methods of Qwen2_5_VLForConditionalGeneration in-place",
        file=sys.stderr,
    )


def patch():
    print("[sitecustomize] patch() called", file=sys.stderr)

    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"

    workspace = os.path.abspath(".")
    old_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
    os.environ["LATENT_START_ID"] = "151666"
    os.environ["LATENT_END_ID"] = "151667"
    os.environ["AVT_LATENT_HOOK_BIN"] = "1"

    sys.modules["vllm.v1.worker.gpu_model_runner"] = importlib.import_module(
        "monet_models.vllm.monet_gpu_model_runner"
    )

    patch_qwen_monet()
    '''sys.modules[
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"
    ] = importlib.import_module(
        "monet_models.transformers.monet_modeling_qwen2_5_vl"
    )'''

    print("[Monet RL patch] vllm & transformers patched", file=sys.stderr)

patch()
