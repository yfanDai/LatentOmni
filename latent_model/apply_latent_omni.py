import importlib.util, sys, pathlib, os
patch_path = pathlib.Path(__file__).with_name("modeling_qwen2_5_omni_latent.py")
spec  = importlib.util.spec_from_file_location(
    "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni",
    patch_path,
)
patched_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patched_mod)

sys.modules["transformers.models.qwen2_5_omni.modeling_qwen2_5_omni"] = patched_mod

print("Replaced the original Qwen2.5-Omni model with the Latent version.")