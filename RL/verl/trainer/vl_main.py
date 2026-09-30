"""
Pure VL entry point for RL training — no Monet / latent model patch needed.

Usage:
    python -m verl.trainer.vl_main config=examples/config_vl.yaml ...

This file is identical to verl/trainer/main.py except:
1. It installs a no-op `monet_rl_patch` shim so that the downstream imports in
   fsdp_workers.py and ray_trainer.py don't crash when the Monet patch is absent.
2. It does NOT set LATENT_START_ID / LATENT_END_ID / AVT_LATENT_HOOK_BIN
   environment variables, so the vLLM engine runs in standard (non-latent) mode.
"""

import sys
import types

# ---------------------------------------------------------------------------
# Install a no-op shim for monet_rl_patch BEFORE any other import so that
# `import monet_rl_patch` in fsdp_workers.py / ray_trainer.py is a no-op.
# ---------------------------------------------------------------------------
_shim = types.ModuleType("monet_rl_patch")
_shim.patch = lambda: None
sys.modules["monet_rl_patch"] = _shim

# ---------------------------------------------------------------------------
# The rest is identical to verl/trainer/main.py
# (We re-implement it here rather than monkey-patching the original so that
#  the original Monet training path is not disturbed.)
# ---------------------------------------------------------------------------

import json
import os

import ray
from omegaconf import OmegaConf

from verl.single_controller.ray import RayWorkerGroup
from verl.utils.tokenizer import get_processor, get_tokenizer
from verl.workers.fsdp_workers import FSDPWorker
from verl.workers.reward import (
    BatchFunctionRewardManager,
    BatchFunctionRuleBasedJudgeManager,
    SequentialFunctionRewardManager,
    SingleFunctionRuleBasedJudgeManager,
)
from verl.trainer.config import PPOConfig
from verl.trainer.data_loader import create_dataloader, create_vl_dataloader
from verl.trainer.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


@ray.remote(num_cpus=2)
class Runner:
    """A runner for VL RL training (no Monet patch)."""

    def run(self, config: PPOConfig):
        import torch, sys
        print("Torch version :", torch.__version__)
        print("Torch path    :", torch.__file__)
        print("Python exec   :", sys.executable)
        print("CUDA_VISIBLE_DEVICES :", os.environ.get("CUDA_VISIBLE_DEVICES"))

        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
            Role.RefPolicy: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        if config.worker.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.worker.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")

        if config.worker.rule_based_judge.judge_type == "single":
            RuleBasedJudgeManager = SingleFunctionRuleBasedJudgeManager
        elif config.worker.rule_based_judge.judge_type == "batch":
            RuleBasedJudgeManager = BatchFunctionRuleBasedJudgeManager
        else:
            raise NotImplementedError(f"Unknown judge type {config.worker.rule_based_judge.judge_type}.")

        RemoteRewardManager = ray.remote(RewardManager).options(
            num_cpus=config.worker.reward.num_cpus
        )
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        RemoteRuleBasedJudgeManager = ray.remote(RuleBasedJudgeManager).options(
            num_cpus=config.worker.rule_based_judge.num_cpus,
            name="rule_based_judge_server",
        )
        config.worker.rule_based_judge.judge_server_name = "rule_based_judge_server"
        rule_based_judge_fn = RemoteRuleBasedJudgeManager.remote(
            config.worker.rule_based_judge, tokenizer
        )

        train_dataloader, val_dataloader = create_vl_dataloader(config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            rule_based_judge=rule_based_judge_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "PYTHONUNBUFFERED": "1",
                "RAY_DEBUG": os.getenv("RAY_DEBUG", "0"),
                "RAY_LOG_TO_STDERR": os.getenv("RAY_LOG_TO_STDERR", "0"),
            }
        }
        local_mode = os.getenv("RAY_LOCAL_MODE", "0").lower() in ("1", "true", "yes")
        _force_local = os.getenv("USE_RAY_LOCAL", "1").lower() in ("1", "true", "yes")
        address = "local" if _force_local else os.getenv("RAY_ADDRESS", None)

        default_spill = os.path.join(os.getcwd(), "ray_spill")
        default_temp = os.path.join(os.getcwd(), "ray_tmp")
        spill_dir = os.path.abspath(os.getenv("RAY_SPILL_DIR", default_spill))
        temp_dir = os.path.abspath(os.getenv("RAY_TMPDIR", default_temp))
        os.makedirs(spill_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        object_store_mem = int(os.getenv("RAY_OBJECT_STORE_MEMORY", str(128 * 1024 ** 2)))
        register_timeout = int(os.getenv("RAY_WORKER_REGISTER_TIMEOUT_SECONDS", "300"))

        if local_mode:
            os.environ.update(
                {"RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"}
            )

        advertised_cpus = int(os.getenv("RAY_NUM_CPUS", "16"))
        advertised_gpus = int(
            os.getenv("RAY_NUM_GPUS", str(getattr(ppo_config.trainer, "n_gpus_per_node", 1)))
        )

        ray.init(
            address=address,
            runtime_env=runtime_env,
            local_mode=local_mode,
            include_dashboard=False,
            _temp_dir=temp_dir,
            object_store_memory=object_store_mem,
            object_spilling_directory=spill_dir,
            _system_config={"worker_register_timeout_seconds": register_timeout},
            num_cpus=advertised_cpus,
            num_gpus=advertised_gpus,
        )

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))


if __name__ == "__main__":
    main()
