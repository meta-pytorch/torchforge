# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml

import asyncio
import getpass
import uuid

import torch
import torchx.specs as specs
from forge.actors.policy import Policy

from forge.actors.reference_model import ReferenceModel
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import RLTrainer
from forge.cli.config import parse
from monarch._rust_bindings.monarch_hyperactor.alloc import AllocConstraints, AllocSpec
from monarch._src.actor.meta.allocator import MastAllocator, MastAllocatorConfig

from monarch._src.actor.proc_mesh import ProcMesh
from monarch.tools import commands
from monarch.tools.commands import info, torchx_runner
from monarch.tools.components.meta import hyperactor
from monarch.tools.config import Config, Workspace
from monarch.tools.mesh_spec import ServerSpec
from omegaconf import DictConfig
from torchx.specs import AppState
from torchx.specs.fb.component_helpers import Packages


SCHEDULER_NAME = "mast_conda"
LEARNER_MESH_NAME = "learner"
POLICY_MESH_NAME = "policy"
REF_MESH_NAME = "ref"

USER = getpass.getuser()
WORK_DIR = f"/data/users/{USER}"  # on DEVGPU
WOKR_DIR_MAST = f"/home/{USER}"  # on MAST
EDITABLE_WORKSPACES = ["forge"]

EDITABLE_WORKSPACE_PATHS = [
    f"{WORK_DIR}/{workspace}" for workspace in EDITABLE_WORKSPACES
]


def _add_additional_packages(packages: Packages) -> Packages:
    packages.add_package("oil.oilfs:stable")
    packages.add_package("manifold.manifoldfs")
    return packages


def _build_appdef(cfg) -> specs.AppDef:

    # create the app definition for the worker
    remote_work_dir = "/packages/monarch_default_workspace/workspace/"
    REMOTE_END_PYTHONPATH = ":".join(
        [f"{remote_work_dir}{workspace}" for workspace in EDITABLE_WORKSPACE_PATHS]
    )

    default_envs = {
        **hyperactor.DEFAULT_NVRT_ENVS,
        **hyperactor.DEFAULT_NCCL_ENVS,
        **hyperactor.DEFAULT_TORCH_ENVS,
        **{"TORCHX_RUN_PYTHONPATH": f"{REMOTE_END_PYTHONPATH}:{remote_work_dir}"},
    }

    packages = Packages()
    sku = "gtt_any"
    meshes = []
    for mesh_name, config in cfg["services"].items():
        num_replicas = config["num_replicas"]
        with_gpus = bool(config["with_gpus"])
        num_hosts = int(config.get("hosts", 0))
        # Create list of mesh names with indices and num_hosts
        if with_gpus and num_hosts > 0:
            mesh_list = [
                f"{mesh_name}_{i}:{num_hosts}:{sku}" for i in range(num_replicas)
            ]
            meshes.extend(mesh_list)

    appdef = hyperactor.host_mesh_conda(
        meshes=meshes,
        additional_packages=_add_additional_packages(packages),
        timeout_sec=1 * 60 * 60,  # Kill the job if idle for 1 hour
        env=default_envs,
    )

    for role in appdef.roles:
        role.resource.capabilities["server_sub_types"] = [
            # role.resource.capabilities["server_sub_types"][2]  # hardcoded to ROCE
            role.resource.capabilities["server_sub_types"][1]  # hardcoded to ROCE
        ]

    return appdef


def create_job_name():
    return f"rithesh-forge-grpo-{uuid.uuid4().hex[:6]}"


def create_server_handle(job_name: str) -> str:
    return f"{SCHEDULER_NAME}:///{job_name}"


def compute_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    context_length = logits.shape[1] - input_ids.shape[1]
    logits = logits[:, context_length - 1 : -1]
    logprobs = torch.log_softmax(logits / temperature, dim=-1).to(input_ids.device)
    logprobs = torch.gather(logprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
    return logprobs


def simple_grpo_loss(
    logits: torch.Tensor,
    response: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    padding_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    logprobs = compute_logprobs(logits, response)
    kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
    per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
    per_token_loss = -(per_token_policy_loss - beta * kl)
    loss = (
        ((per_token_loss * padding_mask).sum(dim=1))
        / (padding_mask.sum(dim=1).clamp(min=1.0))
    ).mean()
    return loss


async def main(cfg: DictConfig):
    """Main GRPO training loop with rollout and training processes."""

    import debugpy

    debugpy.listen(5681)
    print("[MAIN] Waiting for VS Code debugger to attach...")
    debugpy.wait_for_client()
    print("Attached!")

    # # await asyncio.sleep(0)

    # # job_name = "rithesh-forge-grpo-9dc76e"
    # _job_name = create_job_name()
    # handle = create_server_handle(_job_name)
    # server_spec = info(handle)
    # if server_spec and server_spec.state == AppState.RUNNING:
    #     print(f"Job {_job_name} is already running. Skipping launch.")
    #     return server_spec

    # config = Config(
    #     scheduler="mast_conda",
    #     scheduler_args={
    #         # NOTE: default config. Use args to set your own values
    #         "hpcIdentity": "genai_llm_pretraining_data",
    #         "hpcJobOncall": "monarch",
    #         "hpcClusterUuid": "MastProdCluster",
    #         "rmAttribution": "pytorch4all_clients_approved",
    #         # "hpcClusterUuid": "MastGenAICluster",
    #         # "rmAttribution": "gen_ai_llama_systems_training",
    #         # "localityConstraints": ["region", "pci"],
    #     },
    #     appdef=_build_appdef(cfg),
    #     workspace=Workspace(
    #         dirs=[workspace_dir for workspace_dir in EDITABLE_WORKSPACE_PATHS],
    #     ),
    # )

    # await commands.get_or_create(_job_name, config)

    # check_interval_seconds = 3
    # from datetime import datetime

    # start = datetime.now()

    # # This should run pretty fast in seconds
    # while True:
    #     server_spec = info(handle)

    #     if not server_spec:  # server not found
    #         await asyncio.sleep(check_interval_seconds)
    #         continue

    #     # We need to make sure a job has been submitted before detaching the client.
    #     if server_spec.state < AppState.PENDING:  # UNSUBMITTED or SUBMITTED
    #         print(
    #             f"Waiting for {handle} to be {AppState.PENDING} (current: {server_spec.state}); "
    #             f"will check again in {check_interval_seconds} seconds. "
    #             f"Total wait time: {datetime.now() - start}",
    #             end="\r",
    #         )
    #         await asyncio.sleep(check_interval_seconds)
    #     else:
    #         break

    # print(f"\nJob {_job_name} has launched. Detached the client now.")
    # print("I am here")

    # (policy,) = await asyncio.gather(
    #     Policy.options(**cfg.services.policy).as_service(**cfg.policy),
    # )
    # prompt = "What is 3+5?"
    # completions = await policy.generate.choose(prompt=prompt)
    # for completion in completions:
    #     print(completion)

    # rl_trainer = (
    #     await RLTrainer.options(**cfg.services.trainer).as_service(
    #         **cfg.trainer, loss=simple_grpo_loss
    #     ),
    # )

    ref_model = (
        await ReferenceModel.options(**cfg.services.ref_model).as_service(
            **cfg.ref_model
        ),
    )

    print("All services initialized successfully!")


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
