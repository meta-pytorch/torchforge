# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import Metadata as DcpMeta

KEY_DELIM = "."


@dataclass
class DcpHandle:
    checkpoint_id: str = ""
    metadata: DcpMeta | None = None


def load_tensor_from_dcp(handle: DcpHandle, param_name) -> torch.Tensor:
    tensor_meta = handle.metadata.state_dict_metadata[param_name]
    buffer = torch.empty(tensor_meta.size, dtype=tensor_meta.properties.dtype)
    dcp.load(checkpoint_id=handle.checkpoint_id, state_dict={param_name: buffer})
    return buffer


def get_param_prefix(policy_version: int) -> str:
    return f"policy_ver_{policy_version}"


def get_param_key(policy_version: int, name: str) -> str:
    return f"policy_ver_{policy_version}{KEY_DELIM}{name}"


def extract_param_name(key: str) -> str:
    return KEY_DELIM.join(key.split(KEY_DELIM)[1:])
