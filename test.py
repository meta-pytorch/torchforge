# Copyright (c) Meta Platforms, Inc.
# All rights reserved.
#
# Minimal repro: Provisioner host_mesh_from_proc() UID mapping bug
#
# Run this with:
#   python -m forge.tests.test_provisioner_uid_mapping

import asyncio

# import pytest

from forge.controller.provisioner import (
    get_or_create_provisioner,
    get_proc_mesh,
    stop_proc_mesh,
)
from forge.types import ProcessConfig


# @pytest.mark.asyncio
async def test_provisioner_host_mesh_lookup_uid_mapping():
    prov = await get_or_create_provisioner()
    pm = await get_proc_mesh(
        ProcessConfig(procs=1, with_gpus=False, hosts=None, mesh_name="uid_repro")
    )
    # UID is attached locally by the helper
    assert hasattr(pm, "_uid") and pm._uid, "missing _uid on returned ProcMesh"
    print(f"✅ got ProcMesh with UID {pm._uid}")
    hm = await prov.host_mesh_from_proc.call_one(pm._uid)  # if pass pm, _uid is None
    assert hm is not None
    await stop_proc_mesh(pm)
    print("✅ repro passed")


if __name__ == "__main__":
    asyncio.run(test_provisioner_host_mesh_lookup_uid_mapping())
