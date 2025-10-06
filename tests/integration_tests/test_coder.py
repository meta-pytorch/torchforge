# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for forge.actors.coder.SandboxedPythonCoder.

Requires enroot to be installed.

"""

import os
import uuid

import pytest

from forge.actors.coder import SandboxedPythonCoder


@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_coder_runs_python():
    """Integration test for SandboxedPythonCoder with real container execution."""
    # Create unique names to avoid test conflicts
    unique_id = str(uuid.uuid1())
    container_name = f"test_sandbox_{unique_id}"
    image_path = f"/tmp/python_test_{unique_id}.sqsh"

    coder = None
    try:
        coder = await SandboxedPythonCoder.as_actor(
            docker_image="docker://python:3.10",
            sqsh_image_path=image_path,
            container_name=container_name,
        )

        # Execute code
        results, _ = await coder.execute.call_one(
            code="print('hello world')",
        )
        print("Got results", results)
        assert results == "hello world\n"

    finally:
        # Clean up resources
        if coder:
            await SandboxedPythonCoder.shutdown(coder)

        # Clean up the image file
        if os.path.exists(image_path):
            os.unlink(image_path)


@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_coder_catches_error():
    """Integration test for SandboxedPythonCoder with real container execution."""
    # Create unique names to avoid test conflicts
    unique_id = str(uuid.uuid1())
    container_name = f"test_sandbox_{unique_id}"
    image_path = f"/tmp/python_test_{unique_id}.sqsh"

    coder = None
    try:
        print("starting test")
        coder = await SandboxedPythonCoder.as_actor(
            docker_image="docker://python:3.10",
            sqsh_image_path=image_path,
            container_name=container_name,
        )
        print("Got coder")

        # Execute code
        _, stderr = await coder.execute.call_one(
            code="hello world",
        )
        print("got stderr", stderr)
        assert "SyntaxError" in stderr

    finally:
        # Clean up resources
        if coder:
            await SandboxedPythonCoder.shutdown(coder)

        # Clean up the image file
        if os.path.exists(image_path):
            os.unlink(image_path)
