# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for forge.actors.coder.SandboxedPythonCoder.
"""
import os
import tempfile
import uuid
from unittest.mock import Mock, patch

import pytest

from forge.actors.coder import _SandboxedPythonCoder


@pytest.mark.asyncio
async def test_coder_success():
    """Test successful execution."""
    unique_id = str(uuid.uuid4())[:8]
    container_name = f"test_sandbox_{unique_id}"

    with tempfile.NamedTemporaryFile(suffix=".sqsh", delete=False) as temp_image:
        image_path = temp_image.name

    def mock_subprocess_run(*args, **kwargs):
        """Mock subprocess.run for testing."""
        cmd = args[0] if args else kwargs.get("args", [])
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)

        if "import" in cmd_str:
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result
        elif "remove" in cmd_str:
            result = Mock()
            result.returncode = 0
            return result
        elif "create" in cmd_str:
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result
        elif "start" in cmd_str:
            result = Mock()
            result.returncode = 0
            result.stdout = "Hello World\n"
            result.stderr = ""
            return result
        else:
            raise ValueError(f"Unexpected subprocess call: {cmd_str}")

    try:
        with patch(
            "forge.actors.coder.subprocess.run", side_effect=mock_subprocess_run
        ):
            coder = _SandboxedPythonCoder(
                docker_image="docker://python:3.10",
                sqsh_image_path=image_path,
                container_name=container_name,
            )

            await coder.setup()
            result, _ = await coder.execute(code="print('Hello World')")
            assert result == "Hello World\n"
    finally:
        if os.path.exists(image_path):
            os.unlink(image_path)


@pytest.mark.asyncio
async def test_coder_execution_failure():
    """Test execution failure."""
    unique_id = str(uuid.uuid4())[:8]
    container_name = f"test_sandbox_{unique_id}"

    with tempfile.NamedTemporaryFile(suffix=".sqsh", delete=False) as temp_image:
        image_path = temp_image.name

    def mock_subprocess_run(*args, **kwargs):
        """Mock subprocess.run for testing."""
        cmd = args[0] if args else kwargs.get("args", [])
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)

        if "import" in cmd_str:
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result
        elif "remove" in cmd_str:
            result = Mock()
            result.returncode = 0
            return result
        elif "create" in cmd_str:
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result
        elif "start" in cmd_str:
            result = Mock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "SyntaxError: invalid syntax"
            return result
        else:
            raise ValueError(f"Unexpected subprocess call: {cmd_str}")

    try:
        with patch(
            "forge.actors.coder.subprocess.run", side_effect=mock_subprocess_run
        ):
            coder = _SandboxedPythonCoder(
                docker_image="docker://python:3.10",
                sqsh_image_path=image_path,
                container_name=container_name,
            )

            await coder.setup()
            output, err = await coder.execute(code="invalid syntax")
            assert "SyntaxError" in err
    finally:
        if os.path.exists(image_path):
            os.unlink(image_path)
