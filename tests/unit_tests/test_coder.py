# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for forge.actors.coder.SandboxedCoder.
"""
import os
import tempfile
import uuid
from unittest.mock import Mock, patch

import pytest
from forge.actors.coder import SandboxedCoder

from monarch.actor import this_proc


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_coder_execution():
    """Tests basic coder execution with mocked enroot."""
    unique_id = str(uuid.uuid4())[:8]
    container_name = f"test_sandbox_{unique_id}"

    with tempfile.NamedTemporaryFile(suffix=".sqsh", delete=False) as temp_image:
        image_path = temp_image.name

    coder = None
    try:
        with patch("subprocess.run") as mock_run:

            def mock_subprocess_run(*args, **kwargs):
                # Figure out which call this is based on the command
                cmd = args[0]
                if "import" in cmd:
                    result = Mock()
                    result.returncode = 0
                    result.stderr = ""
                    return result
                elif "remove" in cmd:
                    result = Mock()
                    result.returncode = 0
                    return result
                elif "create" in cmd:
                    result = Mock()
                    result.returncode = 0
                    result.stderr = ""
                    return result
                elif "start" in cmd:
                    result = Mock()
                    result.returncode = 0
                    result.stdout = "hello world\n"
                    result.stderr = ""
                    print(f"Mock execute result: stdout = {repr(result.stdout)}")
                    return result
                else:
                    raise ValueError(f"Unexpected subprocess call: {cmd}")

            mock_run.side_effect = mock_subprocess_run

            coder = this_proc().spawn(
                "coder",
                SandboxedCoder,
                "docker://python:3.10",
                image_path,
                container_name,
            )
            await coder.setup.call_one()

            # Execute code (this will trigger more mocked subprocess calls)
            results = await coder.execute.call_one(
                code="print('hello world')",
            )

            # Verify the result
            assert results == "hello world\n"

    finally:
        # Clean up resources
        if coder:
            await SandboxedCoder.shutdown(coder)

        if os.path.exists(image_path):
            os.unlink(image_path)
