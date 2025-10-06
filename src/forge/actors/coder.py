# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from monarch.actor import endpoint

from forge.controller import ForgeActor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SandboxedPythonCoder(ForgeActor):
    """A sandboxed code execution environment using enroot containers.

    This is a proof of concept of using enroot to provided a sandboxed
    environment for executing Python code using NVIDIA's enroot technology.

    It automatically manages the entire container lifecycle including image
    import, container creation, and cleanup.

    The actor follows a three-stage workflow:
    1. Image Management: Automatically imports Docker images to enroot .sqsh format
    2. Container Lifecycle: Creates fresh container instances for isolated execution
    3. Code Execution: Safely runs Python code with proper error handling and output capture

    Dependencies:
    - enroot: NVIDIA's container runtime (must be installed on host)
    - Docker images: Accessible via docker:// URLs or local paths
    - Python 3.x: For the container environment

    Args:
        docker_image: Docker image URL to import (e.g., "docker://python:3.10").
                        Can be any Docker Hub image or custom registry URL.
        sqsh_image_path: Local filesystem path where the enroot .sqsh image will be stored.
                        If the file doesn't exist, it will be created via enroot import.
        container_name: Unique name for the enroot container instance. Used for
                        container lifecycle management (create/remove operations).

    """

    def __init__(
        self,
        docker_image: str = "docker://python:3.10",
        sqsh_image_path: str = "python-image.sqsh",
        container_name: str = "sandbox",
    ):
        self.docker_image = docker_image
        self.sqsh_image_path = sqsh_image_path
        self.container_name = container_name
        self._initialized = False

    @endpoint
    async def setup(self):
        logging.debug("Setting up sandboxed actor")
        await self._maybe_create_image()
        self._reset()

    @endpoint
    async def reset(self):
        """Resets the container instance from the base image."""
        self._reset()

    async def _maybe_create_image(self):
        """Ensure the enroot image exists, import it if necessary."""
        if not os.path.exists(self.sqsh_image_path):
            logging.debug(
                f"Image {self.sqsh_image_path} not found, importing from {self.docker_image}"
            )
            result = subprocess.run(
                ["enroot", "import", "-o", self.sqsh_image_path, self.docker_image],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to import image: {result.stderr}")
            logging.debug(
                f"Successfully imported {self.docker_image} to {self.sqsh_image_path}"
            )
        else:
            logging.info(f"Using existing image: {self.sqsh_image_path}")

    def _reset(self):
        """(Re)create a clean container instance from the base image."""
        # Remove any old container
        logging.debug(f"Removing container {self.container_name}")
        subprocess.run(
            ["enroot", "remove", "-f", self.container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Create new container from image
        result = subprocess.run(
            ["enroot", "create", "--name", self.container_name, self.sqsh_image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logging.debug(f"Container creation result: {result}")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to reset container: {result.stderr}")
        self._initialized = True
        logging.debug("Successfully initialized container")

    @endpoint
    async def execute(self, code: str) -> tuple[str, str]:
        """Executes Python code inside the container and returns the output.

        Args:
            code: Python source code string to execute.

        Returns:
            The captured stdout and stderr from the execution.
        """
        logging.debug(f"Executing {code}")
        if not self._initialized:
            raise RuntimeError("Container not initialized. Call reset() first.")

        # Write code to a temporary file that we can mount
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / "script.py"
            code_path.write_text(code)

            # Run the code inside the container, mounting tmpdir
            cmd = [
                "enroot",
                "start",
                "--mount",
                f"{tmpdir}:/work",
                self.container_name,
                "python3",
                "/work/script.py",
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            output = result.stdout
            error = result.stderr
            return output, error
