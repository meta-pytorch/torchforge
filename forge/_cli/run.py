# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import runpy
import sys
import textwrap

from pathlib import Path
from typing import Optional

import forge
from forge._cli.subcommand import Subcommand

from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.run import get_args_parser as get_torchrun_args_parser, run

ROOT = Path(forge.__file__).parent.parent


class Run(Subcommand):
    """Holds all the logic for the `forge run` subcommand."""

    def __init__(self, subparsers):
        super().__init__()
        self._parser = subparsers.add_parser(
            "run",
            prog="forge run",
            help="Run a recipe. For distributed recipes, this supports all torchrun arguments.",
            description="Run a recipe. For distributed recipes, this supports all torchrun arguments.",
            usage="forge run [TORCHRUN-OPTIONS] <recipe> --config <config> [RECIPE-OPTIONS]",
            epilog=textwrap.dedent(
                """\
                examples:

                    # Run SFT recipe with default values
                    $ forge run --nproc_per_node 4 apps/sft/sft.py --config apps/sft/configs/llama3_8b.yaml
                """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self._parser.set_defaults(func=self._run_cmd)

    def _add_arguments(self) -> None:
        """Add arguments to the parser.

        This is a bit hacky since we need to add the torchrun arguments to our parser.
        This grabs the argparser from torchrun, iterates over it's actions, and adds them
        to our parser. We rename the training_script and training_script_args to recipe and recipe_args
        respectively. In addition, we leave out the help argument since we add it manually to ours.
        """
        torchrun_argparser = get_torchrun_args_parser()
        for action in torchrun_argparser._actions:
            if action.dest == "training_script":
                action.dest = "recipe"
                action.help = """Path to recipe to be launched followed by args."""
            elif action.dest == "training_script_args":
                action.dest = "recipe_args"
                action.help = "Args to be passed to the recipe."
            elif action.dest == "help":
                continue
            self._parser._add_action(action)

    @record
    def _run_distributed(self, args: argparse.Namespace):
        """Run a recipe with torchrun."""
        print("Running with torchrun...")
        # Have to reset the argv so that the recipe can be run with the correct arguments
        args.training_script = args.recipe
        args.training_script_args = args.recipe_args

        # If the user does not explicitly pass a rendezvous endpoint, run in standalone mode.
        # This allows running multiple distributed training jobs simultaneously.
        if not args.rdzv_endpoint:
            args.standalone = True

        args.module = True
        run(args)

    def _convert_to_dotpath(self, recipe_path: str) -> str:
        """Convert a custom recipe path to a dot path that can be run as a module.

        Args:
            recipe_path (str): The path of the recipe.

        Returns:
            The dot path of the recipe.
        """
        filepath, _ = os.path.splitext(recipe_path)
        return filepath.replace("/", ".")

    def _run_cmd(self, args: argparse.Namespace):
        """Run a recipe."""
        # We have to assume that the recipe supports distributed training
        supports_distributed = True
        recipe_path, config_path = None, None

        # Try to find config string in args
        try:
            config_idx = args.recipe_args.index("--config") + 1
            config_str = args.recipe_args[config_idx]
        except ValueError:
            self._parser.error("The '--config' argument is required.")

        # Get recipe path
        recipe_path = self._convert_to_dotpath(args.recipe)

        # Get config path
        config_path = config_str

        # Prepare args
        args.recipe = recipe_path
        args.recipe_args[config_idx] = config_path

        # Make sure user code in current directory is importable
        sys.path.append(os.getcwd())

        self._run_distributed(args)
