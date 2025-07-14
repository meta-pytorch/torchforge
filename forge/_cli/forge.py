# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from forge._cli.download import Download
from forge._cli.run import Run


class ForgeCLIParser:
    """Holds all information related to running the CLI"""

    def __init__(self):
        # Initialize the top-level parser
        self._parser = argparse.ArgumentParser(
            prog="forge",
            description="Welcome to the torchforge CLI!",
            add_help=True,
        )
        # Default command is to print help
        self._parser.set_defaults(func=lambda args: self._parser.print_help())

        # Add subcommands
        subparsers = self._parser.add_subparsers(title="subcommands")
        Download.create(subparsers)
        Run.create(subparsers)

    def parse_args(self) -> argparse.Namespace:
        """Parse CLI arguments"""
        return self._parser.parse_args()

    def run(self, args: argparse.Namespace) -> None:
        """Execute CLI"""
        args.func(args)


def main():
    parser = ForgeCLIParser()
    args = parser.parse_args()
    parser.run(args)


if __name__ == "__main__":
    main()
