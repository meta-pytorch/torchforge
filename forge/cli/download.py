# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import json
import os
import textwrap
import traceback

from pathlib import Path

from forge.cli.subcommand import Subcommand

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

# TODO: update this
REPO_ID_FNAME = "original_repo_id"


class Download(Subcommand):
    """Holds all the logic for the `forge download` subcommand."""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self._parser = subparsers.add_parser(
            "download",
            prog="forge download",
            usage="forge download <repo-id> [OPTIONS]",
            help="Download a model from the Hugging Face Hub.",
            description="Download a model from the Hugging Face Hub.",
            epilog=textwrap.dedent(
                """\
            examples:
                # Download a model from the Hugging Face Hub with a Hugging Face API token
                $ forge download meta-llama/Llama-2-7b-hf --hf-token <TOKEN>
                Successfully downloaded model repo and wrote to the following locations:
                /tmp/Llama-2-7b-hf/config.json
                /tmp/Llama-2-7b-hf/README.md
                /tmp/Llama-2-7b-hf/consolidated.00.pth
                ...

                # Download an ungated model from the Hugging Face Hub
                $ forge download mistralai/Mistral-7B-Instruct-v0.2 --output-dir /tmp/model
                Successfully downloaded model repo and wrote to the following locations:
                /tmp/model/config.json
                /tmp/model/README.md
                /tmp/model/model-00001-of-00002.bin
                ...

            For a list of all models, visit the Hugging Face Hub
            https://huggingface.co/models.
            """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self._parser.set_defaults(func=self._download_cmd)

    def _add_arguments(self) -> None:
        """Add arguments to the parser."""
        self._parser.add_argument(
            "repo_id",
            type=str,
            help="Name of the repository on Hugging Face Hub.",
        )
        self._parser.add_argument(
            "--output-dir",
            type=Path,
            required=False,
            default=None,
            help="Directory in which to save the model. Defaults to `/tmp/<model_name>`.",
        )
        self._parser.add_argument(
            "--hf-token",
            type=str,
            required=False,
            default=os.getenv("HF_TOKEN", None),
            help="Hugging Face API token. Needed for gated models like Llama2.",
        )
        self._parser.add_argument(
            "--ignore-patterns",
            type=str,
            required=False,
            help="If provided, files matching any of the patterns are not downloaded. Example: '*.safetensors'. "
            "Only supported for Hugging Face Hub models.",
        )

    def _download_cmd(self, args: argparse.Namespace) -> None:
        return self._download_from_huggingface(args)

    def _download_from_huggingface(self, args: argparse.Namespace) -> None:
        """Downloads a model from the Hugging Face Hub."""
        # Download the tokenizer and PyTorch model files

        # Default output_dir is `/tmp/<model_name>`
        output_dir = args.output_dir
        if output_dir is None:
            model_name = args.repo_id.split("/")[-1]
            output_dir = Path("/tmp") / model_name

        print(f"Ignoring files matching the following patterns: {args.ignore_patterns}")
        try:
            true_output_dir = snapshot_download(
                args.repo_id,
                local_dir=output_dir,
                ignore_patterns=args.ignore_patterns,
                token=args.hf_token,
            )
        except GatedRepoError:
            if args.hf_token:
                self._parser.error(
                    "It looks like you are trying to access a gated repository. Please ensure you "
                    "have access to the repository."
                )
            else:
                self._parser.error(
                    "It looks like you are trying to access a gated repository. Please ensure you "
                    "have access to the repository and have provided the proper Hugging Face API token "
                    "using the option `--hf-token` or by running `huggingface-cli login`."
                    "You can find your token by visiting https://huggingface.co/settings/tokens"
                )
        except RepositoryNotFoundError:
            self._parser.error(
                f"Repository '{args.repo_id}' not found on the Hugging Face Hub."
            )
        except Exception as e:
            tb = traceback.format_exc()
            msg = f"Failed to download {args.repo_id} with error: '{e}' and traceback: {tb}"
            self._parser.error(msg)

        # save the repo_id. This is necessary because the download step is a separate command
        # from the rest of the CLI. When saving a model adapter, we have to add the repo_id
        # to the adapter config.
        # TODO: this needs to be updated when we start using HF cache
        file_path = os.path.join(true_output_dir, REPO_ID_FNAME + ".json")
        with open(file_path, "w") as json_file:
            json.dump({"repo_id": args.repo_id}, json_file, indent=4)

        print(
            "Successfully downloaded model repo and wrote to the following locations:",
            *list(Path(true_output_dir).iterdir()),
            sep="\n",
        )
