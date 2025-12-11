# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import functools
import sys
from argparse import Namespace
from typing import Any, Callable

from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

from omegaconf import DictConfig, OmegaConf

# Add support for summing lists of numbers, e.g. ${sum:${max_req_tokens},${max_res_tokens}}
OmegaConf.register_new_resolver("sum", lambda *args: sum(args), replace=True)

# Add support for boolean negation, e.g. ${not:${compile}}
OmegaConf.register_new_resolver("not", lambda x: not x, replace=True)


def _has_component(node: Any) -> bool:
    """Check if a node has a _component_ field."""
    return (OmegaConf.is_dict(node) or isinstance(node, dict)) and "_component_" in node


def _remove_key_by_dotpath(nested_dict: dict[str, Any], dotpath: str) -> None:
    """
    Removes a key specified by dotpath from a nested dict. Errors should be handled by
    the calling function.

    Args:
        nested_dict (dict[str, Any]): nested dict to remove key from
        dotpath (str): dotpath of key to remove, e.g., "a.b.c"
    """
    path = dotpath.split(".")

    def delete_non_component(d: dict[str, Any], key: str) -> None:
        if _has_component(d[key]):
            raise ValueError(
                f"Removing components from CLI is not supported: ~{dotpath}"
            )
        del d[key]

    # Traverse to the parent of the final key
    current = nested_dict
    for key in path[:-1]:
        current = current[key]

    # Delete the final key
    delete_non_component(current, path[-1])


# TODO: this is all just a copy-paste hack for now
def _merge_yaml_and_cli_args(yaml_args: Namespace, cli_args: list[str]) -> DictConfig:
    """
    Takes the direct output of argparse's parse_known_args which returns known
    args as a Namespace and unknown args as a dotlist (in our case, yaml args and
    cli args, respectively) and merges them into a single OmegaConf DictConfig.

    If a cli arg overrides a yaml arg with a _component_ field, the cli arg can
    be specified with the parent field directly, e.g., model=my_module.models.my_model
    instead of model._component_=my_module.models.my_model. Nested fields within the
    component should be specified with dot notation, e.g., model.lora_rank=16.

    Example:
        >>> config.yaml:
        >>>     a: 1
        >>>     b:
        >>>       _component_: my_module.models.my_model
        >>>       c: 3

        >>> python main.py --config config.yaml b=my_module.models.other_model b.c=4
        >>> yaml_args, cli_args = parser.parse_known_args()
        >>> conf = _merge_yaml_and_cli_args(yaml_args, cli_args)
        >>> print(conf)
        >>> {"a": 1, "b": {"_component_": "my_module.models.other_model", "c": 4}}

    Args:
        yaml_args (Namespace): Namespace containing args from yaml file, components
            should have _component_ fields
        cli_args (list[str]): list of key=value strings

    Returns:
        DictConfig: OmegaConf DictConfig containing merged args

    Raises:
        ValueError: If a cli override is not in the form of key=value
    """
    # Convert Namespace to simple dict
    yaml_kwargs = vars(yaml_args)
    cli_dotlist = []
    for arg in cli_args:
        # If CLI override uses the remove flag (~), remove the key from the yaml config
        if arg.startswith("~"):
            dotpath = arg[1:].split("=")[0]
            if "_component_" in dotpath:
                raise ValueError(
                    f"Removing components from CLI is not supported: ~{dotpath}"
                )
            try:
                _remove_key_by_dotpath(yaml_kwargs, dotpath)
            except (KeyError, ValueError):
                raise ValueError(
                    f"Could not find key {dotpath} in yaml config to remove"
                ) from None
            continue
        # Get other overrides that should be specified as key=value
        try:
            k, v = arg.split("=")
        except ValueError:
            raise ValueError(
                f"Command-line overrides must be in the form of key=value, got {arg}"
            ) from None
        # If a cli arg overrides a yaml arg with a _component_ field, update the
        # key string to reflect this
        if k in yaml_kwargs and _has_component(yaml_kwargs[k]):
            k += "._component_"

        # None passed via CLI will be parsed as string, but we really want OmegaConf null
        if v == "None":
            v = "!!null"

        # TODO: this is a hack but otherwise we can't pass strings with leading zeroes
        # to define the checkpoint file format. We manually override OmegaConf behavior
        # by prepending the value with !!str to force a string type
        if "max_filename" in k:
            v = "!!str " + v
        cli_dotlist.append(f"{k}={v}")

    # Merge the args
    cli_conf = OmegaConf.from_dotlist(cli_dotlist)
    yaml_conf = OmegaConf.create(yaml_kwargs)

    # CLI takes precedence over yaml args
    return OmegaConf.merge(yaml_conf, cli_conf)


def _resolve_hf_model_path(hf_url: str) -> str:
    """Resolve HuggingFace model URL to local path using snapshot_download."""
    if not hf_url.startswith("hf://"):
        raise ValueError(f"Invalid HuggingFace URL format: {hf_url}")

    repo_name = hf_url.replace("hf://", "")
    if not repo_name:
        raise ValueError("Empty repository name in HuggingFace URL")

    try:
        # First, try to get from cache only (local_files_only=True)
        # This checks if the model is already cached without downloading
        try:
            local_dir = snapshot_download(
                repo_name, revision="main", local_files_only=True
            )
            return local_dir
        except LocalEntryNotFoundError:
            # Model not in cache, download it (local_files_only=False)
            local_dir = snapshot_download(
                repo_name, revision="main", local_files_only=False
            )
            return local_dir

    except Exception as e:
        raise Exception(
            f"Failed to resolve HuggingFace model '{repo_name}': {e}"
        ) from e


def resolve_hf_hub_paths(cfg: DictConfig) -> DictConfig:
    """
    Resolves HuggingFace Hub URLs in configuration by downloading models and
    replacing "hf://repository_name" paths with local cache paths.

    This function uses the official HuggingFace Hub cache management functions
    to efficiently handle model downloads and caching. It first checks if the
    model is already cached using try_to_load_from_cache(), and only downloads
    if necessary using snapshot_download().

    Args:
        cfg (DictConfig): OmegaConf DictConfig containing configuration values.
            Any string value starting with "hf://" will be processed.

    Returns:
        DictConfig: OmegaConf DictConfig with hf:// URLs replaced by local paths.

    Raises:
        ValueError: If cfg is None or not a valid OmegaConf config object.
        Exception: If model download fails (network issues, invalid repository, etc.)

    Examples:
        >>> config = OmegaConf.create({
        ...     "model": "hf://meta-llama/Llama-2-7b-hf",
        ...     "tokenizer": "hf://microsoft/DialoGPT-medium"
        ... })
        >>> resolved = resolve_hf_hub_paths(config)
        >>> print(resolved.model)  # /home/user/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf
    """
    if cfg is None:
        raise ValueError("Configuration cannot be None")

    if not OmegaConf.is_config(cfg):
        raise ValueError(f"Input must be an OmegaConf config object, got {type(cfg)}")

    def _recursively_resolve_paths(obj: Any) -> Any:
        """Recursively resolve hf:// paths in nested data structures."""
        if isinstance(obj, str) and obj.startswith("hf://"):
            return _resolve_hf_model_path(obj)
        elif isinstance(obj, dict):
            return {k: _recursively_resolve_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_recursively_resolve_paths(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(_recursively_resolve_paths(item) for item in obj)
        elif isinstance(obj, DictConfig):
            # Handle nested DictConfig objects by converting to dict first
            return _recursively_resolve_paths(OmegaConf.to_container(obj, resolve=True))
        elif hasattr(obj, "__dict__"):
            # Handle objects with __dict__ by modifying their attributes
            for attr, value in vars(obj).items():
                setattr(obj, attr, _recursively_resolve_paths(value))
            return obj
        else:
            # Return as-is for other types (int, float, bool, None, etc.)
            return obj

    # Convert OmegaConf to container with resolved variables, process it, then convert back
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    resolved_dict = _recursively_resolve_paths(cfg_dict)

    return OmegaConf.create(resolved_dict)


class ForgeRecipeArgParser(argparse.ArgumentParser):
    """
    A helpful utility subclass of the ``argparse.ArgumentParser`` that
    adds a builtin argument "config". The config argument takes a file path to a YAML file
    and loads in argument defaults from said file. The YAML file must only contain
    argument names and their values and nothing more, it does not have to include all of the
    arguments. These values will be treated as defaults and can still be overridden from the
    command line. Everything else works the same as the base ArgumentParser and you should
    consult the docs for more info: https://docs.python.org/3/library/argparse.html.

    Note:
        This class uses "config" as a builtin argument so it is not available to use.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        super().add_argument(
            "--config",
            type=str,
            help="Path/name of a yaml file with recipe args",
            required=True,
        )

    def parse_known_args(self, *args, **kwargs) -> tuple[Namespace, list[str]]:
        """This acts the same as the base parse_known_args but will first load in defaults from
        from the config yaml file if it is provided. The command line args will always take
        precident over the values in the config file. All other parsing method, such as parse_args,
        internally call this method so they will inherit this property too. For more info see
        the docs for the base method: https://docs.python.org/3/library/argparse.html#the-parse-args-method.
        """
        namespace, unknown_args = super().parse_known_args(*args, **kwargs)

        unknown_flag_args = [arg for arg in unknown_args if arg.startswith("--")]
        if unknown_flag_args:
            raise ValueError(
                f"Additional flag arguments not supported: {unknown_flag_args}. Please use --config or key=value overrides"
            )

        config = OmegaConf.load(namespace.config)
        assert "config" not in config, "Cannot use 'config' within a config file"
        self.set_defaults(**OmegaConf.to_container(config, resolve=False))

        namespace, unknown_args = super().parse_known_args(*args, **kwargs)
        del namespace.config

        return namespace, unknown_args


def parse(recipe_main: Any) -> Callable[..., Any]:
    """
    Decorator that handles parsing the config file and CLI overrides
    for a recipe. Use it on the recipe's main function.

    Args:
        recipe_main (Recipe): The main method that initializes
            and runs the recipe

    Examples:
        >>> @parse
        >>> def main(cfg: DictConfig):
        >>>     ...

        >>> # With the decorator, the parameters will be parsed into cfg when run as:
        >>> tune my_recipe --config config.yaml foo=bar

    Returns:
        Callable[..., Any]: the decorated main
    """

    @functools.wraps(recipe_main)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        parser = ForgeRecipeArgParser(
            description=recipe_main.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        # Get user-specified args from config and CLI and create params for recipe
        yaml_args, cli_args = parser.parse_known_args()
        conf = _merge_yaml_and_cli_args(yaml_args, cli_args)
        conf = resolve_hf_hub_paths(conf)

        sys.exit(recipe_main(conf))

    return wrapper
