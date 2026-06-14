# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time

import torchstore as ts
from forge.actors._torchstore_utils import get_param_prefix

logger = logging.getLogger(__name__)


async def drop_weights(version: int):
    print(f"Dropping weights @ version {version}")
    start_time = time.perf_counter()
    prefix = get_param_prefix(version)
    matching_keys = await ts.keys(prefix)
    # TODO: once we have something like `get_meta()` in torchstore, we can just
    # query the type of the object instead of relying on keys.
    for key in matching_keys:
        await ts.delete(key)
    elapsed = time.perf_counter() - start_time
    print(f"Dropped weights @ version {version}, took {elapsed:.2f} seconds")


def warn_if_resuming_from_existing_folder(
    folder: str | None, initial_load_path: str | None = None
) -> bool:
    """Logs a loud WARNING when the checkpointer is about to silently resume
    from an existing ``checkpoint.folder``.

    Torchtitan's checkpointer treats ``folder`` as the source of truth: if it
    already contains saved step directories (``step-N``), it loads from there
    and ignores ``initial_load_path``. Users running back-to-back experiments
    without clearing the folder hit this footgun (see #631) — training
    silently picks up where the prior run left off instead of starting from
    the configured base model.

    This helper logs once before the load happens so the resume is visible
    in the standard training logs. Returns ``True`` when a warning was
    emitted, so callers can also surface it through other channels (e.g. an
    extra console banner) if they want.
    """
    if not folder or not os.path.isdir(folder):
        return False

    try:
        entries = os.listdir(folder)
    except OSError as exc:
        logger.debug("could not list checkpoint folder %s: %s", folder, exc)
        return False

    def _step_number(entry: str) -> int:
        try:
            return int(entry.removeprefix("step-").split("-", 1)[0])
        except ValueError:
            return -1

    step_dirs = [
        entry
        for entry in entries
        if entry.startswith("step-")
        and os.path.isdir(os.path.join(folder, entry))
    ]
    step_dirs.sort(key=_step_number)
    if not step_dirs:
        return False

    extra = ""
    if initial_load_path:
        extra = (
            f" Configured initial_load_path={initial_load_path!r} will be ignored "
            "until the folder is cleared or renamed."
        )
    logger.warning(
        "Resuming training from existing checkpoint folder %r (found %d saved "
        "step dir(s); latest: %s).%s",
        folder,
        len(step_dirs),
        step_dirs[-1],
        extra,
    )
    return True
