# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Example of using forge.monarch APIs.

Run this with python -m examples.stacked_envs

"""

import asyncio

from forge.controller.stack import stack
from monarch.actor import Actor, endpoint, proc_mesh


class Environment(Actor):
    @endpoint
    async def step(self):
        print("env step")


class BrowserEnv(Environment):
    @endpoint
    async def step(self):
        print("browser step")


class CodingEnv(Environment):
    @endpoint
    async def step(self):
        print("coding step")


async def main():
    m = await proc_mesh(gpus=1)
    browser = await m.spawn("browser", BrowserEnv)
    coding = await m.spawn("coding", CodingEnv)
    # note that interface can be deduced, but adding it explicitly
    # helps the type checker
    envs = stack(browser, coding, interface=Environment)
    # Temporary hack due to Monarch changes
    res = envs.step.call()
    [await r for r in res]


if __name__ == "__main__":
    asyncio.run(main())
