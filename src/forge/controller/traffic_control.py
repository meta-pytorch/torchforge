"""Blocking an async endpoint until pending requests are small enough"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator
