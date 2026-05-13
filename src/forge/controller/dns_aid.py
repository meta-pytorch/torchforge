# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DNS-AID service discovery helpers for Forge services.

Provides publish, unpublish, and discover wrappers around the dns_aid library
with torchforge-specific defaults, dual enable guards, and retry logic.

All operations are best-effort: failures are logged but never raised, so
service startup and shutdown are not blocked by DNS issues.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from forge.env import DNS_AID_ENABLED

if TYPE_CHECKING:
    from forge.types import DnsAidConfig

logger = logging.getLogger(__name__)

# Cached dns_aid import — None means not yet attempted, False means import failed.
_dns_aid_module = None
_dns_aid_import_attempted = False


def _get_forge_version() -> str:
    """Return the installed forge package version, or 'unknown'."""
    try:
        from importlib.metadata import version

        return version("forge")
    except Exception:
        return "unknown"


def _fqdn(service_name: str) -> str:
    """Build the canonical DNS-AID name for a forge service."""
    return f"torchforge-{service_name}"


def is_dns_aid_enabled(cfg: "DnsAidConfig | None") -> bool:
    """Check whether DNS-AID is enabled via both env var and config.

    Both ``DNS_AID_ENABLED`` environment variable and ``cfg.enabled`` must
    be true for DNS-AID operations to proceed.
    """
    if cfg is None:
        return False
    return bool(DNS_AID_ENABLED.get_value()) and cfg.enabled


def _try_import_dns_aid():
    """Lazily import dns_aid, returning the module or None.

    The result is cached so the warning is only logged once.
    """
    global _dns_aid_module, _dns_aid_import_attempted
    if _dns_aid_import_attempted:
        return _dns_aid_module

    _dns_aid_import_attempted = True
    try:
        import dns_aid

        _dns_aid_module = dns_aid
        return dns_aid
    except ImportError:
        logger.warning(
            "dns-aid package is not installed. "
            "Install with: pip install forge[dns-aid]"
        )
        _dns_aid_module = None
        return None


async def publish_service(
    service_name: str,
    hostname: str,
    port: int,
    cfg: "DnsAidConfig",
) -> bool:
    """Publish a Forge service as a DNS-AID SVCB record.

    Args:
        service_name: Logical name for the service (e.g. "generator").
        hostname: Host where the service is reachable.
        port: Port where the service is reachable.
        cfg: DNS-AID configuration.

    Returns:
        True if publish succeeded, False otherwise.
    """
    if not is_dns_aid_enabled(cfg):
        return False

    dns_aid = _try_import_dns_aid()
    if dns_aid is None:
        return False

    capabilities = [
        "framework:torchforge",
        f"role:{service_name}",
        *cfg.capabilities,
    ]
    dns_name = _fqdn(service_name)

    try:
        await dns_aid.publish(
            name=dns_name,
            domain=cfg.domain,
            protocol=cfg.protocol,
            endpoint=hostname,
            port=port,
            capabilities=capabilities,
            version=_get_forge_version(),
            description=f"Torchforge {service_name} service",
            category=cfg.category,
            ttl=cfg.ttl,
        )
        logger.info(
            f"DNS-AID: published {dns_name} "
            f"at {hostname}:{port} (domain={cfg.domain}, ttl={cfg.ttl}s)"
        )
        return True
    except Exception:
        logger.warning(
            f"DNS-AID: failed to publish {dns_name}",
            exc_info=True,
        )
        return False


async def unpublish_service(
    service_name: str,
    cfg: "DnsAidConfig",
) -> bool:
    """Remove a Forge service's DNS-AID record. Best-effort.

    Args:
        service_name: Logical name for the service.
        cfg: DNS-AID configuration.

    Returns:
        True if unpublish succeeded, False otherwise.
    """
    if not is_dns_aid_enabled(cfg):
        return False

    dns_aid = _try_import_dns_aid()
    if dns_aid is None:
        return False

    dns_name = _fqdn(service_name)
    try:
        await dns_aid.unpublish(
            name=dns_name,
            domain=cfg.domain,
            protocol=cfg.protocol,
        )
        logger.info(f"DNS-AID: unpublished {dns_name}")
        return True
    except Exception:
        logger.warning(
            f"DNS-AID: failed to unpublish {dns_name}",
            exc_info=True,
        )
        return False


async def discover_peers(
    service_name: str,
    cfg: "DnsAidConfig",
    max_attempts: int = 5,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 8.0,
    retry_on_empty: bool = True,
) -> list:
    """Discover peer Forge services via DNS-AID with exponential backoff.

    Args:
        service_name: Name of the service to discover (e.g. "trainer").
        cfg: DNS-AID configuration.
        max_attempts: Maximum number of discovery attempts.
        initial_delay: Initial retry delay in seconds.
        backoff_factor: Multiplier for each subsequent retry delay.
        max_delay: Maximum retry delay in seconds.
        retry_on_empty: If True (default), retry when discovery succeeds but
            returns no agents. Set to False to return immediately on a
            successful-but-empty response.

    Returns:
        List of discovered AgentRecord objects, or empty list on failure.
    """
    if not is_dns_aid_enabled(cfg):
        return []

    dns_aid = _try_import_dns_aid()
    if dns_aid is None:
        return []

    dns_name = _fqdn(service_name)
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            result = await dns_aid.discover(
                domain=cfg.domain,
                protocol=cfg.protocol,
                name=dns_name,
            )
            if result.agents:
                logger.info(
                    f"DNS-AID: discovered {len(result.agents)} peer(s) "
                    f"for {dns_name} (attempt {attempt})"
                )
                return result.agents

            if not retry_on_empty:
                logger.debug(f"DNS-AID: no peers found for {dns_name}")
                return []

        except Exception:
            logger.debug(
                f"DNS-AID: discover attempt {attempt}/{max_attempts} "
                f"for {dns_name} failed",
                exc_info=True,
            )

        if attempt < max_attempts:
            logger.debug(
                f"DNS-AID: retrying discovery in {delay:.1f}s "
                f"(attempt {attempt}/{max_attempts})"
            )
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    logger.warning(
        f"DNS-AID: failed to discover {dns_name} after {max_attempts} attempts"
    )
    return []
