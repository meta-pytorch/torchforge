# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forge.controller.dns_aid import (
    _fqdn,
    _try_import_dns_aid,
    discover_peers,
    is_dns_aid_enabled,
    publish_service,
    unpublish_service,
)
from forge.types import DnsAidConfig


@pytest.fixture(autouse=True)
def _reset_dns_aid_import_cache():
    """Reset the cached import state between tests."""
    import forge.controller.dns_aid as mod

    mod._dns_aid_import_attempted = False
    mod._dns_aid_module = None
    yield
    mod._dns_aid_import_attempted = False
    mod._dns_aid_module = None


# --- _fqdn ---


def test_fqdn():
    assert _fqdn("generator") == "torchforge-generator"
    assert _fqdn("replay-buffer") == "torchforge-replay-buffer"


# --- is_dns_aid_enabled ---


def test_is_dns_aid_enabled_both_true(monkeypatch):
    monkeypatch.setenv("DNS_AID_ENABLED", "true")
    cfg = DnsAidConfig(enabled=True)
    assert is_dns_aid_enabled(cfg) is True


def test_is_dns_aid_enabled_env_false(monkeypatch):
    monkeypatch.setenv("DNS_AID_ENABLED", "false")
    cfg = DnsAidConfig(enabled=True)
    assert is_dns_aid_enabled(cfg) is False


def test_is_dns_aid_enabled_config_false(monkeypatch):
    monkeypatch.setenv("DNS_AID_ENABLED", "true")
    cfg = DnsAidConfig(enabled=False)
    assert is_dns_aid_enabled(cfg) is False


def test_is_dns_aid_enabled_none_config():
    assert is_dns_aid_enabled(None) is False


# --- _try_import_dns_aid caching ---


def test_import_warning_only_once(monkeypatch):
    """The missing-package warning should fire once, not on every call."""
    import forge.controller.dns_aid as mod

    with patch.dict("sys.modules", {"dns_aid": None}):
        # Simulate ImportError by patching builtins
        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "dns_aid":
                raise ImportError("no dns_aid")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result1 = _try_import_dns_aid()
            result2 = _try_import_dns_aid()

    assert result1 is None
    assert result2 is None
    # Second call should have used cache, not re-imported
    assert mod._dns_aid_import_attempted is True


# --- publish_service ---


@pytest.mark.asyncio
async def test_publish_service_success(monkeypatch):
    monkeypatch.setenv("DNS_AID_ENABLED", "true")
    cfg = DnsAidConfig(enabled=True, domain="test.internal", port=7860, ttl=60)

    mock_dns_aid = MagicMock()
    mock_dns_aid.publish = AsyncMock()

    with patch(
        "forge.controller.dns_aid._try_import_dns_aid", return_value=mock_dns_aid
    ):
        result = await publish_service("generator", "host1", 7860, cfg)

    assert result is True
    mock_dns_aid.publish.assert_called_once()
    call_kwargs = mock_dns_aid.publish.call_args.kwargs
    assert call_kwargs["name"] == "torchforge-generator"
    assert call_kwargs["domain"] == "test.internal"
    assert call_kwargs["endpoint"] == "host1"
    assert call_kwargs["port"] == 7860
    assert call_kwargs["ttl"] == 60
    assert "framework:torchforge" in call_kwargs["capabilities"]
    assert "role:generator" in call_kwargs["capabilities"]


@pytest.mark.asyncio
async def test_publish_service_with_extra_capabilities(monkeypatch):
    monkeypatch.setenv("DNS_AID_ENABLED", "true")
    cfg = DnsAidConfig(enabled=True, port=7861, capabilities=["gpu:8", "shard_count:4"])

    mock_dns_aid = MagicMock()
    mock_dns_aid.publish = AsyncMock()

    with patch(
        "forge.controller.dns_aid._try_import_dns_aid", return_value=mock_dns_aid
    ):
        await publish_service("trainer", "host2", 7861, cfg)

    call_kwargs = mock_dns_aid.publish.call_args.kwargs
    caps = call_kwargs["capabilities"]
    assert caps == ["framework:torchforge", "role:trainer", "gpu:8", "shard_count:4"]


@pytest.mark.asyncio
async def test_publish_service_failure_no_raise(monkeypatch):
    monkeypatch.setenv("DNS_AID_ENABLED", "true")
    cfg = DnsAidConfig(enabled=True, port=7860)

    mock_dns_aid = MagicMock()
    mock_dns_aid.publish = AsyncMock(side_effect=ConnectionError("DNS unreachable"))

    with patch(
        "forge.controller.dns_aid._try_import_dns_aid", return_value=mock_dns_aid
    ):
        result = await publish_service("generator", "host1", 7860, cfg)

    assert result is False


@pytest.mark.asyncio
async def test_publish_skipped_when_disabled(monkeypatch):
    monkeypatch.setenv("DNS_AID_ENABLED", "false")
    cfg = DnsAidConfig(enabled=True, port=7860)

    mock_dns_aid = MagicMock()
    mock_dns_aid.publish = AsyncMock()

    with patch(
        "forge.controller.dns_aid._try_import_dns_aid", return_value=mock_dns_aid
    ):
        result = await publish_service("generator", "host1", 7860, cfg)

    assert result is False
    mock_dns_aid.publish.assert_not_called()


@pytest.mark.asyncio
async def test_publish_uses_forge_version(monkeypatch):
    monkeypatch.setenv("DNS_AID_ENABLED", "true")
    cfg = DnsAidConfig(enabled=True, port=8080)

    mock_dns_aid = MagicMock()
    mock_dns_aid.publish = AsyncMock()

    with patch(
        "forge.controller.dns_aid._try_import_dns_aid", return_value=mock_dns_aid
    ):
        with patch("forge.controller.dns_aid._get_forge_version", return_value="0.5.0"):
            await publish_service("gen", "host", 8080, cfg)

    assert mock_dns_aid.publish.call_args.kwargs["version"] == "0.5.0"


# --- unpublish_service ---


@pytest.mark.asyncio
async def test_unpublish_service_success(monkeypatch):
    monkeypatch.setenv("DNS_AID_ENABLED", "true")
    cfg = DnsAidConfig(enabled=True, domain="test.internal")

    mock_dns_aid = MagicMock()
    mock_dns_aid.unpublish = AsyncMock(return_value=True)

    with patch(
        "forge.controller.dns_aid._try_import_dns_aid", return_value=mock_dns_aid
    ):
        result = await unpublish_service("generator", cfg)

    assert result is True
    mock_dns_aid.unpublish.assert_called_once_with(
        name="torchforge-generator",
        domain="test.internal",
        protocol="mcp",
    )


@pytest.mark.asyncio
async def test_unpublish_service_best_effort(monkeypatch):
    monkeypatch.setenv("DNS_AID_ENABLED", "true")
    cfg = DnsAidConfig(enabled=True)

    mock_dns_aid = MagicMock()
    mock_dns_aid.unpublish = AsyncMock(side_effect=RuntimeError("DNS timeout"))

    with patch(
        "forge.controller.dns_aid._try_import_dns_aid", return_value=mock_dns_aid
    ):
        result = await unpublish_service("generator", cfg)

    assert result is False


# --- discover_peers ---


@pytest.mark.asyncio
async def test_discover_peers_success(monkeypatch):
    monkeypatch.setenv("DNS_AID_ENABLED", "true")
    cfg = DnsAidConfig(enabled=True, domain="test.internal")

    mock_agent = MagicMock()
    mock_result = MagicMock()
    mock_result.agents = [mock_agent]

    mock_dns_aid = MagicMock()
    mock_dns_aid.discover = AsyncMock(return_value=mock_result)

    with patch(
        "forge.controller.dns_aid._try_import_dns_aid", return_value=mock_dns_aid
    ):
        agents = await discover_peers("trainer", cfg)

    assert len(agents) == 1
    assert agents[0] is mock_agent


@pytest.mark.asyncio
async def test_discover_peers_retry_with_backoff(monkeypatch):
    """Verify exponential backoff delays between retry attempts."""
    monkeypatch.setenv("DNS_AID_ENABLED", "true")
    cfg = DnsAidConfig(enabled=True)

    success_result = MagicMock()
    success_result.agents = [MagicMock()]

    mock_dns_aid = MagicMock()
    mock_dns_aid.discover = AsyncMock(
        side_effect=[
            ConnectionError("fail 1"),
            ConnectionError("fail 2"),
            success_result,
        ]
    )

    mock_sleep = AsyncMock()
    with patch(
        "forge.controller.dns_aid._try_import_dns_aid", return_value=mock_dns_aid
    ):
        with patch("forge.controller.dns_aid.asyncio.sleep", mock_sleep):
            agents = await discover_peers(
                "trainer", cfg, initial_delay=1.0, backoff_factor=2.0, max_delay=10.0
            )

    assert len(agents) == 1
    assert mock_dns_aid.discover.call_count == 3
    # Verify exponential backoff: 1.0s after first fail, 2.0s after second
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)


@pytest.mark.asyncio
async def test_discover_peers_all_retries_fail(monkeypatch):
    monkeypatch.setenv("DNS_AID_ENABLED", "true")
    cfg = DnsAidConfig(enabled=True)

    mock_dns_aid = MagicMock()
    mock_dns_aid.discover = AsyncMock(side_effect=ConnectionError("always fails"))

    with patch(
        "forge.controller.dns_aid._try_import_dns_aid", return_value=mock_dns_aid
    ):
        with patch("forge.controller.dns_aid.asyncio.sleep", new_callable=AsyncMock):
            agents = await discover_peers(
                "trainer", cfg, max_attempts=3, initial_delay=0.01
            )

    assert agents == []
    assert mock_dns_aid.discover.call_count == 3


@pytest.mark.asyncio
async def test_discover_peers_retry_on_empty_true(monkeypatch):
    """With retry_on_empty=True (default), empty results trigger retries."""
    monkeypatch.setenv("DNS_AID_ENABLED", "true")
    cfg = DnsAidConfig(enabled=True)

    empty_result = MagicMock()
    empty_result.agents = []
    success_result = MagicMock()
    success_result.agents = [MagicMock()]

    mock_dns_aid = MagicMock()
    mock_dns_aid.discover = AsyncMock(side_effect=[empty_result, success_result])

    with patch(
        "forge.controller.dns_aid._try_import_dns_aid", return_value=mock_dns_aid
    ):
        with patch("forge.controller.dns_aid.asyncio.sleep", new_callable=AsyncMock):
            agents = await discover_peers("trainer", cfg, retry_on_empty=True)

    assert len(agents) == 1
    assert mock_dns_aid.discover.call_count == 2


@pytest.mark.asyncio
async def test_discover_peers_retry_on_empty_false(monkeypatch):
    """With retry_on_empty=False, empty results return immediately."""
    monkeypatch.setenv("DNS_AID_ENABLED", "true")
    cfg = DnsAidConfig(enabled=True)

    empty_result = MagicMock()
    empty_result.agents = []

    mock_dns_aid = MagicMock()
    mock_dns_aid.discover = AsyncMock(return_value=empty_result)

    with patch(
        "forge.controller.dns_aid._try_import_dns_aid", return_value=mock_dns_aid
    ):
        agents = await discover_peers("trainer", cfg, retry_on_empty=False)

    assert agents == []
    mock_dns_aid.discover.assert_called_once()


# --- Import guard ---


@pytest.mark.asyncio
async def test_dns_aid_import_missing(monkeypatch):
    monkeypatch.setenv("DNS_AID_ENABLED", "true")
    cfg = DnsAidConfig(enabled=True)

    with patch("forge.controller.dns_aid._try_import_dns_aid", return_value=None):
        publish_result = await publish_service("gen", "host", 8080, cfg)
        unpublish_result = await unpublish_service("gen", cfg)
        discover_result = await discover_peers("gen", cfg)

    assert publish_result is False
    assert unpublish_result is False
    assert discover_result == []


# --- Provisioner shutdown integration ---


@pytest.mark.asyncio
async def test_provisioner_shutdown_calls_unpublish(monkeypatch):
    """Verify that shutdown_all_allocations unpublishes DNS-AID services."""
    monkeypatch.setenv("DNS_AID_ENABLED", "true")

    dns_cfg = DnsAidConfig(enabled=True, domain="test.internal", port=7860)

    # Build a minimal ServiceInterface-like object with the expected attributes
    mock_service = MagicMock()
    mock_service._dns_aid_cfg = dns_cfg
    mock_service.actor_def.__name__ = "MyGenerator"
    mock_service.shutdown = AsyncMock()

    mock_unpublish = AsyncMock(return_value=True)

    with patch("forge.controller.provisioner.unpublish_service", mock_unpublish):
        with patch(
            "forge.controller.provisioner.is_dns_aid_enabled", return_value=True
        ):
            with patch(
                "forge.controller.provisioner.shutdown_context", new_callable=AsyncMock
            ):
                from forge.controller.provisioner import Provisioner

                provisioner = Provisioner.__new__(Provisioner)
                provisioner._lock = __import__("asyncio").Lock()
                provisioner._registered_services = [mock_service]
                provisioner._registered_actors = []
                provisioner.launcher = None

                await provisioner.shutdown_all_allocations()

    mock_unpublish.assert_called_once_with("mygenerator", dns_cfg)
    mock_service.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_provisioner_shutdown_skips_when_no_dns_cfg(monkeypatch):
    """Services without DNS-AID config should not trigger unpublish."""
    mock_service = MagicMock()
    mock_service._dns_aid_cfg = None
    mock_service.shutdown = AsyncMock()

    mock_unpublish = AsyncMock()

    with patch("forge.controller.provisioner.unpublish_service", mock_unpublish):
        with patch(
            "forge.controller.provisioner.is_dns_aid_enabled", return_value=False
        ):
            with patch(
                "forge.controller.provisioner.shutdown_context", new_callable=AsyncMock
            ):
                from forge.controller.provisioner import Provisioner

                provisioner = Provisioner.__new__(Provisioner)
                provisioner._lock = __import__("asyncio").Lock()
                provisioner._registered_services = [mock_service]
                provisioner._registered_actors = []
                provisioner.launcher = None

                await provisioner.shutdown_all_allocations()

    mock_unpublish.assert_not_called()
    mock_service.shutdown.assert_called_once()
