# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for vllm.v1.monarch_executor._get_host_ip (regression for #743).

Issue #743: inside containers, hostname resolution can return an IPv6
link-local address (fe80::...) which c10d / TCPStore cannot bind, leading
to "IPv4 network addresses ... cannot be retrieved" failures. _get_host_ip
must prefer IPv4 non-link-local addresses before falling back to IPv6.
"""

import socket
import sys
from unittest import mock

import pytest


@pytest.fixture
def get_host_ip():
    """Import the function under test with monarch / vllm stubbed out so the
    test does not require those packages to be installed."""
    stubs = {
        "monarch": mock.MagicMock(),
        "monarch.actor": mock.MagicMock(),
        "monarch.tools": mock.MagicMock(),
        "monarch.tools.network": mock.MagicMock(),
        "vllm": mock.MagicMock(),
        "vllm.v1": mock.MagicMock(),
        "vllm.v1.executor": mock.MagicMock(),
        "vllm.v1.executor.abstract": mock.MagicMock(),
        "vllm.v1.worker": mock.MagicMock(),
        "vllm.v1.worker.worker_base": mock.MagicMock(),
        "cloudpickle": mock.MagicMock(),
    }
    with mock.patch.dict(sys.modules, stubs):
        # Force re-import so stubs are picked up
        sys.modules.pop("forge.actors.vllm.v1.monarch_executor", None)
        from forge.actors.vllm.v1.monarch_executor import _get_host_ip
        yield _get_host_ip


def _addrinfo(family: int, ip: str) -> tuple:
    """Build a getaddrinfo tuple: (family, type, proto, canon, sockaddr)."""
    sockaddr = (ip, 0) if family == socket.AF_INET else (ip, 0, 0, 0)
    return (family, socket.SOCK_STREAM, 0, "", sockaddr)


class TestGetHostIp:
    def test_vllm_host_ip_env_override(self, get_host_ip, monkeypatch):
        monkeypatch.setenv("VLLM_HOST_IP", "10.20.30.40")
        assert get_host_ip() == "10.20.30.40"

    def test_prefers_ipv4_over_ipv6_link_local(self, get_host_ip, monkeypatch):
        """Regression for #743: IPv4 must win over IPv6 link-local."""
        monkeypatch.delenv("VLLM_HOST_IP", raising=False)
        infos = [
            _addrinfo(socket.AF_INET6, "fe80::222:48ff:fe49:ba90"),
            _addrinfo(socket.AF_INET, "192.168.1.50"),
        ]
        with mock.patch("socket.getaddrinfo", return_value=infos):
            assert get_host_ip() == "192.168.1.50"

    def test_skips_ipv4_link_local(self, get_host_ip, monkeypatch):
        """169.254.x.x is APIPA / link-local and not routable."""
        monkeypatch.delenv("VLLM_HOST_IP", raising=False)
        infos = [
            _addrinfo(socket.AF_INET, "169.254.88.125"),
            _addrinfo(socket.AF_INET, "192.168.1.50"),
        ]
        with mock.patch("socket.getaddrinfo", return_value=infos):
            assert get_host_ip() == "192.168.1.50"

    def test_falls_back_to_ipv6_when_only_ipv6_available(
        self, get_host_ip, monkeypatch
    ):
        monkeypatch.delenv("VLLM_HOST_IP", raising=False)
        infos = [
            _addrinfo(socket.AF_INET6, "fe80::1"),  # link-local, must skip
            _addrinfo(socket.AF_INET6, "2001:db8::1"),  # global, OK
        ]
        with mock.patch("socket.getaddrinfo", return_value=infos):
            assert get_host_ip() == "2001:db8::1"

    def test_falls_back_to_monarch_when_only_link_local(
        self, get_host_ip, monkeypatch
    ):
        """If every candidate is link-local, fall back to Monarch's
        resolver to preserve prior behavior in degenerate cases."""
        monkeypatch.delenv("VLLM_HOST_IP", raising=False)
        infos = [
            _addrinfo(socket.AF_INET, "169.254.88.125"),
            _addrinfo(socket.AF_INET6, "fe80::1"),
        ]
        with mock.patch("socket.getaddrinfo", return_value=infos), mock.patch(
            "forge.actors.vllm.v1.monarch_executor.get_ipaddr",
            return_value="fallback-ip",
        ):
            assert get_host_ip() == "fallback-ip"

    def test_falls_back_to_monarch_on_gaierror(self, get_host_ip, monkeypatch):
        monkeypatch.delenv("VLLM_HOST_IP", raising=False)
        with mock.patch(
            "socket.getaddrinfo", side_effect=socket.gaierror("no such host")
        ), mock.patch(
            "forge.actors.vllm.v1.monarch_executor.get_ipaddr",
            return_value="monarch-fallback",
        ):
            assert get_host_ip() == "monarch-fallback"
