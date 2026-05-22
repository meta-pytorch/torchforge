# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for provisioner._get_port (regression for issue #520).

Issue #520: two training jobs on the same node collided on an ephemeral
port because _get_port() bound to 'localhost' (127.0.0.1) while the
returned port was later used on the FQDN interface by TCPStore. The OS
treats those as separate bind scopes, so the same port could be handed
out twice. _get_port must bind to the wildcard address ('') and call
listen() to reserve the port across all interfaces.
"""

import socket
from unittest import mock

from forge.controller.provisioner import _get_port


class TestGetPort:
    def test_returns_valid_tcp_port(self):
        port = int(_get_port())
        assert 1 <= port < 65536

    def test_returned_port_is_currently_free_on_wildcard(self):
        """The returned port must be free across all interfaces, not just
        loopback. Binding to the wildcard address with the same port must
        succeed immediately after _get_port releases its socket."""
        port = int(_get_port())
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))  # would raise OSError if port wasn't truly free

    def test_binds_to_wildcard_address_not_localhost(self):
        """Regression for #520: _get_port must bind on the wildcard address
        ('') so the OS reserves the port across every interface, including
        the FQDN interface that TCPStore later binds to."""
        with mock.patch(
            "forge.controller.provisioner.socket.socket"
        ) as mock_socket_cls:
            sock = mock_socket_cls.return_value.__enter__.return_value
            sock.getsockname.return_value = ("0.0.0.0", 12345)
            _get_port()

        sock.bind.assert_called_once_with(("", 0))

    def test_calls_listen_to_reserve_port(self):
        """Without listen(), the kernel can hand the same port to another
        bind('', 0) caller before the returned port number is consumed."""
        with mock.patch(
            "forge.controller.provisioner.socket.socket"
        ) as mock_socket_cls:
            sock = mock_socket_cls.return_value.__enter__.return_value
            sock.getsockname.return_value = ("0.0.0.0", 12345)
            _get_port()

        assert sock.listen.called, (
            "_get_port must call listen() after bind() to fully reserve the "
            "port until the socket is closed (issue #520)."
        )
