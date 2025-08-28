# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test utilities for tracing function calls via UDP packets.

This module provides utilities for testing distributed/async components where
traditional mocking is difficult due to pickling/unpickling (e.g., with monarch actors).
The UDP tracing approach allows tests to verify that specific functions were called
by having them send UDP packets that can be received and verified by the test.

Warning: This approach has limitations - tests using UDP tracing can be flaky and
only work reliably when run on a single machine because they listen to localhost.

Example usage:
    # In test code
    sampler.sample_keys = add_udp_callback(
        sampler.sample_keys, port=TEST_PORT, message=b"sample_keys"
    )

    # Start UDP receiver in separate thread
    received = []
    server_thread = threading.Thread(
        target=receive_udp_packet,
        args=(TEST_PORT, received),
        kwargs={"timeout": 15},
    )
    server_thread.start()

    # Execute code that should call the wrapped function
    # ...

    # Verify the function was called
    server_thread.join()
    assert b"sample_keys" in received
"""

import socket


def receive_udp_packet(port, received, *, timeout):
    """
    Receives a UDP packet on the specified port and appends it to the received list.

    Args:
        port: The port number to listen on
        received: A list to which received data will be appended
        timeout: Keyword-only argument specifying socket timeout in seconds

    Returns:
        None. Data is appended to the received list if a packet is received before timeout.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("localhost", port))
    sock.settimeout(timeout)
    try:
        data, _ = sock.recvfrom(1024)  # addr is not used
        received.append(data)
    except socket.timeout:
        pass
    finally:
        sock.close()


def send_udp_packet(port, message):
    """
    Sends a UDP packet to localhost on the specified port.

    Args:
        port: The port number to send the packet to
        message: The message/data to send

    Returns:
        None
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(message, ("localhost", port))
    sock.close()


def add_udp_callback(func, port, message):
    """
    Decorator function that wraps another function to send a UDP packet after execution.

    Args:
        func: The function to wrap
        port: The port number to send the packet to
        message: The message/data to send

    Returns:
        A wrapped function that calls the original function and then sends a UDP packet
    """

    def f(*args, **kwargs):
        ret = func(*args, **kwargs)
        send_udp_packet(port, message)
        return ret

    return f
