import socket

from monarch.actor import Actor, endpoint, HostMesh, ProcMesh, this_host


def _get_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        addr = s.getsockname()
        port = addr[1]
        return str(port)


class _SetupActor(Actor):
    @endpoint
    def get_info(self) -> tuple[str, str]:
        return socket.gethostname(), _get_port()
