"""Length-prefixed JSON and raw RGB over a private Unix-domain socket."""

import json
import socket
import struct

MAX_HEADER = 2 * 1024 * 1024
MAX_IMAGE = 1280 * 720 * 3


def read_exact(stream: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = stream.recv(size - len(data))
        if not chunk:
            raise EOFError("truncated IPC message")
        data.extend(chunk)
    return bytes(data)


def read_message(stream: socket.socket):
    size, = struct.unpack("!I", read_exact(stream, 4))
    if not 0 < size <= MAX_HEADER:
        raise ValueError("invalid JSON message length")
    return json.loads(read_exact(stream, size))


def send_message(stream: socket.socket, message) -> None:
    payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > MAX_HEADER:
        raise ValueError("IPC JSON message exceeds the limit")
    stream.sendall(struct.pack("!I", len(payload)) + payload)


def request(path, operation, rgb=None, **options):
    header = {"op": operation, **options}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.settimeout(240)
        stream.connect(str(path))
        if rgb is None:
            send_message(stream, header)
        else:
            height, width, channels = rgb.shape
            if channels != 3 or rgb.dtype.name != "uint8" or rgb.nbytes > MAX_IMAGE:
                raise ValueError("expected a supported uint8 RGB array")
            send_message(stream, dict(header, height=height, width=width))
            stream.sendall(rgb.tobytes())
        response = read_message(stream)
    if "error" in response:
        raise RuntimeError(response["error"])
    return response
