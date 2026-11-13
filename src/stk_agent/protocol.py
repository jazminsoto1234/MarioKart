from __future__ import annotations
import json
import struct

MAX_HEADER = 65536
MAX_PAYLOAD = 64 * 1024 * 1024


def read_exact(stream, size):
    chunks, got = [], 0
    while got < size:
        data = stream.read(size-got)
        if not data:
            raise EOFError("Canal del agente cerrado")
        chunks.append(data)
        got += len(data)
    return b"".join(chunks)


def send_packet(stream, header, payload=b""):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("Payload demasiado grande")
    header = {**header, "payload_bytes": len(payload)}
    data = json.dumps(header, allow_nan=False).encode("utf-8")
    if len(data) > MAX_HEADER:
        raise ValueError("Cabecera demasiado grande")
    stream.write(struct.pack("!I", len(data)))
    stream.write(data)
    if payload:
        stream.write(payload)
    stream.flush()


def receive_packet(stream, max_payload=MAX_PAYLOAD):
    size = struct.unpack("!I", read_exact(stream, 4))[0]
    if not 0 < size <= MAX_HEADER:
        raise ValueError("Cabecera de agente inválida; no escribas bytes en stdout")
    header = json.loads(read_exact(stream, size))
    payload_size = header.get("payload_bytes")
    if type(payload_size) is not int or not 0 <= payload_size <= max_payload:
        raise ValueError("Tamaño de payload no permitido")
    payload = read_exact(stream, payload_size) if payload_size else b""
    return header, payload
