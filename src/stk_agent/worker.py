from __future__ import annotations
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
import time
import numpy as np
from .protocol import send_packet, receive_packet
from .types import Action


def main():
    source, sink = sys.stdin.buffer, sys.stdout.buffer
    agent = None
    try:
        header, _ = receive_packet(source, max_payload=0)
        if header.get("kind") != "load":
            raise ValueError("Se esperaba load")
        path = Path(header["entry"]).resolve()
        sys.path.insert(0, str(path.parent))
        with contextlib.redirect_stdout(sys.stderr):
            spec = importlib.util.spec_from_file_location("agent_impl", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["agent_impl"] = module
            spec.loader.exec_module(module)
            agent = getattr(module, header.get("class", "Agent"))(header.get("config", {}))
        send_packet(sink, {"ok": True})
        while True:
            header, payload = receive_packet(source)
            kind = header.get("kind")
            if kind == "close":
                break
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                if kind == "reset":
                    agent.reset()
                    result = None
                elif kind == "act":
                    shape = tuple(header["shape"])
                    if len(shape) != 3 or shape[2] != 3 or len(payload) != int(np.prod(shape)):
                        raise ValueError("RGB inválido")
                    rgb = np.frombuffer(payload, np.uint8).reshape(shape).copy()
                    result = Action.from_any(agent.act(rgb)).as_dict()
                else:
                    raise ValueError(f"Comando desconocido: {kind}")
            send_packet(sink, {"ok": True, "action": result,
                               "compute_seconds": time.perf_counter()-started})
    except EOFError:
        pass
    except BaseException:
        try:
            send_packet(sink, {"ok": False, "error": traceback.format_exc()})
        except (BrokenPipeError, OSError):
            pass
        return 1
    finally:
        if agent is not None and hasattr(agent, "close"):
            with contextlib.redirect_stdout(sys.stderr):
                agent.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
