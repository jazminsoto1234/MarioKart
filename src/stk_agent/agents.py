from __future__ import annotations
from pathlib import Path
import json
import queue
import subprocess
import sys
import threading
import time
import numpy as np
from .protocol import send_packet, receive_packet
from .types import Action, check_rgb


class AgentWorker:
    def __init__(self, manifest, log_path, load_timeout=120):
        path = Path(manifest).resolve()
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        entry = (path.parent/self.manifest["entry"]).resolve()
        if not entry.is_file():
            raise FileNotFoundError(entry)
        self.name = self.manifest.get("name", path.parent.name)
        self.inbox = queue.Queue(maxsize=4)
        self.waiting = False
        self.closed = False
        self.wall_times, self.compute_times = [], []
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log = Path(log_path).open("wb")
        python = self.manifest.get("python") or sys.executable
        # The selected interpreter must also have stk-agent installed.
        self.process = subprocess.Popen([python, "-m", "stk_agent.worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        self.reader = threading.Thread(target=self._read, daemon=True, name=f"read-{self.name}")
        self.reader.start()
        config = dict(self.manifest.get("config", {}))
        if "checkpoint" in config:
            config["checkpoint"] = str((path.parent/config["checkpoint"]).resolve())
        try:
            send_packet(self.process.stdin, {"kind": "load", "entry": str(entry),
                        "class": self.manifest.get("class", "Agent"), "config": config})
            self._receive(time.perf_counter()+load_timeout)
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            while not self.closed:
                packet = receive_packet(self.process.stdout, max_payload=0)
                try:
                    self.inbox.put(packet[0], timeout=1)
                except queue.Full:
                    break
        except BaseException as e:
            try:
                self.inbox.put_nowait({"ok": False, "error": str(e)})
            except queue.Full:
                pass

    def _receive(self, deadline):
        remaining = deadline-time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"{self.name}: tiempo de espera excedido")
        try:
            message = self.inbox.get(timeout=remaining)
        except queue.Empty as e:
            raise TimeoutError(f"{self.name}: tiempo de espera excedido") from e
        if not message.get("ok"):
            raise RuntimeError(f"Agente {self.name}: {message.get('error')}")
        return message

    def reset(self, timeout=30):
        send_packet(self.process.stdin, {"kind": "reset"})
        self._receive(time.perf_counter()+timeout)

    def request(self, rgb):
        if self.waiting:
            raise RuntimeError("Solicitud anterior pendiente")
        image = check_rgb(rgb)
        self.request_time = time.perf_counter()
        send_packet(self.process.stdin, {"kind": "act", "shape": image.shape},
                    np.ascontiguousarray(image).tobytes())
        self.waiting = True

    def receive(self, deadline):
        if not self.waiting:
            raise RuntimeError("No hay solicitud pendiente")
        response = self._receive(deadline)
        self.waiting = False
        action = Action.from_any(response["action"])
        self.wall_times.append(time.perf_counter()-self.request_time)
        self.compute_times.append(float(response.get("compute_seconds", 0)))
        return action

    def timing(self):
        def summary(values):
            return {} if not values else {"mean_ms": float(np.mean(values)*1000),
                "p95_ms": float(np.percentile(values, 95)*1000), "max_ms": float(max(values)*1000)}
        return {"calls": len(self.wall_times), "host_latency": summary(self.wall_times),
                "worker_compute": summary(self.compute_times)}

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.poll() is None:
                send_packet(self.process.stdin, {"kind": "close"})
                self.process.wait(timeout=2)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            self.process.kill()
            self.process.wait()
        finally:
            for stream in (self.process.stdin, self.process.stdout):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
            self.log.close()
