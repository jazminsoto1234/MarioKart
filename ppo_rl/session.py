"""Sesiones por bloques (§10): límite de tiempo, Ctrl+C seguro, checkpoints atómicos y watchdog."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import torch

from .config import SessionConfig


# ---------------------------------------------------------------- sistema

def set_low_priority():
    if sys.platform == "win32":
        import ctypes
        BELOW_NORMAL = 0x4000
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
    else:
        os.nice(5)


def keep_awake(on: bool):
    """Evita que Windows suspenda mientras corre el comando."""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def gpu_temperature() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
        return max(float(x) for x in out.split())
    except (ValueError, subprocess.SubprocessError, OSError):
        return None


def _deadline(cfg: SessionConfig) -> float | None:
    limits = []
    if cfg.max_hours > 0:
        limits.append(time.time() + cfg.max_hours * 3600)
    if cfg.until:
        h, m = map(int, cfg.until.split(":"))
        now = datetime.now()
        stop = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if stop <= now:
            stop += timedelta(days=1)
        limits.append(stop.timestamp())
    return min(limits) if limits else None


# ---------------------------------------------------------------- checkpoints

def save_checkpoint(state: dict, run_dir: Path, keep: int = 3, name: str = "last.pt"):
    """Escritura atómica (tmp + replace) y rotación de los `keep` últimos."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / name
    if path.exists() and keep > 1:
        for i in range(keep - 1, 1, -1):
            older = run_dir / f"{path.stem}.{i - 1}.pt"
            if older.exists():
                os.replace(older, run_dir / f"{path.stem}.{i}.pt")
        shutil.copy2(path, run_dir / f"{path.stem}.1.pt")
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(run_dir: Path, name: str = "last.pt", device="cpu") -> dict | None:
    path = Path(run_dir) / name
    return torch.load(path, map_location=device, weights_only=False) if path.exists() else None


# ---------------------------------------------------------------- logging

class MetricsLogger:
    """Escribe métricas en metrics.jsonl y, si está instalado, en TensorBoard."""

    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.file = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(run_dir / "tb"))
        except ImportError:
            self.tb = None

    def log(self, step: int, metrics: dict, histograms: dict | None = None):
        clean = {k: float(v) for k, v in metrics.items()}
        self.file.write(json.dumps({"step": step, "time": time.time(), **clean}) + "\n")
        self.file.flush()
        if self.tb:
            for k, v in clean.items():
                self.tb.add_scalar(k, v, step)
            for k, v in (histograms or {}).items():
                self.tb.add_histogram(k, v, step)

    def close(self):
        self.file.close()
        if self.tb:
            self.tb.close()


# ---------------------------------------------------------------- sesión

class Session:
    """Uso: `with Session(cfg) as s:` y en el bucle `if s.should_stop(): break`."""

    def __init__(self, cfg: SessionConfig):
        self.cfg = cfg
        self.deadline = _deadline(cfg)
        self.stop_requested = False
        self.last_save = time.time()
        self.last_temp_check = 0.0
        self.hot_events = 0
        self._old_handler = None

    def __enter__(self):
        torch.set_num_threads(self.cfg.torch_threads)
        if self.cfg.low_priority:
            set_low_priority()
        if self.cfg.prevent_sleep:
            keep_awake(True)
        self._old_handler = signal.signal(signal.SIGINT, self._on_sigint)
        return self

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self._old_handler or signal.default_int_handler)
        if self.cfg.prevent_sleep:
            keep_awake(False)
        return False

    def _on_sigint(self, *_):
        if self.stop_requested:          # segundo Ctrl+C: salir ya
            raise KeyboardInterrupt
        print("\n[session] Ctrl+C: termino el paso actual, guardo y salgo (otra vez para forzar).")
        self.stop_requested = True

    def should_stop(self) -> bool:
        if self.deadline and time.time() >= self.deadline:
            print("[session] Límite de tiempo alcanzado.")
            self.stop_requested = True
        return self.stop_requested

    def time_to_save(self) -> bool:
        if time.time() - self.last_save >= self.cfg.checkpoint_minutes * 60:
            self.last_save = time.time()
            return True
        return False

    def thermal_pause(self) -> bool:
        """Pausa si la GPU pasa del umbral. Devuelve True si hubo pausa (para reducir entornos)."""
        if time.time() - self.last_temp_check < self.cfg.temp_check_seconds:
            return False
        self.last_temp_check = time.time()
        temp = gpu_temperature()
        if temp is None or temp < self.cfg.gpu_temp_pause:
            return False
        self.hot_events += 1
        print(f"[session] GPU a {temp:.0f} °C: pausa hasta {self.cfg.gpu_temp_resume:.0f} °C.")
        while not self.stop_requested:
            time.sleep(30)
            temp = gpu_temperature()
            if temp is None or temp <= self.cfg.gpu_temp_resume:
                break
        return True
