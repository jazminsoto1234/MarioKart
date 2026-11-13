from __future__ import annotations
import multiprocessing as mp
import traceback
from .config import RaceSettings, Participant


def _server(conn, settings):
    backend = None
    try:
        from .backend import PySTKBackend
        backend = PySTKBackend(settings)
        conn.send((True, backend.provenance))
        while True:
            command, payload = conn.recv()
            try:
                if command == "close":
                    conn.send((True, None))
                    break
                if command == "reset":
                    result = backend.reset(**payload)
                elif command == "step":
                    result = backend.step(**payload)
                elif command == "assets":
                    result = backend.assets()
                elif command == "stop":
                    result = backend.stop_race()
                else:
                    raise ValueError(f"Comando desconocido: {command}")
                conn.send((True, result))
            except Exception:
                conn.send((False, traceback.format_exc()))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            conn.send((False, traceback.format_exc()))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if backend is not None:
            backend.close()
        conn.close()


class Engine:
    def __init__(self, settings: RaceSettings, startup_timeout: float = 900.0, step_timeout: float = 60.0):
        self.settings, self.timeout = settings, step_timeout
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(target=_server, args=(child, settings), name="stk-simulator")
        self.process.start()
        child.close()
        try:
            self.provenance = self._receive(startup_timeout)
        except BaseException:
            self.close(force=True)
            raise

    def _receive(self, timeout):
        if not self.conn.poll(timeout):
            raise TimeoutError("El simulador excedió el tiempo de espera; revisa OpenGL, assets y el log nativo")
        try:
            ok, value = self.conn.recv()
        except EOFError as e:
            raise RuntimeError(f"El proceso del motor terminó inesperadamente (exitcode={self.process.exitcode})") from e
        if not ok:
            raise RuntimeError("Error en el simulador:\n" + value)
        return value

    def _call(self, name, **payload):
        if not self.process.is_alive():
            raise RuntimeError(f"El simulador no está activo (exitcode={self.process.exitcode})")
        self.conn.send((name, payload))
        return self._receive(self.timeout)

    # API no bloqueante: permite avanzar varios simuladores en paralelo
    # (enviar a todos y luego recibir de todos). Usado por ppo_rl.
    def send(self, name, **payload):
        if not self.process.is_alive():
            raise RuntimeError(f"El simulador no está activo (exitcode={self.process.exitcode})")
        if name == "reset" and payload.get("settings") is not None:
            self.settings = payload["settings"]
        self.conn.send((name, payload))

    def recv(self, timeout=None):
        return self._receive(self.timeout if timeout is None else timeout)

    def reset(self, participants: list[Participant], settings: RaceSettings | None = None):
        if settings is not None:
            self.settings = settings
        return self._call("reset", settings=self.settings, participants=participants)

    def step(self, actions, render=True):
        return self._call("step", actions=actions, render=render)

    def assets(self):
        return self._call("assets")

    def stop(self):
        return self._call("stop")

    def close(self, force=False):
        if getattr(self, "process", None) is not None:
            if self.process.is_alive() and not force:
                try:
                    self.conn.send(("close", {}))
                    if self.conn.poll(2):
                        self.conn.recv()
                except (OSError, EOFError):
                    pass
            self.process.join(3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
