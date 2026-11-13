from __future__ import annotations
from pathlib import Path
import os
import shutil
import subprocess
import numpy as np
from .types import check_rgb


def ffmpeg_executable():
    explicit = os.environ.get("STK_FFMPEG")
    if explicit:
        if not Path(explicit).is_file():
            raise FileNotFoundError(explicit)
        return explicit
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _read_exact(stream, n):
    pieces = []
    total = 0
    while total < n:
        block = stream.read(n - total)
        if not block:
            if total:
                raise IOError(f"Frame de video truncado: {total}/{n} bytes")
            return None
        pieces.append(block)
        total += len(block)
    return b"".join(pieces)


class VideoWriter:
    def __init__(self, path, width, height, fps, codec="h264"):
        self.path, self.width, self.height = Path(path), width, height
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames = 0
        self.closed = False
        self.log = self.path.with_suffix(self.path.suffix + ".ffmpeg.log").open("wb")
        args = [ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                "-r", str(fps), "-i", "pipe:0", "-an", "-threads", "2"]
        if codec == "h264":
            args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                     "-pix_fmt", "yuv420p", "-g", str(fps), "-bf", "0", "-movflags", "+faststart"]
        elif codec == "ffv1":
            args += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0"]
        else:
            self.log.close()
            raise ValueError("codec debe ser h264 o ffv1")
        args.append(str(self.path))
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)

    def write(self, rgb):
        if self.closed:
            raise RuntimeError("Video cerrado")
        a = check_rgb(rgb, self.width, self.height)
        if self.process.poll() is not None:
            raise RuntimeError(f"FFmpeg terminó; revisa {self.log.name}")
        self.process.stdin.write(np.ascontiguousarray(a).tobytes())
        self.frames += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        error = None
        try:
            self.process.stdin.close()
        except BrokenPipeError as e:
            error = e
        try:
            code = self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError("FFmpeg no finalizó el archivo")
        finally:
            self.log.close()
        if code or error:
            raise RuntimeError(f"FFmpeg falló con código {code}; revisa {self.log.name}")


def iter_video(path, width, height, start=0, stop=None):
    """Decodificación exacta por índice, no por aproximación de timestamps.

    Para lecturas aleatorias se decodifica desde el inicio (correcto pero más
    costoso). Para entrenamiento masivo usa Episode.iter_transitions().
    """
    if start < 0 or (stop is not None and stop < start):
        raise ValueError("Rango de video inválido")
    args = [ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-map", "0:v:0", "-threads", "1"]
    if start or stop is not None:
        expression = f"gte(n,{start})" if stop is None else f"between(n,{start},{stop-1})"
        args += ["-vf", f"select='{expression}'", "-vsync", "0"]
    args += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    # Archivo temporal evita el deadlock de stderr=PIPE en un video dañado.
    import tempfile
    with tempfile.TemporaryFile() as err:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=err)
        complete = False
        try:
            while True:
                data = _read_exact(process.stdout, width * height * 3)
                if data is None:
                    break
                yield np.frombuffer(data, np.uint8).reshape(height, width, 3).copy()
            complete = True
            code = process.wait(timeout=30)
            if code:
                err.seek(0)
                raise RuntimeError("No se pudo decodificar video: " + err.read(8000).decode(errors="replace"))
        finally:
            process.stdout.close()
            if not complete and process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
