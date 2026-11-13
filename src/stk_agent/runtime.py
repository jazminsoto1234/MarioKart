"""Native runtime helpers."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .util import atomic_json, sha256

_NATIVE_HANDLES = []
STK_RUNTIME_URL = "https://github.com/supertuxkart/dependencies/releases/download/preview/dependencies-win-x86_64.zip"
LLVM_RUNTIME_URL = "https://github.com/mstorsjo/llvm-mingw/releases/download/20260616/llvm-mingw-20260616-msvcrt-x86_64.zip"


def runtime_root():
    return Path(sys.prefix) / "share" / "stk_agent" / "native"


def _legacy_runtime_root():
    return Path(sys.prefix) / "share" / "stk_course" / "native"


def pe_machine(path):
    with Path(path).open("rb") as f:
        if f.read(2) != b"MZ":
            raise ValueError(f"No es PE: {path}")
        f.seek(0x3C)
        offset = struct.unpack("<I", f.read(4))[0]
        f.seek(offset)
        if f.read(4) != b"PE\x00\x00":
            raise ValueError(f"Cabecera PE inválida: {path}")
        return struct.unpack("<H", f.read(2))[0]


def _runtime_from_root(root):
    pointer = root / "active.json"
    if not pointer.is_file():
        return None
    data = json.loads(pointer.read_text(encoding="utf-8"))
    folder = (root / data["directory"]).resolve()
    if folder.parent != root.resolve() or not folder.is_dir():
        raise RuntimeError(f"Runtime nativo inválido: {folder}")
    return folder


def get_runtime_directory():
    explicit = os.environ.get("STK_DLL_DIR")
    if explicit:
        folder = Path(explicit).expanduser().resolve()
        if not folder.is_dir():
            raise RuntimeError(f"STK_DLL_DIR no existe: {folder}")
        return folder
    current = _runtime_from_root(runtime_root())
    if current is not None:
        return current
    return _runtime_from_root(_legacy_runtime_root())


def register_dll_directory():
    if os.name != "nt":
        return []
    folder = get_runtime_directory()
    if folder is None:
        return []
    handle = os.add_dll_directory(str(folder))
    _NATIVE_HANDLES.append(handle)
    return [handle]


def install_runtime(source, llvm_bin=None):
    if os.name != "nt" or struct.calcsize("P") != 8:
        raise RuntimeError("El runtime nativo solo aplica a Windows x86-64")
    source = Path(source).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    root = runtime_root()
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    try:
        for file in source.glob("*.dll"):
            shutil.copy2(file, stage / file.name)
        if llvm_bin:
            llvm_bin = Path(llvm_bin)
            for name in ("libc++.dll", "libunwind.dll"):
                shutil.copy2(llvm_bin / name, stage / name)
        if not (stage / "libzlib.dll").exists() and (stage / "libz.dll").exists():
            shutil.copy2(stage / "libz.dll", stage / "libzlib.dll")
        required = (
            "libzlib.dll", "libpng16.dll", "libcurl.dll", "libmbedcrypto.dll",
            "libjpeg-62.dll", "libfreetype.dll", "libharfbuzz.dll", "libastcenc.dll",
            "SDL2.dll", "libshaderc_shared.dll", "libc++.dll", "libunwind.dll",
        )
        missing = [name for name in required if not (stage / name).is_file()]
        if missing:
            raise RuntimeError(f"Faltan DLL: {missing}")
        files = list(stage.glob("*.dll"))
        for file in files:
            machine = pe_machine(file)
            if machine != 0x8664:
                raise RuntimeError(f"{file.name}: se exige x86-64")
        probe = (
            "import os,sys,importlib.metadata; "
            "h=os.add_dll_directory(sys.argv[1]); import pystk2; "
            "print(importlib.metadata.version('PySuperTuxKart2'),pystk2.__version__)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe, str(stage)],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode:
            raise RuntimeError("El runtime no pasó la prueba de importación:\n" + proc.stdout + proc.stderr)
        hashes = {f.name: sha256(f) for f in files}
        identifier = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
        final = root / ("win64-" + identifier)
        atomic_json(stage / "manifest.json", {
            "architecture": "x86_64", "files": hashes, "import_probe": proc.stdout.strip()
        })
        if final.exists():
            shutil.rmtree(stage)
        else:
            os.replace(stage, final)
        atomic_json(root / "active.json", {"directory": final.name})
        return {"installed": str(final), "dll_count": len(files), "import_probe": proc.stdout.strip()}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "stk-agent/0.5"})
    with urllib.request.urlopen(request, timeout=120) as response, Path(destination).open("wb") as out:
        shutil.copyfileobj(response, out)


def _find_parent(root, filename, must_contain=None):
    for path in Path(root).rglob(filename):
        if must_contain is None or must_contain in path.parts:
            return path.parent
    raise RuntimeError(f"No se encontró {filename} en el runtime descargado")


def bootstrap_windows_runtime():
    """Downloads and validates the native runtime required by the Windows wheel."""
    if os.name != "nt":
        return None
    existing = get_runtime_directory()
    if existing is not None:
        return {"installed": str(existing), "reused": True}
    print("Preparando runtime nativo de SuperTuxKart...", file=sys.stderr, flush=True)
    with tempfile.TemporaryDirectory(prefix="stk-agent-runtime-") as tmp:
        tmp = Path(tmp)
        stk_zip = tmp / "stk.zip"
        llvm_zip = tmp / "llvm.zip"
        _download(STK_RUNTIME_URL, stk_zip)
        _download(LLVM_RUNTIME_URL, llvm_zip)
        stk_dir = tmp / "stk"
        llvm_dir = tmp / "llvm"
        with zipfile.ZipFile(stk_zip) as z:
            z.extractall(stk_dir)
        with zipfile.ZipFile(llvm_zip) as z:
            z.extractall(llvm_dir)
        stk_bin = _find_parent(stk_dir, "SDL2.dll")
        llvm_bin = _find_parent(llvm_dir, "libc++.dll", "x86_64-w64-mingw32")
        return install_runtime(stk_bin, llvm_bin)
