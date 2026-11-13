from __future__ import annotations
import argparse
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from .config import RaceSettings
from .util import atomic_json, jsonable


def _settings(args):
    return RaceSettings.load(args.config) if getattr(args, "config", None) else RaceSettings()


def parser():
    p = argparse.ArgumentParser(prog="stk-agent", description="Herramientas para agentes visuales en SuperTuxKart.")
    sub = p.add_subparsers(dest="command", required=True)
    def configured(name, help):
        c = sub.add_parser(name, help=help)
        c.add_argument("--config", help="JSON RaceSettings; ver configs/collect.json")
        return c
    d = configured("doctor", "Verificar dependencias y, opcionalmente, el motor real")
    d.add_argument("--engine", action="store_true")
    d.add_argument("--seconds", type=float, default=8)
    d.add_argument("--views", type=int, default=1)
    d.add_argument("--show", action="store_true")
    d.add_argument("--out", default="runs/doctor")
    configured("assets", "Listar pistas y karts del motor instalado")
    sub.add_parser("gamepads", help="Listar mandos y sus ejes/botones")
    c = configured("collect", "Jugar contra bots y registrar RGB, controles y estado privilegiado")
    c.add_argument("--out", default="data/local")
    c.add_argument("--group", required=True)
    c.add_argument("--driver", required=True, help="Identificador del conductor")
    c.add_argument("--episodes", type=int, default=1)
    c.add_argument("--random-seed", action="store_true", help="Compatibilidad: fuerza seed=random aunque el JSON tenga un entero")
    c.add_argument("--tracks", nargs="+")
    c.add_argument("--gamepad", type=int)
    c.add_argument("--keymap", help="Keymap configurable .yaml/.yml/.json para teclado y gamepad")
    c.add_argument("--mapping", help="JSON plano de mapeo del mando")
    c.add_argument("--scale", type=float, default=1.5)
    c.add_argument("--codec", choices=("h264", "ffv1"), default="h264")
    c.add_argument("--publish-dir", help="Carpeta LOCAL sincronizada por Drive; se copia al finalizar")
    v = sub.add_parser("validate", help="Verificar integridad del episodio")
    v.add_argument("episode")
    v.add_argument("--deep", action="store_true", help="Decodificar todo el video y contar frames")
    i = sub.add_parser("inspect", help="Consultar o visualizar un frame del dataset")
    i.add_argument("episode")
    i.add_argument("--frame", type=int, default=0, help="Índice RGB, no tick de 60 Hz")
    i.add_argument("--full-state", action="store_true", help="Imprimir todo el estado privilegiado")
    i.add_argument("--show", action="store_true")
    a = sub.add_parser("audit", help="Auditar un episodio o una carpeta completa de demostraciones")
    a.add_argument("episode", help="Episodio (READY.json) o raíz que contiene varios episodios")
    a.add_argument("--deep", action="store_true")
    a.add_argument("--out", help="Guardar informe JSON")
    r = sub.add_parser("runtime-install", help="Instalar un runtime nativo de Windows en el entorno activo")
    r.add_argument("--source", required=True, help="Carpeta actual STK_DLL_DIR")
    r.add_argument("--llvm-bin", help="Opcional: x86_64-w64-mingw32/bin, NO aarch64")
    x = sub.add_parser("index", help="Indexar episodios finalizados y validados")
    x.add_argument("root")
    x.add_argument("--out", required=True)
    x = sub.add_parser("publish", help="Copiar a carpeta local sincronizada con Drive")
    x.add_argument("episode")
    x.add_argument("--drive-dir", required=True)
    x = sub.add_parser("pack", help="Empaquetar episodio validado para subir manualmente")
    x.add_argument("episode")
    x.add_argument("--out", required=True)
    for name, help in (("play", "Probar un modelo contra bots"), ("race", "Enfrentar hasta cuatro modelos")):
        r = configured(name, help)
        r.add_argument("manifests", nargs="+", help="Ruta(s) a agent.json")
        r.add_argument("--out", required=True, help="Carpeta nueva para resultados/logs")
        r.add_argument("--slots", nargs="+", type=int)
        r.add_argument("--show", action="store_true")
        r.add_argument("--realtime", action="store_true", help="Pacing sin descartar acciones tardías")
        r.add_argument("--video", action="store_true")
        r.add_argument("--action-timeout", type=float, default=10.0)
    l = configured("league", "Generar o ejecutar calendario de mangas y rotaciones")
    l.add_argument("manifests", nargs="+")
    l.add_argument("--out", required=True)
    l.add_argument("--tracks", nargs="+", required=True)
    l.add_argument("--seeds", type=int, nargs="+", required=True)
    l.add_argument("--heat-size", type=int, default=4)
    l.add_argument("--execute", action="store_true", help="Sin esto solo crea schedule.json")
    l.add_argument("--action-timeout", type=float, default=10.0)
    f = sub.add_parser("freeze", help="Guardar versiones instaladas tras validar el equipo")
    f.add_argument("--out", default="requirements.local.lock.txt")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        command = args.command
        if command == "doctor":
            from .diagnostics import package_report, probe
            result = probe(_settings(args), args.out, args.seconds, args.views, args.show) if args.engine else package_report()
        elif command == "assets":
            from .engine import Engine
            with Engine(_settings(args)) as e:
                result = e.assets()
        elif command == "gamepads":
            from .input import gamepads
            result = gamepads()
        elif command == "collect":
            from .collection import collect
            from .input import load_keymap
            if args.keymap and args.mapping:
                raise ValueError("Usa --keymap o --mapping, no ambos")
            keymap = load_keymap(args.keymap) if args.keymap else None
            mapping = json.loads(Path(args.mapping).read_text()) if args.mapping else None
            cfg = _settings(args)
            if args.random_seed:
                cfg = cfg.updated(seed='random')
            result = collect(
                cfg, args.out, args.group, args.driver, args.episodes, args.tracks,
                joystick=args.gamepad, mapping=mapping, keymap=keymap, scale=args.scale,
                codec=args.codec, publish_dir=args.publish_dir,
            )
        elif command == "validate":
            from .recording import validate_episode
            result = validate_episode(args.episode, verify_hashes=True, decode_video=args.deep)
        elif command == "inspect":
            from .dataset import Episode
            ep = Episode(args.episode)
            sample = ep.transition(args.frame, privileged=True)
            state = sample["privileged"]
            result = {"metadata": ep.meta, "rgb_index": args.frame,
                      "frame_id": sample["frame_id"], "next_frame_id": sample["next_frame_id"],
                      "action": sample["action"], "action_sequence": sample["action_sequence"],
                      "substeps": sample["substeps"], "phase": sample["phase"],
                      "privileged": state if args.full_state else {
                          "phase": state.get("phase"), "time": state.get("time"),
                          "num_karts": len(state.get("karts", [])), "num_items": len(state.get("items", [])),
                          "note": "--full-state imprime el contenido completo"}}
            if args.show:
                from .input import HumanUI
                image = sample["observation"]
                ui = HumanUI(ep.settings["width"], ep.settings["height"])
                import time
                try:
                    ui.show(image, f"Frame {args.frame} | RGB | ESC cierra")
                    while not ui.quit:
                        ui.events()
                        time.sleep(0.02)
                finally:
                    ui.close()
        elif command == "audit":
            from .audit import audit_path
            result = audit_path(args.episode,args.deep)
            if args.out: atomic_json(args.out,result)
        elif command == "runtime-install":
            from .runtime import install_runtime
            result = install_runtime(args.source,args.llvm_bin)
        elif command == "index":
            from .transfer import build_index
            result = build_index(args.root, args.out)
        elif command == "publish":
            from .transfer import publish
            result = {"copied_to": str(publish(args.episode, args.drive_dir)),
                      "cloud_upload_confirmed": False, "note": "Comprueba la sincronización de Google Drive Desktop"}
        elif command == "pack":
            from .transfer import pack_episode
            result = {"archive": str(pack_episode(args.episode, args.out))}
        elif command in ("play", "race"):
            from .evaluation import run_race
            if command == "play" and len(args.manifests) != 1:
                raise ValueError("play requiere exactamente un modelo")
            result = run_race(args.manifests, _settings(args), args.out, args.slots,
                              args.show, args.realtime, args.video, args.action_timeout)
        elif command == "league":
            from .evaluation import run_league
            result = run_league(args.manifests, _settings(args), args.out, args.tracks, args.seeds,
                               args.heat_size, args.execute, action_timeout=args.action_timeout)
        elif command == "freeze":
            out = Path(args.out)
            if out.exists():
                raise FileExistsError(out)
            out.write_bytes(subprocess.check_output([sys.executable, "-m", "pip", "freeze"]))
            result = {"lock_file": str(out)}
        else:
            raise ValueError(command)
        print(json.dumps(jsonable(result), ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if isinstance(result, dict) and result.get("valid") is False else 0
    except KeyboardInterrupt:
        print("Interrumpido. Las carpetas .partial no son entregas válidas.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
