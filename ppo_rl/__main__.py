"""CLI: python -m ppo_rl <comando> [--config X.yaml] [--set sección.clave=valor] [--max-hours H]"""
from __future__ import annotations

import argparse
import sys

from .config import load_config


# ---------------------------------------------------------------- comandos

def cmd_bench(cfg, args):
    from .bench import bench
    res = [tuple(map(int, r.lower().split("x"))) for r in args.res]
    bench(cfg, args.envs, res, args.seconds, args.out)


def cmd_ingest(cfg, args):
    from . import ingest
    if args.action == "inventory":
        ingest.inventory(args.root, args.out)
    elif args.action == "audit":
        ingest.audit(args.root, args.out)
    else:
        ingest.make_split(args.root, args.out, cfg.env.heldout_tracks, args.val_groups, audit_json=args.audit)


def cmd_build_dataset(cfg, args):
    from .offline import build_cache
    from .session import Session
    with Session(cfg.session) as s:
        build_cache(cfg, s)


def cmd_pretrain(cfg, args):
    from .offline import pretrain
    from .session import Session
    with Session(cfg.session) as s:
        pretrain(cfg, s)


def cmd_train(cfg, args):
    from .ppo import train
    from .session import Session
    with Session(cfg.session) as s:
        train(cfg, s)


def cmd_evaluate(cfg, args):
    from .evaluate import evaluate
    from .session import Session
    out = args.out or str(cfg.run_dir / "eval")
    with Session(cfg.session) as s:
        evaluate(args.checkpoint or str(cfg.run_dir / "policy.pt"), out, s, overrides=cfg)


# ---------------------------------------------------------------- parser

def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="YAML/JSON (p. ej. ppo_rl/configs/stage4a.yaml)")
    common.add_argument("--set", nargs="*", default=[], metavar="K=V", help="overrides: ppo.num_envs=4")
    common.add_argument("--max-hours", type=float, help="para solo al llegar al límite")
    common.add_argument("--until", help="HH:MM hora local de parada")

    p = argparse.ArgumentParser(prog="python -m ppo_rl")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", parents=[common], help="Etapa 0: dec/s, RAM y temperatura")
    b.add_argument("--envs", type=int, nargs="+", default=[2, 4, 6])
    b.add_argument("--res", nargs="+", default=["640x360", "320x180"])
    b.add_argument("--seconds", type=float, default=60)
    b.add_argument("--out", default="runs/ppo_rl/bench.json")
    b.set_defaults(fn=cmd_bench)

    i = sub.add_parser("ingest", parents=[common], help="Etapa 0b: inventory | audit | split")
    i.add_argument("action", choices=["inventory", "audit", "split"])
    i.add_argument("root", help="carpeta de episodios (inventory/audit) o inventory.csv (split)")
    i.add_argument("--out", required=True)
    i.add_argument("--val-groups", type=float, default=0.15)
    i.add_argument("--audit", default="", help="audit.json para descartar inválidos (split)")
    i.set_defaults(fn=cmd_ingest)

    sub.add_parser("build-dataset", parents=[common], help="Etapa 1: shards a 15 Hz").set_defaults(fn=cmd_build_dataset)
    sub.add_parser("pretrain", parents=[common], help="Etapa 2: AWR-BC + crítico").set_defaults(fn=cmd_pretrain)
    sub.add_parser("train", parents=[common], help="Etapas 3–5: PPO").set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate", parents=[common], help="Matriz de evaluación (§6)")
    e.add_argument("--checkpoint", help="policy.pt (por defecto el del run)")
    e.add_argument("--out", help="carpeta de resultados (por defecto <run>/eval)")
    e.set_defaults(fn=cmd_evaluate)
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # consola de Windows redirigida usa cp1252
        stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.max_hours is not None:
        cfg.session.max_hours = args.max_hours
    if args.until:
        cfg.session.until = args.until
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
