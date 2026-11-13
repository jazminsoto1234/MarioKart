# ppo_rl

PPO asimétrico para SuperTuxKart (ver `PLAN.md`). Todos los comandos se reanudan volviendo a ejecutarlos.

## Instalar (una vez)

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install tensorboard psutil
```

## Módulos

| Archivo | Qué hace |
|---|---|
| `config.py` | Dataclasses de configuración + overrides `--set sección.clave=valor` |
| `features.py` | Vector privilegiado, máscaras, acción política ↔ motor |
| `reward.py` | Recompensa con desglose por término |
| `vec_env.py` | `VecRace`: N simuladores en lockstep, escenarios por cuota, rescue por regla |
| `preprocess.py` | Resize 72×128 y frame stack (compartido por todo) |
| `model.py` | IMPALA-CNN, actor híbrido, crítico V(h, s), KL por cabeza |
| `session.py` | `--max-hours`, Ctrl+C seguro, checkpoints atómicos, watchdog de GPU |
| `offline.py` | Etapas 1–2: caché a 15 Hz y AWR-BC + crítico |
| `ppo.py` | Etapas 3–4: PPO con warmup, rampa de lr y KL hacia π_BC |
| `evaluate.py` | Matriz de evaluación con Wilson e IQM |
| `bench.py` | Etapa 0: decisiones/s, RAM y temperatura |
| `ingest.py` | Etapa 0b: inventario, auditoría y split por grupo |
| `agent.py` | Agente de despliegue (solo RGB) para `stk_agent` |

## Comandos por etapa (desde la raíz del proyecto)

```bash
python -m ppo_rl bench --envs 2 4 6
python -m ppo_rl ingest inventory data/drive --out ppo_rl/cache/inventory.csv
python -m ppo_rl ingest audit data/drive --out ppo_rl/cache/audit.json
python -m ppo_rl ingest split ppo_rl/cache/inventory.csv --out ppo_rl/cache/split.json --audit ppo_rl/cache/audit.json
python -m ppo_rl build-dataset --config ppo_rl/configs/pretrain.yaml
python -m ppo_rl pretrain --config ppo_rl/configs/pretrain.yaml --max-hours 2
python -m ppo_rl train --config ppo_rl/configs/stage4a.yaml --set "env.train_tracks=[lighthouse]" run_name=stage4a_lighthouse
python -m ppo_rl train --config ppo_rl/configs/stage4a.yaml --max-hours 2
python -m ppo_rl evaluate --config ppo_rl/configs/stage4a.yaml
python -m stk_agent play ppo_rl/agent.json --config configs/evaluation.json --out runs/play_01
```

Métricas: `runs/ppo_rl/<run>/metrics.jsonl` y `tensorboard --logdir runs/ppo_rl`.
