# Guía de ejecución paso a paso

Cómo correr cada parte de `ppo_rl`, qué necesita cada paso (**input**) y qué deja (**output**).

---

## Antes de empezar (cada vez que abras PowerShell)

```powershell
cd C:\Users\JAZMIN\Desktop\MarioCar
.venv\Scripts\Activate.ps1
```

Todos los comandos se escriben **desde esa carpeta** (la raíz del proyecto). Si ves `(.venv)` al inicio de la línea, estás listo.

Solo la primera vez, instala las dependencias:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install tensorboard psutil
python -c "import torch; print(torch.cuda.is_available())"   # debe imprimir True
```

---

## La idea general

No ejecutas los archivos `.py` uno por uno. Todo pasa por un solo comando, `python -m ppo_rl <paso>`, y cada paso usa los archivos que necesita.

```
 TÚ PONES                     EL PROGRAMA GENERA
 ─────────                    ──────────────────
 data/local/  ──(1) build-dataset──▶  ppo_rl/cache/
                                          │
                     (2) pretrain ◀───────┘
                          │
                          ▼
              runs/ppo_rl/pretrain/bc.pt
                          │
                     (3) train
                          │
                          ▼
              runs/ppo_rl/stage4a/policy.pt
                    │              │
          (4) evaluate        (5) stk_agent play
                    │              │
                    ▼              ▼
      runs/ppo_rl/stage4a/eval/   runs/play_01/
```

El único input que das tú son **las partidas grabadas** en `data/local/`. Todo lo demás lo produce el paso anterior.

### Qué archivo `.py` usa cada paso

| Paso | Comando | Archivos que trabajan |
|---|---|---|
| 0 | `bench` | `bench.py`, `vec_env.py` |
| 0b | `ingest` | `ingest.py` |
| 1 | `build-dataset` | `offline.py`, `features.py`, `reward.py`, `preprocess.py` |
| 2 | `pretrain` | `offline.py`, `model.py`, `session.py` |
| 3 | `train` | `ppo.py`, `model.py`, `vec_env.py`, `session.py` |
| 4 | `evaluate` | `evaluate.py`, `vec_env.py`, `model.py` |
| 5 | `stk_agent play` | `agent.py` (+ `agent.json`) |

`config.py` lo usan todos: lee el YAML de `ppo_rl/configs/` y aplica los `--set`.

---

## Paso 0 · Benchmark (ya lo hiciste)

**Qué hace:** abre varios simuladores, maneja al azar y mide cuántas decisiones por segundo aguanta tu laptop.

```powershell
python -m ppo_rl bench --envs 2 4 6
```

| | |
|---|---|
| **Input** | Nada. Solo el juego instalado. |
| **Output** | `runs/ppo_rl/bench.json` |
| **Tarda** | ~1 min por prueba (6 pruebas ≈ 7 min) |

**Qué verás:**
```
[bench] 4 entornos a 640x360 ...
[bench]   189 dec/s (~0.68 M/h), RAM 2.2 GB, GPU 50.0 °C, arranque 7.6 s, reset 1.8 s
```
- `dec/s`: decisiones por segundo. Más es mejor.
- `M/h`: millones de pasos por hora de entrenamiento.

Las líneas `Antarctica Rendering Engine` y `Failed to init SDL sensor` son normales y no son errores.

**Tu resultado:** 6 entornos a 640×360 dan 215 dec/s con 3 GB de RAM. Usa `ppo.num_envs=6` (paso 3).

---

## Paso 0b · Ingesta del Drive (opcional, solo si bajas más partidas)

Sáltalo si solo vas a usar las 10 partidas de `data/local`.

**Qué hace:** revisa las partidas descargadas del Drive, descarta las incompletas y las reparte en entrenamiento/validación por grupo.

```powershell
python -m ppo_rl ingest inventory data/drive --out ppo_rl/cache/inventory.csv
python -m ppo_rl ingest audit     data/drive --out ppo_rl/cache/audit.json
python -m ppo_rl ingest split ppo_rl/cache/inventory.csv --out ppo_rl/cache/split.json --audit ppo_rl/cache/audit.json
```

| Comando | Input | Output |
|---|---|---|
| `inventory` | Carpeta con partidas (`data/drive/<grupo>/<episode_id>/meta.json`) | `inventory.csv`: una fila por partida (grupo, pista, duración, si tiene video, etc.) |
| `audit` | La misma carpeta | `audit.json`: qué partidas están sanas y cuáles rotas |
| `split` | `inventory.csv` (+ `audit.json`) | `split.json`: listas `train`, `val` y `heldout_track` |

Para que el paso 1 use ese split, agrega a los comandos 1 y 2:
```powershell
--set offline.split_file=ppo_rl/cache/split.json
```

---

## Paso 1 · Construir el dataset

**Qué hace:** toma cada partida humana y la convierte a lo que ve la red: imágenes de 72×128 a 15 por segundo, la acción que hizo el humano y la recompensa de cada momento.

```powershell
python -m ppo_rl build-dataset --config ppo_rl/configs/pretrain.yaml
```

| | |
|---|---|
| **Input** | `data/local/` — cada partida tiene `rgb.mp4`, `data.h5`, `track.npz`, `meta.json` |
| **Output** | `ppo_rl/cache/shards/<episode_id>/` (una carpeta por partida) y `ppo_rl/cache/index.json` |
| **Tarda** | ~30 s por partida (10 partidas ≈ 5 min) |

**Qué hay dentro de cada carpeta de `shards`:**

| Archivo | Contenido |
|---|---|
| `frames.npy` | Las imágenes chicas (N × 3 × 72 × 128) |
| `actions.npy` | Lo que hizo el humano: steer, gas/freno, drift, fire, nitro |
| `prev_actions.npy` | La acción del paso anterior |
| `priv.npy` | 98 números del estado del juego (velocidad, posición, rivales…) para el crítico |
| `mask.npy` | Qué botones se podían usar (p. ej. fire solo con ítem) |
| `reward.npy` / `returns.npy` | Recompensa de cada paso y la suma futura |
| `fail.npy` | Marca los 1,5 s antes de un choque o rescate (no se imitan) |
| `meta.json` | Resumen: pista, pasos, retorno total, suma de cada término de recompensa |

**Qué verás:**
```
[cache] 1/10 minigolf: 2392 pasos, retorno -232.3, fallo 19.2%
...
[cache] 10 episodios, 24000 pasos. Retorno MC: media -9.12, ...
```

**Si lo cortas:** vuelve a correr el mismo comando. Las partidas ya hechas se saltan.

---

## Paso 2 · Preentrenamiento (imitar al humano)

**Qué hace:** entrena la red para que **copie** al humano (BC), dando más peso a las jugadas buenas, y entrena al crítico para que prediga la recompensa futura.

```powershell
python -m ppo_rl pretrain --config ppo_rl/configs/pretrain.yaml --max-hours 2
```

| | |
|---|---|
| **Input** | `ppo_rl/cache/` (del paso 1) |
| **Output** | Carpeta `runs/ppo_rl/pretrain/` |
| **Tarda** | 30 épocas; en la RTX 5060, del orden de minutos por época con los datos locales |

**Qué queda en `runs/ppo_rl/pretrain/`:**

| Archivo | Para qué |
|---|---|
| **`bc.pt`** | **El resultado.** La red que imita al humano. Se actualiza al final de cada época. |
| `pretrain_last.pt` (+ `.1.pt`, `.2.pt`) | Puntos de guardado para continuar si cortas. No los toques. |
| `metrics.jsonl` | Historial de pérdidas |
| `tb/` | Gráficas para TensorBoard |

**Qué verás:**
```
[pretrain] train 19500 pasos (8 ep.), val 2400 (1 ep.)
[pretrain] ep 0 b 50/337 bc_nll=1.255 v_loss=0.310 aux=0.874 w_mean=1.000 steer_mae=0.496
[pretrain] fin época 1: val/bc_nll=... val/steer_mae=...
```
- `bc_nll`: qué tan mal imita. **Debe bajar.**
- `v_loss`: error del crítico. **Debe bajar.**
- `steer_mae`: error promedio del volante (0 = perfecto, 1 = muy malo).

De las 10 partidas, la de `ravenbridge_mansion` no se usa: es una pista reservada para el test. Otra se aparta para validación (`val/...`).

**Si lo cortas:** Ctrl+C una vez (guarda y sale) o espera a `--max-hours`. Para seguir, corre el mismo comando y verás `Reanudando en época X`.

---

## Paso 2b · Ver jugar a la red imitadora (recomendado)

Antes de entrenar con PPO, comprueba que `bc.pt` al menos maneja.

1. Abre `ppo_rl/agent.json` y cambia la línea `checkpoint`:
   ```json
   "checkpoint": "../runs/ppo_rl/pretrain/bc.pt",
   ```
2. Ejecuta:
   ```powershell
   python -m stk_agent play ppo_rl/agent.json --config configs/evaluation.json --out runs/play_bc_01 --show
   ```

| | |
|---|---|
| **Input** | `ppo_rl/agent.json` (que apunta a `bc.pt`) y `configs/evaluation.json` (pista snowmountain, 3 vueltas, 8 karts) |
| **Output** | `runs/play_bc_01/result.json` (posición, si terminó, tiempos) y `agent_0.log` |

- `--show` abre una ventana para verla manejar.
- Agrega `--video` para guardar `view_0.mp4`.
- `--out` debe ser **una carpeta nueva** cada vez (`play_bc_02`, `play_bc_03`…).

Si no completa ni una vuelta, no sigas a PPO: revisa primero el paso 1 y el 2 (el plan, §8, lo explica).

---

## Paso 3 · Entrenar con PPO (la red practica sola)

**Qué hace:** abre 6 simuladores, la red maneja, recibe recompensa y mejora. Empieza desde `bc.pt`. Las primeras 30 actualizaciones solo aprende el crítico (Etapa 3); después aprende el actor (Etapa 4A, contrarreloj).

### 3a · Primero en una sola pista (para validar la recompensa)

```powershell
python -m ppo_rl train --config ppo_rl/configs/stage4a.yaml --set ppo.num_envs=6 ppo.rollout_steps=384 "env.train_tracks=[lighthouse]" run_name=stage4a_lighthouse --max-hours 2
```

### 3b · Después en todas las pistas

```powershell
python -m ppo_rl train --config ppo_rl/configs/stage4a.yaml --set ppo.num_envs=6 ppo.rollout_steps=384 --max-hours 2
```

| | |
|---|---|
| **Input** | `runs/ppo_rl/pretrain/bc.pt` (lo dice `init_checkpoint` en `stage4a.yaml`) + el juego |
| **Output** | `runs/ppo_rl/<run_name>/` (`stage4a_lighthouse` o `stage4a`) |
| **Tarda** | ~0,77 M pasos/h con 6 entornos. Una pista: 3–6 h en total. Todas: 13–26 h. Se hace en bloques de 2 h. |

**Qué queda en `runs/ppo_rl/stage4a/`:**

| Archivo | Para qué |
|---|---|
| **`policy.pt`** | **El resultado.** La red entrenada (lo que usa el agente). |
| `last.pt` (+ `.1.pt`, `.2.pt`) | Guardado completo para continuar. No los toques. |
| `metrics.jsonl`, `tb/` | Métricas y gráficas |

**Qué verás (una línea por actualización):**
```
[ppo] Pesos iniciales: runs/ppo_rl/pretrain/bc.pt
[ppo] u12 24,576 pasos | 210 sps | ret -3.21 | fin 0.40 | kl 0.0081 | ev 0.62
```

| Campo | Significa | Qué esperar |
|---|---|---|
| `u12` | Actualización número 12 | Sube de 1 en 1 |
| `24,576 pasos` | Decisiones jugadas en total | |
| `sps` | Decisiones por segundo | ~200 con 6 entornos |
| `ret` | Recompensa promedio por carrera | **Debe subir** con el tiempo |
| `fin` | Fracción de carreras terminadas (0 a 1) | **Debe subir.** Objetivo de 4A: ≥ 0,9 |
| `kl` | Cuánto cambió la red en esta actualización | Pequeño (< 0,03) |
| `ev` | Qué tan bien predice el crítico (≤ 1) | > 0,5 es bueno |

`ret` y `fin` salen `nan` mientras ninguna carrera haya terminado todavía; es normal al inicio.

**Ver gráficas mientras entrena** (en otra ventana de PowerShell con el `.venv` activado):
```powershell
tensorboard --logdir runs/ppo_rl
```
y abre http://localhost:6006. Las más útiles son `ep/return`, `ep/finished`, `terms/*` (cada parte de la recompensa) y `std/steer`.

**Si lo cortas:** Ctrl+C **una vez** y espera: termina la ronda actual, guarda y cierra los simuladores. Para seguir, corre **el mismo comando** y verás `Reanudando: update X`.

### 3c · Etapas siguientes (cuando 4A termine ≥ 90 % de las carreras)

```powershell
python -m ppo_rl train --config ppo_rl/configs/stage4b.yaml --set ppo.num_envs=6 ppo.rollout_steps=384 --max-hours 2
python -m ppo_rl train --config ppo_rl/configs/stage4c.yaml --set ppo.num_envs=6 ppo.rollout_steps=384 --max-hours 2
```

| Etapa | Input | Output | Qué añade |
|---|---|---|---|
| 4B | `runs/ppo_rl/stage4a/policy.pt` | `runs/ppo_rl/stage4b/` | Rivales (tráfico) |
| 4C | `runs/ppo_rl/stage4b/policy.pt` | `runs/ppo_rl/stage4c/` | Ítems (fire y nitro) |

---

## Paso 4 · Evaluar con números

**Qué hace:** corre muchas carreras con la red (sin azar) en distintas condiciones y calcula tasa de victoria, carreras terminadas, posición media, etc.

La matriz completa son 48 combinaciones × 50 carreras: **demasiado** para una tarde. Empieza con una versión chica:

```powershell
python -m ppo_rl evaluate --config ppo_rl/configs/stage4a.yaml --set eval.races_per_cell=10 eval.num_envs=6 "eval.difficulties=[1]" "eval.items=[false]" "eval.start=[random]"
```

Eso son 4 combinaciones (pistas de entrenamiento/reservadas × 4/8 karts) × 10 carreras.

| | |
|---|---|
| **Input** | `runs/ppo_rl/stage4a/policy.pt` (otro: `--checkpoint ruta/al/archivo.pt`) |
| **Output** | `runs/ppo_rl/stage4a/eval/` (otra carpeta: `--out ruta`) |

**Qué queda en `eval/`:**
- Un `.json` por combinación, p. ej. `heldout_d1_n8_noitems_random.json`, con cada carrera.
- **`summary.json`**: la tabla resumen. Es el que hay que leer.

Campos de `summary.json`:

| Campo | Significa |
|---|---|
| `win_rate` | Fracción de carreras ganadas |
| `win_ci95` | Rango probable real de esa tasa (95 %) |
| `finished_rate` | Fracción de carreras terminadas |
| `rank_mean` | Posición media (1 = primero) |
| `offroad_frac` | Fracción del tiempo fuera de pista |
| `finish_time_iqm` | Tiempo típico de carrera (solo las terminadas) |

`heldout` son pistas que la red **nunca vio**: comparar con `train` dice si generaliza.

**Si lo cortas:** vuelve a correrlo; las combinaciones ya guardadas se saltan.

---

## Paso 5 · Ver jugar a la red final

Igual que el paso 2b, pero con `policy.pt`:

1. En `ppo_rl/agent.json`:
   ```json
   "checkpoint": "../runs/ppo_rl/stage4a/policy.pt",
   ```
2. Ejecuta:
   ```powershell
   python -m stk_agent play ppo_rl/agent.json --config configs/evaluation.json --out runs/play_4a_01 --show --video
   ```

| | |
|---|---|
| **Input** | `agent.json` → `policy.pt`; `configs/evaluation.json` |
| **Output** | `runs/play_4a_01/result.json`, `agent_0.log`, `view_0.mp4` |

Esta es la prueba "oficial": el agente solo recibe la imagen, como en la evaluación del curso.

---

## Resumen rápido

| # | Comando | Input | Output principal |
|---|---|---|---|
| 0 | `bench` | — | `runs/ppo_rl/bench.json` |
| 1 | `build-dataset` | `data/local/` | `ppo_rl/cache/` |
| 2 | `pretrain` | `ppo_rl/cache/` | `runs/ppo_rl/pretrain/bc.pt` |
| 3 | `train` (4A) | `bc.pt` | `runs/ppo_rl/stage4a/policy.pt` |
| 4 | `evaluate` | `policy.pt` | `runs/ppo_rl/stage4a/eval/summary.json` |
| 5 | `stk_agent play` | `agent.json` → `.pt` | `runs/play_xx/result.json` (+ video) |

## Trucos

- **Cambiar un valor sin tocar el YAML:** `--set seccion.clave=valor`, p. ej. `--set ppo.num_envs=4`. Las listas van entre comillas: `"env.train_tracks=[lighthouse, minigolf]"`.
- **Parar a una hora:** `--until 23:30` en lugar de `--max-hours`.
- **Ctrl+C dos veces** sale sin guardar: evítalo.
- **La laptop va lenta mientras entrena:** usa `--set ppo.num_envs=4 ppo.rollout_steps=512`.
- **Se calienta:** el programa pausa solo si la GPU pasa de 85 °C y sigue al bajar de 75 °C.
