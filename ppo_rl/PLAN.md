# Plan de implementación: PPO asimétrico para SuperTuxKart

Basado en `reports/Entrenamiento PPO autónomo en SuperTuxKart.md` y en una revisión del código de `src/stk_agent` y de los datos de `data/local`.

**Modo de ejecución elegido:**
- **Base:** todo en la laptop, **por bloques de 1–3 h que se pueden pausar y reanudar**, con ≤ 4–6 simuladores (§10).
- **Opcional:** Kaggle (30 h de GPU por semana) solo para la Etapa 2, el preentrenamiento offline (§9). PPO y la evaluación siempre son locales.

Todas las cifras de entornos, batch y tiempos de este documento asumen ese modo.

---

## 0. Hechos del proyecto que condicionan el diseño

| Hecho | Consecuencia para el plan |
|---|---|
| El agente desplegado solo recibe `act(rgb)` (`stk_agent/worker.py`, `evaluation.py`), a 640×360 y `policy_hz=30`. | El actor **no puede** recibir velocidad, ítem ni posición numéricos. Sus únicas entradas legales son los frames (con HUD) y sus propias acciones previas. La velocidad se infiere del frame stack y del velocímetro del HUD. |
| El motor (`Engine`) ya corre en un subproceso por simulador y pystk2 solo admite un `init` por proceso. | El entorno vectorizado no debe envolver procesos en otros procesos: un `Engine` por entorno, avanzados en *lockstep* (enviar a todos y luego recibir de todos). Hay que añadir a `Engine` un par `send()`/`recv()` no bloqueante. |
| `MultiAgentEnv.step` renderiza en **cada** tick y `reset` solo acepta `track/reverse/difficulty/laps`. | Conviene un entorno propio para RL que renderice solo en el último tick de cada decisión y que pueda cambiar `num_karts` y el slot de salida. |
| El estado privilegiado trae `overall_distance`, `distance_down_track`, `is_on_road`, `velocity`, `front`, `position`, `energy`, `powerup`, `attachment`, rivales e ítems. La pista trae `path_nodes`, `path_distance` y `path_width`. | Bastan para la recompensa, el crítico y los rayos geométricos. **No hay una señal de choque contra la pared**: se usa como aproximación una caída brusca de velocidad sin frenar. El campo `speed` vale 0 en los datos, así que la rapidez se calcula con `‖velocity‖`. |
| Datos humanos: 10 episodios, 1 conductor, 10 pistas (una carrera de 3 vueltas por pista, 8 karts, dificultad 2), unos 118 k ticks, es decir **unos 33 min de juego**, grabados a teclado. | Alcanzan para un BC inicial y para preentrenar el crítico. Son pocos para generalizar: el BC solo servirá de arranque y lo principal lo hará PPO. Conviene grabar más (las dos direcciones de pista y otras dificultades). |
| Medido aquí: 2 simuladores a 640×360 y decisión a 15 Hz dan unas **133 decisiones/s** (unos 530 ticks/s), con arranque de 6,5 s y `reset` de 2,4 s. | Con 4 entornos (el límite cómodo por RAM, §10) se esperan unas 250 decisiones/s, es decir ≈ 1 M pasos por hora; con 6, algo más. Hay que medirlo en la Etapa 0. Si no escala, se baja el render a 320×180. |

### 0b. Ampliar los datos humanos con los videos del Drive

Lo que se puede sacar de esos videos depende de **en qué formato están**:

| Formato en el Drive | Qué aporta | Cómo entra al pipeline |
|---|---|---|
| **A. Episodios de `stk-agent collect`** (carpetas con `meta.json`, `data.h5`, `rgb.mp4`, `track.npz`, `READY.json`, p. ej. subidos con `--publish-dir`) | Lo mismo que los 33 min actuales: frames, acciones a 60 Hz, estado privilegiado y pista. | 1) Sincronizar o copiar a una carpeta local, p. ej. `data/drive/`. 2) `stk-agent audit data/drive --out audit.json` y descartar los inválidos. 3) Deduplicar por `episode_id`. 4) Comprobar que `width/height/tick_hz/rgb_hz` coinciden (640×360, 60/30). 5) Entran directo a la Etapa 1. **Es el caso ideal.** |
| **B. Solo video** (grabaciones de pantalla, mp4 sueltos, sin `data.h5`) | Solo píxeles: sin acciones, sin estado y por tanto sin recompensa. | No sirven para el BC ni para el crítico tal cual. Hay dos usos, de menor a mayor esfuerzo: **B1. Preentrenar el encoder** de forma autosupervisada (predecir el frame siguiente o contrastivo) y luego hacer BC sobre los datos etiquetados. Es seguro y barato. **B2. Etiquetado estilo VPT:** entrenar un *modelo de dinámica inversa* (IDM) con los episodios etiquetados, que a partir de los frames t−k…t+k predice la acción en t; con él se etiquetan los videos y se hace BC. Es arriesgado con solo 33 min de entrenamiento para el IDM. Requiere misma resolución/HUD/cámara y que el video sea de la versión de STK del motor. Sin estado no hay ventaja para ponderar: como máximo se filtra por calidad leyendo el rango final del HUD. |
| **C. Mezcla** | – | Se separa: A al pipeline completo y B a B1. |

**Reglas para el split y la calidad (caso A):**
- El split de validación se hace **por conductor y por pista**, no por frames, para medir si el BC generaliza.
- Varios conductores de calidad distinta es justo el caso para el que se pensó el BC ponderado por ventaja: los errores entrenan al crítico y las buenas jugadas al actor.
- Más horas en las mismas pistas y en el mismo sentido rinden poco (en PIRLNav, la mejora del RL bajó de +10,1 a +6,3 puntos al pasar de 20 k a 77 k demos). Rinde más cubrir pistas en `reverse`, dificultades 0–1, salidas desde el último puesto y recuperaciones.
- Hay que mantener fuera del BC los episodios de las **pistas reservadas para test**.

#### Lo que hay en el Drive "MarioKart" (revisado el 12-11-2026, sin iniciar sesión)
- **Caso A confirmado.** Hay unas **26 carpetas de grupo** (IAn & 6Soto, 2B, Attention is all we need, … WAYSTOMAKEIT). Cada una tiene **unos 20–30 episodios** de `collect`, con carpetas nombradas por `episode_id` (32 hex).
- **Tamaño por episodio:** `data.h5` de unos 25 MB y `rgb.mp4` de unos 150 MB (muestra del grupo 2B), más `meta.json`, `track.npz`, `checksums.json` y `READY.json`.
- **Volumen estimado** (extrapolado de 3 grupos, hay que confirmarlo con el inventario): **unos 500–700 episodios**, es decir **unas 25–40 h** de juego frente a los 33 min actuales. Ocupan **unos 100 GB** en total, de los que solo unos 15 GB son `data.h5`.
- **No todos están completos:** en los 2 episodios revisados del grupo "IAn & 6Soto" **faltan `data.h5` y `rgb.mp4`**; solo están los JSON, el log y `track.npz`. Esos episodios no sirven y `audit` los rechazará.

#### Plan de ingesta del Drive (por fases, de menos a más bytes)
1. **Inventario (unos MB).** Descargar solo los `meta.json`, `READY.json` y `checksums.json` de todos los episodios. Construir una tabla con grupo, `driver_id`, pista, `reverse`, `difficulty`, `num_karts`, `laps`, `input_device` (teclado o mando), duración, resolución, `schema_version` y si están `data.h5` y `rgb.mp4`. Con ella se decide qué bajar.
2. **Estado y acciones (unos 15 GB).** Descargar `data.h5` y `track.npz` de **todos** los episodios válidos. Esto ya permite:
   - etiquetar la recompensa y la ventaja de todas las partidas;
   - preentrenar el **crítico** sobre el estado privilegiado;
   - entrenar un **maestro con estado** por BC ponderado sin video (vía alternativa de §4), algo barato y muy informativo.
3. **Video selectivo (unos 150 MB por episodio).** Bajar `rgb.mp4` solo de los episodios que entren al BC visual. Prioridad:
   - pistas de entrenamiento poco cubiertas, `reverse` y dificultades variadas;
   - mando (steer continuo, más fácil de imitar que el teclado);
   - grupos distintos, para tener variedad de conductores.

   Primero unos 10 GB y luego más si el BC mejora.
4. **Validación:** `stk-agent audit <carpeta> --out audit.json`, deduplicar por `episode_id` y comprobar 640×360, 60/30 Hz y `schema_version` compatible.
5. **Split por grupo:** varios grupos completos van solo a validación y se excluyen los episodios en pistas reservadas para test.

**Dónde (revisado en esta máquina):**
- **Hardware:** SSD NVMe de 954 GB con **602 GB libres**, i7-13650HX (14 núcleos y 20 hilos), **15,7 GB de RAM** y RTX 5060 Laptop de 8 GB.
- **Espacio necesario:** Drive completo, unos 100 GB, más la caché de frames a 72×128 y 15 Hz, unos 1,5 KB por segundo por frame (unos 45 GB para 30 h; mucho menos con video selectivo). Cabe sobrado.
- **Destino:** `data/drive/<grupo>/<episode_id>/` dentro del proyecto (el Escritorio no está sincronizado con OneDrive, así que no hay conflicto).
- **El límite real es la RAM, no el disco:**
  - la caché offline se lee con memmap desde el SSD, nunca cargada entera en memoria;
  - PPO con **4 simuladores de inicio, 6 como máximo** (cada proceso de STK ocupa cientos de MB); se mide en la Etapa 0.
- **Ejecución:**
  - **Local por bloques** (§10). El cuello de botella es el simulador, que necesita OpenGL con ventana oculta y ya funciona aquí.
  - **Kaggle solo para la Etapa 2** (§9), subiendo los shards ya procesados (unos 45 GB), no el Drive completo.

**Cómo descargar:** la carpeta es pública, pero "Descargar todo" de unos 100 GB en zips no es práctico. Hay dos opciones razonables:
- Añadir un acceso directo a "Mi unidad" y usar **Google Drive para escritorio** con sincronización selectiva por tipo de archivo o carpeta.
- Usar `rclone` con tu cuenta de Google, que permite filtrar por nombre (`--include "*/meta.json"` y luego `--include "*/data.h5"`).

Antes de bajar, revisa el espacio en disco.

**Orden recomendado:** ingerir primero A y repetir la Etapa 2 midiendo la tasa de vueltas del BC con 33 min frente a todo el conjunto. B1 solo si queda video sin etiquetar en cantidad (decenas de horas). B2 solo si B1 no basta y hay al menos varias horas etiquetadas para el IDM.

---

## 1. Arquitectura final (la propuesta con las 5 correcciones del informe)

```
              ENTRENAMIENTO                                         DESPLIEGUE
┌─────────────────────────────────────────┐            ┌───────────────────────────────┐
│ Crítico V(h, s)                         │            │ Actor ← rgb 640×360 (HUD)      │
│  entrada: stopgrad(feat_actor) ⊕        │            │  → resize 72×128, stack 4     │
│           acción previa ⊕ priv_norm(98) │            │  ⊕ acción previa (propia)     │
│  MLP 512-512 → V (normalizado, PopArt)  │            └───────────────┬───────────────┘
└───────────────────┬─────────────────────┘                            │
                    │ Â (GAE, λ=0.95, γ=0.99)                          │
┌───────────────────▼─────────────────────────────────────────────────▼┐
│ Actor: IMPALA-CNN [16,32,32]→256  ⊕ acción previa(5) → MLP 256-256    │
│   ├─ Gaussiana 2D: steer, lon ∈[-1,1] (gas>0 / freno<0)  μ, log σ global │
│   ├─ Bernoulli ×3: drift, fire, nitro   (máscara −∞ en entrenamiento)   │
│   └─ Cabeza auxiliar (solo entrenamiento): predice 10 rasgos privilegiados│
└───────────────────────────────────────────────────────────────────────┘
 Rescue: REGLA (sin progreso ≥1,5 m en 2,5 s) → no es una salida de la red
```

| Componente | Decisión | Por qué |
|---|---|---|
| Entrada del crítico | `concat(stopgrad(feat_actor), acción_prev, priv_norm)` | V(h,s) es insesgado (Baisero & Amato). Con stop-grad el valor no deforma la visión del actor. |
| Encoder | Solo uno, el del actor. El crítico reutiliza sus features **sin gradiente**. | Así no se comparte gradiente y no se paga una segunda CNN. |
| Acción continua | Gaussiana con log σ independiente del estado, inicializada en log 0,5 y limitada a [−2,5; 0,5]. Se guarda la muestra **sin recortar** y se recorta solo al enviarla al motor. | Siguiendo los "37 detalles". Si el steer se satura en ±1, se prueba Beta. |
| Gas/freno | Un único eje `lon`: `acceleration = max(lon,0)`, `brake = lon < −0,15`. | Siguiendo a GT Sophy y a Fuchs. Evita gas y freno simultáneos. |
| Botones | Bernoulli independientes. `fire` se enmascara si no hay ítem o si el escenario no tiene ítems; `nitro`, si `energy = 0`; `drift` siempre está libre. | La máscara se guarda en el buffer. En despliegue no hay máscara, lo cual es inocuo porque disparar sin ítem no hace nada. |
| log-prob conjunto | Suma de los log-prob de las cabezas, con un solo ratio y un solo clip. | Siguiendo H-PPO y los 37 detalles. |
| Entropía | Coeficiente por cabeza: continua 1e-3, discreta 0,01. | Las escalas son distintas. |
| Pérdida auxiliar | MSE del actor sobre 10 rasgos que se pueden inferir del píxel: v_fwd, v_lat, lateral, rumbo (sen y cos), curva a 20 y 45 m, en pista, energía y tiene ítem. | Acelera el encoder. Se descarta en despliegue. |

### Vector privilegiado (98 dims, normalizado con media y desviación móviles)
- **Ego (10, también son los objetivos auxiliares):** velocidad longitudinal y lateral en el marco del kart, desviación lateral sobre ancho/2, rumbo relativo a la pista, curva a 20/45 m, `is_on_road`, energía, tiene ítem.
- **Carrera (9):** rango/N, N, fracción de vuelta, vueltas completadas, yaw rate, salto, escudo, número de ítems, tiempo de attachment.
- **Rayos geométricos (21):** punto central a 5, 10, 20, 30, 45, 60 y 80 m por delante en coordenadas del kart (lateral, frontal, ancho).
- **Rivales (4×5):** los 4 más cercanos en distancia de pista: gap, posición local, velocidad relativa y un flag de presente.
- **Ítems en pista (4×4):** las cajas y objetos más próximos por delante.
- **One-hot:** tipo de powerup (12) y attachment (10).

---

## 2. Recompensa (por decisión a 15 Hz)

```
r_prog   = 0.1 · clip(Δoverall_distance, ±8 m) · 1[en pista ∧ ¬rescate]
r_wall   = −2.0 · (v/30)²     si v cae >4 m/s en una decisión sin frenar   (proxy de choque)
r_off    = −0.5 · (v/30)      fuera de pista
r_rescue = −4.0               cuando la regla dispara rescue
r_pass   = 0.05 · Σ_i (gap_i(t−1) − gap_i(t)),  rivales con gap ∈ (−20 m, +40 m)   (suma cero)
r_time   = −0.01
r_smooth = −0.02 · |Δsteer|
r_end    = +10·(N−rank)/(N−1) al terminar;  −5 si se agota el tiempo
```
- La **misma función** etiqueta los datos humanos y entrena a PPO.
- Se registra la suma de cada término por episodio. Si `r_wall` o `r_off` dominan, el agente está aprendiendo a quedarse quieto: se baja el coeficiente.
- Ítems: no hay recompensa por disparar; su efecto llega a través de `r_pass` y `r_end`.

---

## 3. Escenarios y currículo (cuotas fijas, no etapas que se reemplazan)

Cada entorno paralelo pertenece a una clase con cuota fija y re-sortea pista, sentido, dificultad y slot de salida en cada episodio.

| Clase | Karts | Dificultad | Ítems | Vueltas | Cuota en 4A | 4B | 4C | 5 |
|---|---|---|---|---|---|---|---|---|
| `time_trial` | 1 | – | no | 1 | 100 % | 30 % | 20 % | 15 % |
| `traffic` | 4/6/8 | 0–2 | no | 1 | – | 70 % | 30 % | 25 % |
| `full_race` | 8 | 1–2 | sí | 1–3 | – | – | 50 % | 40 % |
| `league` (vs checkpoints) | 8 | – | sí | 3 | – | – | – | 20 % |

- **Pistas de entrenamiento (15):** lighthouse, snowmountain, minigolf, xr591, snowtuxpeak, black_forest, fortmagma, olivermath, cocoa_temple, zengarden, hacienda, scotland, volcano_island, candela_city, abyss. Todas están verificadas como disponibles en el motor.
- **Pistas reservadas para test (≈4):** ravenbridge_mansion, sandtrack, stk_enterprise, mines. Hay que verificar con `stk-agent assets` que existen. Ojo: ravenbridge_mansion **tiene datos humanos**, así que conviene excluir ese episodio del BC o cambiar de pista reservada.
- **Aleatorización:** sentido (50 %), kart, slot de salida (incluido el último), dificultad, N.
- **Mejora posterior:** Prioritized Level Replay sobre tuplas (pista, sentido, N, dificultad), muestreando según |GAE| medio.

---

## 4. Pipeline por etapas

| Etapa | Qué se hace | Criterio de salida |
|---|---|---|
| **0. Infraestructura** | `Engine.send/recv` y `VecRace` en lockstep, render solo en el último tick, avance rápido de la cuenta atrás. Reanudación, `--max-hours` y watchdog (§10). Benchmark con 2/4/6 entornos y 640×360 vs 320×180, midiendo también RAM y temperatura. | Pasos/s, RAM y temperatura medidos; elegir N (≤ 6) y resolución. |
| **0b. Ingesta del Drive** | Por fases y por grupos (§0b): primero el inventario de `meta.json`, luego los `data.h5` y al final el video selectivo; después `audit`. | Inventario con episodios válidos y split por grupo. |
| **1. Etiquetado humano** | Remuestrear los episodios a 15 Hz (video 30 FPS: un frame de cada 2; estado a 60 Hz: uno de cada 4 ticks). Convertir las acciones (steer medio, `lon = acc − brake`, botones = any) y calcular `priv`, máscara, recompensa, retornos y ventanas de fallo (1,5 s antes de cada rescate o choque). Filtrar fases de conducción (`GO/MUSIC/RACE`). Un shard por episodio, reanudable, a prioridad baja. Los 10 locales ocupan ≈ 0,8 GB; con el Drive, unos 45 GB para unas 30 h. | Dataset con su recompensa; histograma de ventajas. |
| **2. Preentrenamiento offline** (local en 2–3 bloques, o Kaggle en 1 corrida) | Crítico V(h,s) por regresión a los retornos MC. Actor con **BC ponderado por ventaja**: `w = clip(exp(β·Â/σ_Â), 0, 20)`, β = 2, con BC uniforme las primeras 3 épocas. Más la pérdida auxiliar. Baselines: BC plano y BC sobre el 10–30 % mejor. | El BC completa una vuelta en ≥ 50 % de las pistas con datos (evaluación en el motor). |
| **3. Calentamiento del crítico** | PPO con el actor congelado (lr del actor = 0) durante unas 20–50 actualizaciones. | Varianza explicada > 0,5 y pérdida de valor estable. |
| **4A. Contrarreloj** | Rampa del lr del actor de 0 a 2,5e-4 en unas 50 actualizaciones, encoder congelado las primeras 30, KL hacia π_BC por cabeza (0,5 continuo y 1,0 discreto, decaimiento 0,995 por actualización). Solo steer, lon y drift (fire y nitro enmascarados). | ≥ 90 % de vueltas completadas en las pistas de entrenamiento. |
| **4B. Tráfico** | Se añade la clase `traffic`. Se activa `r_pass`. | Tasa de victoria creciente frente a la dificultad 2; sin regresión en 4A. |
| **4C. Ítems** | Se desenmascaran fire y nitro en `full_race`, con KL discreto más alto. Se mantiene el 20–30 % de contrarreloj. | Sin regresión en 4A/4B; uso de ítems > 0. |
| **5. Liga** | Rivales que son checkpoints congelados del propio agente: el `Participant` externo controla un kart con la política vieja, mediante `MultiAgentEnv` con 2 participantes. Se aplica PLR. | El mejor checkpoint según el n-athlon (ver §6). |
| **Alt. (si los pasos/s son bajos)** | Maestro con estado en lugar de píxeles: un actor sobre features que se pueden inferir del frame, entrenado sin renderizado (posible si `GraphicsConfig.none()` funciona con el binding). Se destila con DAgger al actor de píxeles y se termina con 4A–5. | El alumno alcanza ≥ 90 % del maestro. |

---

## 5. Hiperparámetros iniciales de PPO

| Parámetro | Valor |
|---|---|
| Entornos × pasos | 4 × 512 (batch de 2048), 4 minibatches, 4 épocas; con 6 entornos, 6 × 384 |
| γ / λ / clip | 0,99 / 0,95 / 0,15 |
| lr | 2,5e-4 con recocido lineal hasta 10 %; Adam eps 1e-5; clip de gradiente 0,5 |
| Pérdida de valor | coeficiente 0,5; valor normalizado (media y desviación móviles de los retornos) |
| Recompensa | normalizada por la desviación del retorno descontado móvil |
| Truncamiento | bootstrap `r += γ·V(obs_final)` si se trunca; sin bootstrap si termina |
| Corte temprano | `approx_kl > 0,03` → cortar épocas |
| Entropía | continua 1e-3, discreta 0,01 (≈ 0 en las cabezas con KL activo) |
| Auxiliar | 0,1 |
| Presupuesto | 5–20 M decisiones para 4A; bastante más para 4C y 5 |

**Monitorización:** approx_kl, clipfrac, varianza explicada, σ por dimensión, histograma de `lon` (para detectar la política que siempre acelera), suma de cada término de recompensa, rescates por episodio y tasa de victoria por clase de escenario.

---

## 6. Evaluación

- **Matriz:** {pistas de entrenamiento, pistas reservadas} × {dificultad 0, 1, 2} × {3, 7 rivales} × {con ítems, sin ítems} × {salida aleatoria, salida última}.
- **Métricas por celda:** tasa de victoria con IC de Wilson al 95 %, rango medio, tasa de carreras terminadas, rescates por vuelta, fracción fuera de pista y tiempo de vuelta frente al humano. Son 50–100 carreras por celda, con ≥ 3 semillas e IQM.
- **Política:** determinista (μ recortada y p > 0,5).
- **Prueba final con el harness oficial:** `stk_agent evaluation` con un `agent.py` que solo usa `act(rgb)`. Ese agente reproduce el preprocesado y el frame stack, decide cada 2 llamadas (30 → 15 Hz) y aplica un rescue heurístico basado en píxeles: frames casi idénticos durante 3 s mientras acelera. **Lo que reporta este harness es la cifra válida.**
- **n-athlon para elegir checkpoint:** salida desde el último puesto, curva difícil, adelantamiento y defensa con ítem; después, round-robin entre candidatos.

---

## 7. Estructura de archivos propuesta (`ppo_rl/`)

| Archivo | Contenido |
|---|---|
| `config.py` + `configs/*.yaml` | Dataclasses (obs, reward, env, scenarios, model, ppo, offline), overrides por CLI y una config por etapa (4A, 4B, 4C, 5). |
| `features.py` | Vector privilegiado, geometría de pista, máscaras y conversión de acciones política ↔ motor. |
| `reward.py` | La recompensa de §2 con desglose por término. |
| `vec_env.py` | `RaceInstance` y `VecRace` (lockstep, escenarios por cuota, rescue por regla, autoreset). |
| `preprocess.py` | Resize en torch (`area`) y frame stack; compartido por el entrenamiento, el BC y `agent.py`. |
| `model.py` | IMPALA-CNN, actor híbrido, crítico V(h,s), cabeza auxiliar, normalizadores y KL por cabeza. |
| `offline.py` | Construcción de la caché desde `DemonstrationDataset` y preentrenamiento AWR-BC + crítico. |
| `ppo.py` | Rollouts, GAE, actualización, warmup del crítico, rampa del lr, KL decreciente y logging. |
| `evaluate.py` | La matriz de §6 con Wilson e IQM, en JSON, una celda a la vez y reanudable. |
| `agent.py` + `agent.json` | Agente de despliegue para `stk_agent` (solo RGB). |
| `session.py` | Todo lo de §10: checkpoint atómico y reanudación (`last.pt`), `--max-hours` / `--until`, Ctrl+C seguro, watchdog de temperatura con `nvidia-smi`, evitar la suspensión mientras corre, prioridad baja y límite de hilos. Lo usan `offline.py`, `ppo.py` y `evaluate.py`. |
| `ingest.py` | Inventario del Drive (`meta.json` → CSV), selección de episodios, `audit` y split por grupo. |
| `kaggle/pretrain.ipynb` | Envoltorio de `offline.py` para Kaggle: lee los shards de `/kaggle/input`, reanuda desde la salida anterior y guarda en `/kaggle/working` (§9). |
| `__main__.py` | CLI: `bench`, `ingest`, `build-dataset`, `pretrain`, `train`, `evaluate`, todos con `--max-hours`. |

Cambio fuera de `ppo_rl`: añadir `send()`/`recv()` a `src/stk_agent/engine.py`, unas 10 líneas que no rompen la API.

**Dependencias nuevas:** `torch` (build CUDA 12.8 para la RTX 5060) y `tensorboard`.

---

## 8. Orden de trabajo y riesgos

1. **Etapa 0** (benchmark, RAM y temperatura).
2. **Etapa 0b** (ingesta del Drive por fases).
3. **Etapa 1** (caché por grupos).
4. **Etapa 2** (BC, local o Kaggle) **y evaluar en el motor**. Si el BC no da ni una vuelta, revisar la alineación entre frames y acciones antes de seguir.
5. **Etapas 3 y 4A en una sola pista** (lighthouse), para validar la recompensa.
6. **4A con varias pistas.**
7. **4B, 4C y 5.**

El reparto por semanas está en el calendario de §10.

| Riesgo | Mitigación |
|---|---|
| El proxy de choque (caída de velocidad) confunde los saltos o los golpes de ítems con choques de pared. | Registrar `r_wall` por episodio y revisarlo en video. Excluir los ticks con `jumping` o con un attachment activo. |
| 33 min de datos de un solo conductor y una sola dirección de pista. | Ingerir los unos 500–700 episodios del Drive (§0b), por fases: inventario, `data.h5` y video selectivo; BC solo como arranque; KL decreciente; grabar más (reverse, dificultad 0–1). |
| Desfase de dominio entre entrenamiento y despliegue (resolución y ritmo). | Entrenar con render 640×360 igual que la evaluación; el resize se hace en el agente. |
| Generalización a pistas nuevas con unas 20 pistas en total. | Reservar pistas desde el principio y reportar la diferencia entre entrenamiento y reservadas. |
| No se verificó si pystk2 permite snapshots de estado. | Sin "mistake replay" por ahora; usar salida aleatoria. |
| Calentamiento o falta de RAM de la laptop con sesiones largas. | ≤ 4–6 entornos, watchdog de 85 °C, bloques de 1–3 h que se pueden reanudar (§10). |
| Que un bloque se corte a mitad (tapa cerrada, corte de luz). | Checkpoint atómico cada 15 min; reanudar = volver a ejecutar el comando. |

---

## 9. Ejecución en Kaggle (límite: unas 30 h de GPU por semana)

### Límites que condicionan el diseño
Fuentes secundarias; confírmalos en tu cuenta, porque Kaggle los cambia.
- **GPU:** unas 30 h por semana compartidas entre todos los notebooks; la cifra varía y no hay número oficial publicado.
- **Hardware:** P100 de 16 GB o 2×T4; **solo unas 4 vCPU**.
- **Sesión:** máximo 12 h; la sesión interactiva se corta tras unos 20 min de inactividad.
- **Disco:** `/kaggle/working` tiene unos 20 GB persistentes (la salida del notebook); `/kaggle/input` es de solo lectura.
- **Datasets privados:** unos 100–200 GB de cuota total, medida sin comprimir.
- **Las sesiones de CPU no gastan cuota de GPU.**

### Consecuencia: separar lo que necesita GPU de lo que necesita simulador
| Trabajo | Cuello de botella | Dónde | Cuota de GPU |
|---|---|---|---|
| Descargar el Drive, `audit`, inventario | red y disco | **local** (602 GB libres) | 0 |
| Decodificar video → shards 72×128 a 15 Hz + features + recompensa (Etapa 1) | CPU y ffmpeg | **local** (20 hilos); en Kaggle con 4 vCPU sería lento | 0 |
| Subir los shards como **dataset privado** de Kaggle | red | local → Kaggle | 0 |
| Maestro con estado (BC/AWR sobre `data.h5`, MLP) | CPU | Kaggle **CPU** o local | 0 |
| **BC ponderado por ventaja (AWR) + crítico V(h,s) offline** (Etapa 2) | GPU (CNN sobre millones de frames) | **Kaggle GPU** ✅ | unas 3–4 h por corrida |
| Ablaciones del BC (plano, top-30 %, AWR β=1/2/3) | GPU | **Kaggle GPU** | unas 3 h cada una |
| **PPO online** (Etapas 3–5) | **simulador** (CPU + OpenGL) | **local recomendado** | 0 |
| Evaluación en el motor y harness oficial | simulador | local | 0 |

**Por qué PPO no conviene en Kaggle:**
- **Pocos simuladores:** con 4 vCPU solo caben unos 2–3 en paralelo, frente a 4–6 en tu laptop. Eso da del orden de 150–250 decisiones/s, aunque hay que medirlo.
- **GPU ociosa:** la GPU pasa la mayor parte del tiempo esperando al simulador, así que se quema cuota sin usarla.
- **Coste en cuota:** 10 M pasos de la Etapa 4A serían unas 12–18 h, más de media semana.
- **Renderizado sin pantalla:** STK necesita renderizar y en Kaggle no hay pantalla (habría que usar Xvfb/EGL). No está verificado que funcione con PySuperTuxKart2.

### Etapa 0-K (gratis, antes de gastar cuota): prueba de viabilidad en una sesión de **CPU**
1. `pip install PySuperTuxKart2==0.7.4` y este repo. La wheel es `manylinux_2_34`; comprobar `ldd --version` ≥ 2.34.
2. Arrancar una carrera con `stk_agent doctor --engine`, primero tal cual y después dentro de `xvfb-run`.
3. Medir decisiones/s con 1, 2 y 4 simuladores.
4. **Decisión:**
   - si no renderiza, o da < 100 decisiones/s, PPO va **solo en local**;
   - si va bien, se puede usar Kaggle para PPO en **sesiones de CPU** (sin cuota), con la red en CPU, porque es pequeña.

### Presupuesto semanal de GPU propuesto
| Semana | Uso de las unas 30 h | Resultado |
|---|---|---|
| 1 | AWR-BC + crítico con todos los datos (unas 4 h); baselines de BC plano y top-30 % (unas 6 h); AWR con β ∈ {1, 3} (unas 6 h); margen para errores (unas 10 h) | `bc.pt` y `critic.pt`, más una tabla de tasa de vueltas del BC (evaluada **en local**) |
| 2 | Re-entrenar el BC con más video si el inventario lo aconseja; pérdida auxiliar y aumentación DrAC (unas 10–15 h) | El mejor checkpoint inicial para PPO |
| 3+ | Solo si PPO local se queda corto: fine-tuning con lotes grandes a partir de rollouts guardados (off-policy limitado) o destilación DAgger del maestro al alumno de píxeles (unas 10 h) | Opcional |

### Reglas para no perder horas
- **Correr en modo commit** ("Save Version → Save & Run All"): corre en segundo plano y evita el corte por inactividad.
- **Checkpoint cada unos 30 min** en `/kaggle/working/ckpt/`. Al empezar, el script busca el último checkpoint en `/kaggle/input/<salida-anterior>/` y **reanuda** (modelo, optimizador, normalizadores, época y RNG). Así una corrida larga se encadena en sesiones de menos de 12 h.
- **AMP (fp16)** en T4 y `torch.backends.cudnn.benchmark=True`.
- **Carga de datos:**
  - Shards de unos 1–2 GB leídos con memmap y prefetch en 3–4 workers; no se carga todo en RAM.
  - Las 4 vCPU son el límite, así que los frames llegan ya redimensionados a 72×128 desde la Etapa 1 local.
  - Si la carga se queda corta, se baja a 64×112.
- **Volumen subido:** unos 1,5 KB por frame comprimido. 30 h de juego a 15 Hz son 1,6 M frames, unos 45 GB sin comprimir, que entran en la cuota de datasets privados. Si no entran, se suben solo las pistas de entrenamiento y los episodios seleccionados.
- **Probar todo en local antes:** smoke test de 1 época con 2 shards, en local o en una sesión de CPU, para no gastar GPU en errores.
- **Registrar** las horas de GPU por corrida en un CSV, para planificar la semana siguiente.

---

## 10. Modo principal: todo en tu laptop, por bloques (sin dejarla 24/7)

La idea es que **cada etapa sea reanudable** y se ejecute en sesiones cortas, de 1 a 3 h, con límites de recursos para que la laptop siga siendo usable y no se recaliente. Kaggle (§9) queda como opción para descargar la parte de la imitación, que es la que más GPU consume.

### Cómo se trocea cada etapa
| Etapa | Unidad de trabajo reanudable | Qué se guarda al parar |
|---|---|---|
| Descarga del Drive | Un **grupo** por sesión (unas 20–30 carpetas) | Lista de episodios ya descargados y auditados |
| Etapa 1 (decodificar video → shards) | Un **episodio** a la vez; se salta lo ya hecho | Shard por episodio + `index.json` |
| Etapa 2 (BC/AWR + crítico) | **Épocas**, o N minibatches | Modelo, optimizador, scaler AMP, época, RNG y orden de los datos |
| Etapas 3–5 (PPO) | **Updates** de PPO; el buffer on-policy se descarta al parar y no se pierde nada importante | Modelo, optimizador, normalizadores de observación, recompensa y valor, coeficiente KL actual, π_BC, estado de los escenarios, contador de pasos y RNG |
| Evaluación | Una **celda** de la matriz a la vez | JSON por celda |

### Reglas de cada comando
- **`--max-hours 2`** (o `--until 23:30`): al llegar al límite termina el update o la época en curso, guarda y **sale solo**.
- **Ctrl+C seguro:** se captura la señal, se guarda el checkpoint y se cierran los simuladores. Nunca quedan procesos de STK huérfanos.
- **Checkpoint cada unos 15 min**, con escritura atómica (`tmp` + `replace`) y los 3 últimos guardados. Si se va la luz o se cierra la tapa, se pierden como mucho 15 min.
- **Reanudar = volver a ejecutar el mismo comando**, que busca `last.pt` en la carpeta del run.
- **Evitar que Windows suspenda** solo mientras el comando corre (`SetThreadExecutionState`). Al salir, la laptop vuelve a su comportamiento normal.

### Límites para no abrumar la laptop
| Recurso | Límite propuesto | Motivo |
|---|---|---|
| Simuladores en paralelo | **4** de entrada; 6 si la RAM lo permite (medir en la Etapa 0) | Con 15,7 GB de RAM hay que dejar unos 5 GB para Windows y para ti |
| Hilos de CPU | `torch.set_num_threads(4)`; dataloader con 2–3 workers | Que queden núcleos libres para el resto del sistema |
| Prioridad | Procesos en **"Por debajo de lo normal"** | El sistema responde aunque se esté entrenando |
| GPU | AMP (fp16) y batch moderado; VRAM objetivo < 6 GB de 8 | Margen para el escritorio |
| Render | 640×360 (si va pesado, 320×180, aceptando un pequeño desfase con la evaluación) | Es el mayor coste por simulador |
| Temperatura | Un *watchdog* lee `nvidia-smi` cada minuto. Si la GPU pasa de **85 °C**, pausa hasta bajar de 75 °C; si se repite, reduce los entornos | Protege la laptop en sesiones largas |
| Energía | Siempre **enchufada**, modo de energía "Equilibrado" o "Rendimiento", base elevada o ventilada | Con batería el rendimiento cae y se gastan ciclos de la batería |

### Cuánto tarda (estimaciones; se corrigen con la Etapa 0)
- **PPO:** con 4 simuladores, unas 250 decisiones/s (estimado a partir de los 133/s medidos con 2), es decir **≈ 1 M pasos por hora**.
  - Etapa 4A en una pista (2–5 M): 1–3 sesiones de 2 h.
  - Etapa 4A en varias pistas (10–20 M): unas 5–10 sesiones.
  - Etapas 4B y 4C: unas 10–20 sesiones más.
- **BC/AWR en la RTX 5060:** del orden de 10–20 min por época con unos 1,6 M frames, es decir 2–3 sesiones de 2 h por corrida completa. Es el candidato natural a mandar a Kaggle para ahorrar horas de laptop.
- **Etapa 1 (decodificar unas 30 h de video):** usa mucha CPU. Se hace **por grupos**, 1–2 grupos por sesión, a prioridad baja, mientras usas la laptop para otra cosa.

### Calendario sugerido (unas 2 h al día, 5 días por semana)
| Semana | Sesiones |
|---|---|
| 1 | Etapa 0 (benchmark de 4 y 6 entornos, 1 sesión). Descarga e inventario por grupos. Etapa 1 con los 10 episodios locales más los primeros grupos |
| 2 | Termina la Etapa 1. **BC/AWR** (local en 2–3 sesiones o Kaggle en 1 corrida). Evaluar el BC en el motor |
| 3 | Etapa 3 (calentamiento del crítico) + Etapa 4A en **una pista** para validar la recompensa |
| 4–5 | Etapa 4A en varias pistas |
| 6+ | Etapas 4B → 4C → 5, evaluando entre bloques |

**Regla de avance:** no se pasa de etapa sin cumplir el criterio de salida de §4. Si una sesión no mejora la métrica principal, la siguiente sesión se dedica a revisar (videos, términos de recompensa), no a entrenar más.
