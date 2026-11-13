# STK Agent

Toolkit for visual agents in SuperTuxKart using PySuperTuxKart2.

It provides game interaction, human-data collection, dataset access, Gymnasium environments and evaluation. Learning algorithms and pretrained models are not included.

## Interface

Policy observation: RGB with the game HUD.

Actions: `steer`, `acceleration`, `brake`, `nitro`, `drift`, `fire`, `rescue`.

Recorded episodes contain RGB at 30 FPS by default, actions and privileged numerical state at 60 Hz, plus static track geometry.

## Install

Python 3.11 is recommended.

```text
conda create -n stk python=3.11 -y
conda activate stk
python -m pip install --upgrade pip
python -m pip install -e .
```

Verify the engine with one command:

```text
python -m stk_agent doctor --engine --config configs/test_collect.json
```

On Windows, the first engine check can prepare the native runtime automatically if the upstream wheel needs it. On Linux, PySuperTuxKart2 0.7.4 provides x86-64 `manylinux_2_34` wheels.

## Human collection

Controls are configured in `configs/keymap.yaml` or `configs/keymap.json`.

```text
python -m stk_agent collect --config configs/collect.json --keymap configs/keymap.yaml --group group_01 --driver driver_01 --out data/local
```

`seed` and `player_slot` can be set to `"random"`. Each effective value is stored in `meta.json`.

List available tracks and karts:

```text
python -m stk_agent assets --config configs/collect.json
```

Audit one episode or a complete collection:

```text
python -m stk_agent audit data/local --out audit.json
```

## Gymnasium

`STKPixelEnv` supports:

- `render_mode=None`: no window.
- `render_mode="rgb_array"`: `render()` returns the current RGB frame.
- `render_mode="human"`: shows the policy view in a window.

The environment can expose privileged state through `info` during training, but the policy observation remains RGB.

## Episode files

```text
episode_id/
  meta.json
  rgb.mp4
  data.h5
  track.npz
  checksums.json
  READY.json
```
