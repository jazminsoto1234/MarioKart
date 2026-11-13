from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import time
import numpy as np
from .types import Action, check_rgb

DEFAULT_KEYBOARD = {
    "steer_left": ["left", "a"],
    "steer_right": ["right", "d"],
    "acceleration": ["up", "w"],
    "brake": ["down", "s"],
    "fire": ["space"],
    "rescue": ["r"],
    "drift": ["left shift", "right shift"],
    "nitro": ["left ctrl", "right ctrl"],
    "steer_rate_per_second": 6.0,
}

DEFAULT_GAMEPAD = {
    "steer_axis": 0,
    "throttle_axis": None,
    "throttle_axis_min": -1.0,
    "throttle_axis_max": 1.0,
    "acceleration_button": 0,
    "brake_button": 1,
    "fire_button": 2,
    "rescue_button": 3,
    "drift_button": 4,
    "nitro_button": 5,
    "deadzone": 0.12,
}

DEFAULT_KEYMAP = {"keyboard": DEFAULT_KEYBOARD, "gamepad": DEFAULT_GAMEPAD}


def _merge(base, override):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _validate_keymap(data):
    if not isinstance(data, dict):
        raise ValueError('El keymap debe ser un objeto')
    unknown = set(data) - {'keyboard', 'gamepad'}
    if unknown:
        raise ValueError(f'Secciones desconocidas en keymap: {sorted(unknown)}')
    result = _merge(DEFAULT_KEYMAP, data)
    kb = result['keyboard']
    gp = result['gamepad']
    for name in ('steer_left','steer_right','acceleration','brake','fire','rescue','drift','nitro'):
        keys = kb.get(name)
        if isinstance(keys, str):
            keys = [keys]
            kb[name] = keys
        if not isinstance(keys, list) or not keys or any(not isinstance(x, str) or not x.strip() for x in keys):
            raise ValueError(f'keyboard.{name} debe ser una lista no vacía de nombres de teclas')
    rate = kb.get('steer_rate_per_second')
    if not isinstance(rate, (int,float)) or isinstance(rate, bool) or not np.isfinite(rate) or rate <= 0:
        raise ValueError('keyboard.steer_rate_per_second debe ser positivo')
    if not 0 <= gp.get('deadzone', -1) < 1:
        raise ValueError('gamepad.deadzone debe estar en [0,1)')
    for name in ('steer_axis','acceleration_button','brake_button','fire_button','rescue_button','drift_button','nitro_button'):
        value = gp.get(name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f'gamepad.{name} debe ser un entero >=0 o null')
    if gp.get('throttle_axis') is not None and (type(gp['throttle_axis']) is not int or gp['throttle_axis'] < 0):
        raise ValueError('gamepad.throttle_axis debe ser un entero >=0 o null')
    low, high = gp.get('throttle_axis_min'), gp.get('throttle_axis_max')
    if not all(isinstance(v, (int,float)) and not isinstance(v, bool) and np.isfinite(v) for v in (low, high)) or high <= low:
        raise ValueError('gamepad.throttle_axis_min/max inválidos')
    return result


def load_keymap(path=None, *, data=None):
    """Carga JSON o YAML. Si no se proporciona archivo, usa el mapeo documentado por defecto."""
    if path is not None and data is not None:
        raise ValueError('Usa path o data, no ambos')
    if path is None:
        return _validate_keymap(data or {})
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig')
    suffix = path.suffix.lower()
    if suffix == '.json':
        raw = json.loads(text)
    elif suffix in ('.yaml', '.yml'):
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError('Para keymap YAML instala el extra [game] (incluye PyYAML)') from e
        raw = yaml.safe_load(text)
    else:
        raise ValueError('El keymap debe usar extensión .json, .yaml o .yml')
    return _validate_keymap(raw or {})


class HumanUI:
    def __init__(self, width, height, scale=1.5, joystick=None, keymap=None, mapping=None):
        try:
            import pygame
        except ImportError as e:
            raise RuntimeError('Instala el extra [game] para usar teclado/gamepad') from e
        if keymap is not None and mapping is not None:
            raise ValueError('Usa keymap o mapping, no ambos')
        # mapping accepts a gamepad-only mapping.
        if mapping is not None:
            keymap = {'gamepad': mapping}
        self.keymap = load_keymap(data=keymap or {})
        self.keyboard = self.keymap['keyboard']
        self.mapping = self.keymap['gamepad']
        self.pg = pygame
        pygame.init()
        pygame.joystick.init()
        self.width, self.height = width, height
        self.surface = pygame.display.set_mode((int(width*scale), int(height*scale)))
        self.quit = False
        self.focused = True
        self.steer = 0.0
        self.last_poll = time.perf_counter()
        self.latched = set()
        self.joy = None
        if joystick is not None:
            if joystick < 0 or joystick >= pygame.joystick.get_count():
                raise ValueError('Gamepad no encontrado. Ejecuta gamepads')
            self.joy = pygame.joystick.Joystick(joystick)
            self.joy.init()
        self.key_codes = {}
        for name in ('steer_left','steer_right','acceleration','brake','fire','rescue','drift','nitro'):
            try:
                self.key_codes[name] = [pygame.key.key_code(x) for x in self.keyboard[name]]
            except ValueError as e:
                raise ValueError(f'Tecla inválida en keyboard.{name}: {self.keyboard[name]}') from e
        self.latch_key_to_action = {}
        for action in ('fire','rescue','drift','nitro'):
            for code in self.key_codes[action]:
                self.latch_key_to_action[code] = action

    @property
    def keyboard_steer_rate(self):
        return float(self.keyboard['steer_rate_per_second'])

    def show(self, rgb, caption='SuperTuxKart'):
        a = check_rgb(rgb, self.width, self.height)
        pg = self.pg
        pg.display.set_caption(caption)
        img = pg.surfarray.make_surface(a.transpose(1, 0, 2))
        if img.get_size() != self.surface.get_size():
            img = pg.transform.scale(img, self.surface.get_size())
        self.surface.blit(img, (0, 0))
        pg.display.flip()
        return time.perf_counter_ns()

    def events(self):
        pg = self.pg
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                self.quit = True
            elif e.type == pg.WINDOWFOCUSLOST:
                self.focused = False
                self.latched.clear()
                self.steer = 0.0
            elif e.type == pg.WINDOWFOCUSGAINED:
                self.focused = True
            elif e.type == pg.KEYDOWN and e.key in self.latch_key_to_action:
                self.latched.add(self.latch_key_to_action[e.key])
            elif e.type == pg.JOYBUTTONDOWN and self.joy is not None:
                if getattr(e, 'instance_id', self.joy.get_instance_id()) == self.joy.get_instance_id():
                    for name in ('fire', 'rescue', 'drift', 'nitro'):
                        if e.button == self.mapping[name+'_button']:
                            self.latched.add(name)

    @staticmethod
    def _pressed(keys, codes):
        return any(bool(keys[c]) for c in codes)

    def action(self):
        self.events()
        pg, mp = self.pg, self.mapping
        now = time.perf_counter()
        dt = min(now-self.last_poll, 0.1)
        self.last_poll = now
        if not self.focused:
            return Action()
        keys = pg.key.get_pressed()
        target = int(self._pressed(keys, self.key_codes['steer_right'])) - int(self._pressed(keys, self.key_codes['steer_left']))
        rate = self.keyboard_steer_rate
        self.steer += float(np.clip(target-self.steer, -rate*dt, rate*dt))
        accel = float(self._pressed(keys, self.key_codes['acceleration']))
        brake = self._pressed(keys, self.key_codes['brake'])
        buttons = {name: name in self.latched or self._pressed(keys, self.key_codes[name])
                   for name in ('fire', 'rescue', 'drift', 'nitro')}
        if self.joy is not None:
            def button(name):
                index = mp[name+'_button']
                if index is None:
                    return False
                if not 0 <= index < self.joy.get_numbuttons():
                    raise ValueError(f'Botón {name} inválido en keymap')
                return bool(self.joy.get_button(index))
            axis = mp['steer_axis']
            if axis is None or not 0 <= axis < self.joy.get_numaxes():
                raise ValueError('Eje de dirección inválido')
            value = self.joy.get_axis(axis)
            dz = mp['deadzone']
            joy_steer = np.sign(value)*max(abs(value)-dz, 0)/(1-dz)
            if abs(joy_steer) > abs(self.steer) or target == 0:
                self.steer = float(np.clip(joy_steer, -1, 1))
            accel = max(accel, float(button('acceleration')))
            if mp['throttle_axis'] is not None:
                index = mp['throttle_axis']
                if not 0 <= index < self.joy.get_numaxes():
                    raise ValueError('Eje del acelerador inválido')
                low, high = mp['throttle_axis_min'], mp['throttle_axis_max']
                accel = max(accel, float(np.clip((self.joy.get_axis(index)-low)/(high-low), 0, 1)))
            brake |= button('brake')
            for name in buttons:
                buttons[name] |= button(name)
        self.latched.clear()
        return Action(steer=self.steer, acceleration=accel, brake=brake, **buttons)

    def wait_until(self, deadline):
        """Atiende eventos mientras espera; nunca duerme sin procesar el teclado."""
        while time.perf_counter() < deadline and not self.quit:
            self.events()
            time.sleep(min(0.001, max(0, deadline-time.perf_counter())))

    def pause_if_unfocused(self):
        start = time.perf_counter()
        paused = False
        while not self.focused and not self.quit:
            paused = True
            self.pg.display.set_caption('PAUSADO por pérdida de foco | vuelve a esta ventana')
            self.events()
            time.sleep(0.03)
        return time.perf_counter()-start if paused else 0.0

    def close(self):
        self.pg.quit()


def gamepads():
    import pygame
    pygame.init()
    pygame.joystick.init()
    result = []
    for index in range(pygame.joystick.get_count()):
        joy = pygame.joystick.Joystick(index)
        joy.init()
        pygame.event.pump()
        result.append({'index': index, 'name': joy.get_name(), 'guid': joy.get_guid(),
                       'axes': joy.get_numaxes(), 'buttons': joy.get_numbuttons(),
                       'axis_values': [joy.get_axis(i) for i in range(joy.get_numaxes())]})
    pygame.quit()
    return result
