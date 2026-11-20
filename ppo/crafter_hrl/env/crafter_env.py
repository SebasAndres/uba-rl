"""Wrapper mínimo sobre crafter.Env.

Crafter usa la API vieja de gym (obs, reward, done, info) y `reset()` no
devuelve info. Este wrapper:
  - devuelve info también en reset (para que el traductor tenga estado inicial),
  - separa terminated (muerte) de truncated (límite de pasos),
  - expone el estado interno que necesitan las habilidades scripteadas y el
    curriculum de la fase 1 (facing, sleeping, edición de inventario).
"""

import numpy as np
import crafter


class CrafterEnv:
    def __init__(self, seed: int | None = None, length: int = 10000, area=(64, 64), size=(64, 64)):
        self.env = crafter.Env(area=area, size=size, length=length, seed=seed, reward=True)
        self.length = length
        self.last_info: dict | None = None

    # -- API estilo gymnasium ------------------------------------------------
    def reset(self):
        obs = self.env.reset()
        # Un noop para obtener el info inicial (Crafter no lo da en reset).
        obs, _, _, info = self.env.step(0)
        self.last_info = info
        return obs, info

    def step(self, action: int):
        obs, reward, done, info = self.env.step(int(action))
        self.last_info = info
        terminated = info["discount"] == 0.0  # murió
        truncated = done and not terminated
        return obs, reward, terminated, truncated, info

    # -- acceso al estado interno -----------------------------------------------
    @property
    def player(self):
        return self.env._player

    @property
    def facing(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.player.facing)

    @property
    def sleeping(self) -> bool:
        return bool(self.player.sleeping)

    @property
    def num_steps(self) -> int:
        return int(self.env._step or 0)

    def set_inventory(self, **items: int) -> None:
        """Edita el inventario (curriculum de la fase 1: practicar habilidades avanzadas)."""
        for k, v in items.items():
            self.player.inventory[k] = int(v)

    def render(self, size: int = 256) -> np.ndarray:
        """Frame RGB grande para video/ventana (la obs del agente es 64x64)."""
        return self.env.render((size, size))
