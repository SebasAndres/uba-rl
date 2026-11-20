"""Wrapper mínimo sobre crafter.Env.

Crafter usa la API vieja de gym (obs, reward, done, info) y `reset()` no
devuelve info. Este wrapper:
  - devuelve info también en reset (para que el traductor tenga estado inicial),
  - separa terminated (muerte) de truncated (límite de pasos),
  - expone el estado interno que necesitan las habilidades scripteadas y el
    curriculum de la fase 1 (facing, sleeping, edición de inventario).
"""

from typing import Any

import crafter
import numpy as np

NOOP_ACTION = 0


class CrafterEnv:
    """Crafter con API estilo gymnasium y acceso al estado interno del jugador.

    Parameters
    ----------
    seed : int or None, optional
        Semilla del generador de mundos.
    length : int, optional
        Pasos máximos de un episodio de Crafter (después se trunca).
    area : tuple of int, optional
        Tamaño del mundo en celdas.
    size : tuple of int, optional
        Resolución de la observación RGB que ve el agente.
    """

    def __init__(self, seed: int | None = None, length: int = 10000,
                 area: tuple[int, int] = (64, 64), size: tuple[int, int] = (64, 64)) -> None:
        self.env = crafter.Env(area=area, size=size, length=length, seed=seed, reward=True)
        self.length = length
        self.last_info: dict | None = None

    # ------------------------------------------------------------ API gymnasium
    def reset(self) -> tuple[np.ndarray, dict]:
        """Genera un mundo nuevo.

        Crafter no devuelve `info` en reset, así que se da un paso noop para obtenerlo.

        Returns
        -------
        obs : np.ndarray
            Imagen RGB (64, 64, 3) uint8.
        info : dict
            Estado inicial (inventario, logros, mapa semántico, posición, ...).
        """
        self.env.reset()
        obs, _, _, info = self.env.step(NOOP_ACTION)
        self.last_info = info
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Ejecuta una acción primitiva.

        Parameters
        ----------
        action : int
            Índice de la acción (ver `constants.ACTIONS`).

        Returns
        -------
        obs : np.ndarray
            Imagen RGB después de la acción.
        reward : float
            Recompensa nativa de Crafter (+1 por logro nuevo, ±0.1 por vida).
        terminated : bool
            True si el jugador murió.
        truncated : bool
            True si se alcanzó `length` sin morir.
        info : dict
            Estado del juego después de la acción.
        """
        obs, reward, done, info = self.env.step(int(action))
        self.last_info = info
        terminated = info["discount"] == 0.0
        truncated = done and not terminated
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------- estado interno
    @property
    def player(self) -> Any:
        """crafter.objects.Player: objeto interno del jugador."""
        return self.env._player

    @property
    def facing(self) -> tuple[int, int]:
        """tuple of int: dirección (dx, dy) hacia la que mira el jugador."""
        return tuple(int(v) for v in self.player.facing)

    @property
    def sleeping(self) -> bool:
        """bool: si el jugador está dormido."""
        return bool(self.player.sleeping)

    @property
    def num_steps(self) -> int:
        """int: pasos primitivos dados en el episodio actual."""
        return int(self.env._step or 0)

    def set_inventory(self, **items: int) -> None:
        """Edita el inventario (curriculum de la fase 1: practicar habilidades avanzadas).

        Parameters
        ----------
        **items : int
            Cantidad por ítem, ej. ``set_inventory(wood=3, stone=1)``.
        """
        for item, amount in items.items():
            self.player.inventory[item] = int(amount)

    def set_material(self, pos: tuple[int, int], name: str) -> None:
        """Cambia el material de una celda (curriculum: poner una mesa/horno cerca).

        Parameters
        ----------
        pos : tuple of int
            Celda (x, y).
        name : str
            Material interno de Crafter (ej. "table").
        """
        self.env._world[(int(pos[0]), int(pos[1]))] = name

    def refresh(self) -> tuple[np.ndarray, dict]:
        """Recalcula obs, inventario y mapa semántico tras editar el estado sin dar un paso.

        Returns
        -------
        obs : np.ndarray
            Imagen RGB actual.
        info : dict
            `last_info` con "inventory" y "semantic" actualizados.
        """
        info = dict(self.last_info or {})
        info["inventory"] = self.player.inventory.copy()
        info["semantic"] = self.env._sem_view()
        self.last_info = info
        return self.env._obs(), info

    def render(self, size: int = 256) -> np.ndarray:
        """Frame RGB grande para video/ventana (la obs del agente es 64x64).

        Parameters
        ----------
        size : int, optional
            Lado del frame en píxeles.

        Returns
        -------
        np.ndarray
            Imagen (size, size, 3) uint8.
        """
        return self.env.render((size, size))
