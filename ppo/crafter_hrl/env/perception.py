"""Lectura del mapa semántico: qué hay cerca, en qué dirección y cómo llegar.

Lo usan el traductor (estado -> texto), las habilidades scripteadas y el oráculo.
`semantic` es info["semantic"] (array [x, y] de ids) y `pos` es info["player_pos"].
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .constants import ID_TO_NAME, MAT_ID, OBJ_ID, VIEW_RADIUS_X, VIEW_RADIUS_Y, WALKABLE

Cell = tuple[int, int]
Step = tuple[int, int]

# Lo que nunca vale la pena reportar al LLM.
_IGNORED = {"grass", "sand", "path", "player", None}
_WALKABLE_IDS = np.array([MAT_ID[m] for m in WALKABLE])
_NEIGHBOR_STEPS: tuple[Step, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class Seen:
    """Algo visible en el mapa, relativo al jugador.

    Attributes
    ----------
    name : str
        Nombre interno de Crafter (tree, cow, ...).
    dx, dy : int
        Desplazamiento desde el jugador (en Crafter, dy < 0 es hacia arriba/norte).
    """

    name: str
    dx: int
    dy: int

    @property
    def dist(self) -> int:
        """int: distancia Manhattan al jugador."""
        return abs(self.dx) + abs(self.dy)

    @property
    def direction(self) -> str:
        """str: dirección cardinal en castellano (N, S, E, O, NE, ...) o "aquí"."""
        north_south = "N" if self.dy < 0 else "S" if self.dy > 0 else ""
        east_west = "E" if self.dx > 0 else "O" if self.dx < 0 else ""
        return north_south + east_west or "aquí"


def name_to_ids(name: str) -> list[int]:
    """Convierte un nombre de material u objeto a sus ids del mapa semántico.

    Parameters
    ----------
    name : str
        Nombre interno de Crafter (ej. "tree", "cow").

    Returns
    -------
    list of int
        Ids que representan a `name` en info["semantic"].

    Raises
    ------
    KeyError
        Si `name` no es un material ni un objeto conocido.
    """
    if name in MAT_ID:
        return [MAT_ID[name]]
    if name in OBJ_ID:
        return [OBJ_ID[name]]
    raise KeyError(name)


def visible(semantic: np.ndarray, pos: Sequence[int], rx: int = VIEW_RADIUS_X,
            ry: int = VIEW_RADIUS_Y) -> list[Seen]:
    """Lista todo lo interesante dentro de la ventana visible.

    Parameters
    ----------
    semantic : np.ndarray
        Mapa semántico [x, y] de ids.
    pos : sequence of int
        Posición (x, y) del jugador.
    rx, ry : int, optional
        Radio horizontal y vertical de la ventana visible.

    Returns
    -------
    list of Seen
        Elementos visibles (sin pasto/arena/camino/jugador), ordenados por
        distancia y luego por nombre.
    """
    x0, y0 = int(pos[0]), int(pos[1])
    x_range = range(max(0, x0 - rx), min(semantic.shape[0], x0 + rx + 1))
    y_range = range(max(0, y0 - ry), min(semantic.shape[1], y0 + ry + 1))
    seen = []
    for x in x_range:
        for y in y_range:
            name = ID_TO_NAME.get(int(semantic[x, y]))
            if name not in _IGNORED:
                seen.append(Seen(name, x - x0, y - y0))
    return sorted(seen, key=lambda s: (s.dist, s.name))


def closest_per_type(seen: list[Seen]) -> dict[str, Seen]:
    """Se queda con el elemento más cercano de cada tipo.

    Parameters
    ----------
    seen : list of Seen
        Elementos ordenados por distancia (como los devuelve `visible`).

    Returns
    -------
    dict of str to Seen
        El primero (más cercano) de cada nombre, en orden de distancia.
    """
    best: dict[str, Seen] = {}
    for s in seen:
        best.setdefault(s.name, s)
    return best


def is_near(semantic: np.ndarray, pos: Sequence[int], name: str, radius: int) -> bool:
    """Indica si hay `name` a distancia de Chebyshev <= `radius` del jugador.

    Parameters
    ----------
    semantic : np.ndarray
        Mapa semántico [x, y] de ids.
    pos : sequence of int
        Posición (x, y) del jugador.
    name : str
        Material u objeto buscado (ej. "table").
    radius : int
        Radio de búsqueda (Crafter usa 1 para fabricar).

    Returns
    -------
    bool
        True si hay al menos una celda con `name` dentro del radio.
    """
    x0, y0 = int(pos[0]), int(pos[1])
    window = semantic[max(0, x0 - radius): x0 + radius + 1, max(0, y0 - radius): y0 + radius + 1]
    return bool(np.isin(window, name_to_ids(name)).any())


def _reconstruct_path(prev: dict[Cell, Cell | None], end: Cell) -> list[Step]:
    """Reconstruye los pasos (dx, dy) desde el inicio del BFS hasta `end`.

    Parameters
    ----------
    prev : dict of Cell to Cell or None
        Predecesor de cada celda visitada (el inicio apunta a None).
    end : Cell
        Última celda del camino.

    Returns
    -------
    list of Step
        Pasos a dar, en orden, para ir del inicio a `end`.
    """
    path, node = [], end
    while prev[node] is not None:
        px, py = prev[node]
        path.append((node[0] - px, node[1] - py))
        node = prev[node]
    return path[::-1]


def find_path_to_adjacent(semantic: np.ndarray, pos: Sequence[int], target_ids: Sequence[int],
                          max_radius: int | None = None) -> tuple[list[Step] | None, Cell | None]:
    """BFS sobre celdas caminables hasta quedar AL LADO de una celda objetivo.

    Los objetos (vacas, zombis) bloquean el paso; la lava se evita porque no
    está entre las celdas caminables.

    Parameters
    ----------
    semantic : np.ndarray
        Mapa semántico [x, y] de ids.
    pos : sequence of int
        Posición (x, y) del jugador.
    target_ids : sequence of int
        Ids de las celdas objetivo.
    max_radius : int or None, optional
        Distancia Manhattan máxima a explorar desde el jugador.

    Returns
    -------
    path : list of Step or None
        Pasos (dx, dy) a dar; vacío si ya está al lado. None si no hay objetivo alcanzable.
    target_cell : Cell or None
        Celda objetivo encontrada, o None.
    """
    width, height = semantic.shape
    start = (int(pos[0]), int(pos[1]))
    is_target = np.isin(semantic, target_ids)
    is_walkable = np.isin(semantic, _WALKABLE_IDS)
    prev: dict[Cell, Cell | None] = {start: None}
    queue = deque([start])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in _NEIGHBOR_STEPS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if is_target[nx, ny]:
                return _reconstruct_path(prev, (cx, cy)), (nx, ny)
            too_far = max_radius is not None and abs(nx - start[0]) + abs(ny - start[1]) > max_radius
            if (nx, ny) in prev or not is_walkable[nx, ny] or too_far:
                continue
            prev[(nx, ny)] = (cx, cy)
            queue.append((nx, ny))
    return None, None
