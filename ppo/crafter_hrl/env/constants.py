"""Constantes de Crafter leídas de la librería (no hardcodear costos).

Todo lo que sale de `crafter.constants` (data.yaml) se lee en runtime: si cambia
la versión de Crafter, cambian solos los costos/requisitos.
"""

import crafter.constants as cc

# --- Acciones primitivas (17) -------------------------------------------------
ACTIONS: list[str] = list(cc.actions)
ACTION_ID: dict[str, int] = {name: i for i, name in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)

# Dirección (dx, dy) de cada movimiento. En Crafter "up" es y-1.
MOVES = {
    "move_left": (-1, 0),
    "move_right": (1, 0),
    "move_up": (0, -1),
    "move_down": (0, 1),
}
DIR_TO_MOVE = {v: k for k, v in MOVES.items()}

# --- Mapa semántico (info["semantic"], indexado [x, y]) -----------------------
# Mismo orden que usa crafter.engine.World / crafter.Env._sem_view.
MATERIALS: list[str] = list(cc.materials)
MAT_ID: dict[str, int] = {name: i + 1 for i, name in enumerate(MATERIALS)}  # 0 = None
OBJECTS: list[str] = ["player", "cow", "zombie", "skeleton", "arrow", "plant"]
OBJ_ID: dict[str, int] = {name: len(MATERIALS) + 1 + i for i, name in enumerate(OBJECTS)}
ID_TO_NAME: dict[int, str] = {**{v: k for k, v in MAT_ID.items()}, **{v: k for k, v in OBJ_ID.items()}}

WALKABLE: list[str] = list(cc.walkable)  # grass, sand, path (lava también, pero mata)

# --- Reglas del juego ---------------------------------------------------------
COLLECT = cc.collect   # material -> {require, receive, leaves}
PLACE = cc.place       # stone/table/furnace/plant -> {uses, where}
MAKE = cc.make         # wood_pickaxe/... -> {uses, nearby, gives}
ITEMS = cc.items       # inventario: max/initial
ACHIEVEMENTS: list[str] = list(cc.achievements)

VITALS = ["health", "food", "drink", "energy"]

# Área visible en la observación RGB: 9 de ancho x 7 de alto (las 2 filas de
# abajo son el inventario). Para el traductor sólo reportamos lo que se ve.
VIEW_RADIUS_X = 4
VIEW_RADIUS_Y = 3

# --- Nombres en castellano (texto para el LLM) ---------------------------------
ES = {
    # vitales
    "health": "vida", "food": "comida", "drink": "agua", "energy": "energía",
    # inventario
    "sapling": "brote", "wood": "madera", "stone": "piedra", "coal": "carbón",
    "iron": "hierro", "diamond": "diamante",
    "wood_pickaxe": "pico de madera", "stone_pickaxe": "pico de piedra",
    "iron_pickaxe": "pico de hierro", "wood_sword": "espada de madera",
    "stone_sword": "espada de piedra", "iron_sword": "espada de hierro",
    # mundo
    "water": "agua", "grass": "pasto", "path": "camino", "sand": "arena",
    "tree": "árbol", "lava": "lava", "table": "mesa", "furnace": "horno",
    "cow": "vaca", "zombie": "zombi", "skeleton": "esqueleto", "arrow": "flecha",
    "plant": "planta", "player": "jugador",
}
