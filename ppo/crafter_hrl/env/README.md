# `crafter_hrl/env` — Entorno

Wrapper fino sobre `crafter.Env` y utilidades para leer el mapa semántico.

## Qué hay

| Archivo | Contenido |
|---|---|
| `crafter_env.py` | `CrafterEnv`: API tipo gymnasium (`reset → obs, info`, `step → obs, r, terminated, truncated, info`), acceso a `facing`, `sleeping`, `set_inventory` (curriculum), `render(size)` para videos. |
| `constants.py` | Acciones, ids del mapa semántico, reglas de juego (`COLLECT`, `PLACE`, `MAKE`) **leídas de `crafter.constants`**, nombres en castellano. |
| `perception.py` | `visible()` (qué hay en la ventana 9×7 con dirección y distancia), `is_near()`, `find_path_to_adjacent()` (BFS). |

## Datos útiles de Crafter

- `obs`: RGB 64×64×3 uint8. La vista es 9×7 casillas del mundo + 2 filas de inventario.
- `info["semantic"]`: mapa **completo** 64×64 indexado `[x, y]`. Ids: 0 vacío, 1-12
  materiales (`water, grass, stone, path, sand, tree, lava, coal, iron, diamond, table, furnace`),
  13-18 objetos (`player, cow, zombie, skeleton, arrow, plant`).
- `info["player_pos"]`: `(x, y)`. `move_up` es `y-1` (Norte).
- `info["inventory"]`: vitales (`health, food, drink, energy`) + ítems, todos saturan en 9.
- `info["achievements"]`: **contadores** de los 22 logros (no booleanos).
- Crafter no devuelve `info` en `reset()`: el wrapper hace un `noop` para obtenerlo.

## Qué falta / ideas

- [ ] Si se usa Craftax para entrenar más rápido: un `CraftaxEnv` con la misma
      interfaz que `CrafterEnv` (y verificar que el agente transfiere).
- [ ] `visible()` reporta lo que hay en la ventana 9×7 aunque esté tapado o de
      noche; decidir si el traductor debería ver menos.
