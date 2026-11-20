# `crafter_hrl/missions` — Misiones

Cada `Mission` tiene: texto para el LLM, `goals` estructurados (los usa el oráculo),
`split` (`train`/`test`) y un chequeo extra opcional sobre `info`.

| key | split | chequeo |
|---|---|---|
| `pico_madera` | train | `inventory["wood_pickaxe"] >= 1` |
| `espada_madera` | train | `inventory["wood_sword"] >= 1` |
| `espada_piedra` | train | `inventory["stone_sword"] >= 1` |
| `horno` | train | logro `place_furnace` |
| `carbon_5` | train | `inventory["coal"] >= 5` |
| `pico_piedra` | test | `inventory["stone_pickaxe"] >= 1` |
| `pico_hierro` | test | `inventory["iron_pickaxe"] >= 1` |
| `base` | test | mesa y horno colocados y ambos a ≤ 2 casillas del jugador |
| `equipo_piedra` | test | pico de piedra **y** espada de piedra (composición de dos de train) |

## Por qué importa el split

Es el experimento que justifica el LLM: entrenar (SFT y PPO) sólo con `train` y
medir en `test`. Hay tres tipos de generalización en `test`:
- **misma estructura, otro objeto** (`pico_piedra`: en train está `espada_piedra`),
- **más profundo en el árbol** (`pico_hierro`: necesita horno, carbón, hierro),
- **composición** (`base`, `equipo_piedra`).

## Qué hay que hacer

- [ ] Discutir el split con el equipo (¿`pico_piedra` es demasiado fácil para test?).
- [ ] Misiones compuestas nuevas, con chequeo propio y regla en el oráculo.
      Ideas factibles: "granja" (plantar 2 brotes, necesita una habilidad
      `colocar(planta)`), "arsenal" (espada de madera + piedra), "sobreviví a la noche"
      (seguir vivo al paso N con el logro `wake_up`).
- [ ] Ojo con misiones de construcción: en Crafter, para mirar hacia una celda
      caminable hay que moverse a ella, por eso "encerrarse" (la "casa" de IDEA.md)
      casi no se puede hacer. La sacamos.
- [ ] Variantes de redacción del texto de cada misión (parafraseos) para que el
      LLM no memorice la frase exacta.
