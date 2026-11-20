# LLM planificador + agente PPO en Crafter

RL jerárquico (ver [IDEA.md](IDEA.md) y el diagrama `Pipeline LLM + agente@1x.png`):
un **LLM** recibe una misión en texto y elige **qué** habilidad ejecutar en cada turno;
un **agente PPO congelado** la ejecuta con acciones primitivas (**cómo**).

```
misión ─► LLM ─► "recolectar(madera)" ─► parser ─► goal_vec ─► agente PPO ─► acciones ─► Crafter
           ▲                                                                              │
           └──────────────────── estado → texto (traductor) ◄──────────── obs + info ◄────┘
```

## Estado actual

**Todavía no hay nada entrenado.** Lo que ya está y funciona:

| Pieza | Estado |
|---|---|
| Entorno Crafter + wrapper (`crafter_hrl/env`) | ✅ |
| Catálogo de habilidades, `goal_vec`, chequeo de éxito (`crafter_hrl/skills/catalog.py`) | ✅ |
| Entorno de la fase 1, condicionado a la habilidad (`SkillEnv`) | ✅ |
| Red del agente + formato de checkpoint (`SkillAgent`) | ✅ (sin entrenar) |
| Habilidades **scripteadas** (BFS sobre el mapa semántico) | ✅ |
| Traductor estado→texto, parser, prompt | ✅ |
| Planificador LLM (Qwen + LoRA) | ✅ (zero-shot funciona; falta entrenar) |
| **Oráculo** scripteado (árbol tecnológico) | ✅ |
| Loop jerárquico = entorno de la fase 2 (`PlannerEnv`) | ✅ |
| Juego (`scripts/play.py`) y evaluación (`evaluation/evaluate.py`) | ✅ |
| Generador del dataset de SFT | ✅ |
| Entrenamiento fase 1 (PPO habilidades) | 🟡 esqueleto con TODOs |
| SFT del LLM | 🟡 esqueleto |
| PPO del LLM | 🟡 esqueleto |
| Baseline PPO plano | 🟡 esqueleto |

`play.py` **supone los modelos entrenados** (busca `checkpoints/`). Si no los
encuentra, avisa y usa los equivalentes scripteados (oráculo + habilidades con
BFS), así el loop completo corre desde hoy y nadie queda bloqueado.

## Instalación

```bash
cd ppo
uv sync                  # crea .venv (Python 3.11) con entorno + fase 1
uv sync --extra llm      # + LLM (fase 2)
uv run python scripts/play.py ...   # o: source .venv/bin/activate
```

En Colab (sin venv): `uv pip install --system -r pyproject.toml --extra llm`.

`numba` **no es opcional en la práctica**: sin numba Crafter tarda ~2 s en generar
cada mundo; con numba, ~0,1 s (el primer reset tarda unos segundos por la compilación JIT).

## Jugar

```bash
python scripts/play.py --mission espada_piedra                 # auto: modelos entrenados o fallback scripteado
python scripts/play.py --mission base --video runs/base.mp4    # guarda video
python scripts/play.py --window                                # ventana en vivo (pygame)
python scripts/play.py --planner zeroshot --skills scripted    # Qwen sin entrenar + habilidades scripteadas
python scripts/play.py --planner llm --adapter checkpoints/llm/ppo --skills ppo   # pipeline final
python scripts/random_episode.py                               # inspeccionar obs/info de Crafter
```

Salida por turno: habilidad elegida, si se cumplió, pasos, recompensa, y el
texto que va a ver el LLM en el siguiente turno.

## Evaluar

```bash
python evaluation/evaluate.py --planner oracle --skills scripted --episodes 20
```

Referencia actual (oráculo + habilidades scripteadas, 10 episodios por misión, semillas 1000-1009):

| Misión | Split | Éxito | Decisiones | Pasos |
|---|---|---|---|---|
| pico_madera | train | 90% | 5 | 19 |
| espada_madera | train | 90% | 5 | 19 |
| espada_piedra | train | 90% | 11 | 54 |
| horno | train | 90% | 10 | 42 |
| carbon_5 | train | 50% | 15 | 154 |
| pico_piedra | test | 80% | 11 | 62 |
| pico_hierro | test | 0% | 26 | 174 |
| base | test | 90% | 12 | 50 |
| equipo_piedra | test | 90% | 15 | 86 |

Casi todos los fracasos son muertes (zombis de noche, esqueletos en las cuevas).
`pico_hierro` falla porque el oráculo no sabe buscar hierro (es raro) ni cuidar
la vida: es el logro más difícil de Crafter y un buen lugar para que el LLM le gane al oráculo.

## Estructura

```
ppo/
├── crafter_hrl/                 # paquete: todo lo que comparten juego, entrenamiento y evaluación
│   ├── env/                     # wrapper de Crafter, constantes, lectura del mapa semántico   → README
│   ├── skills/                  # catálogo + goal_vec, SkillEnv (fase 1), red PPO, scripteadas  → README
│   ├── planner/                 # traductor, parser, oráculo, LLM                              → README
│   ├── missions/                # misiones, chequeos de éxito, split train/test                 → README
│   ├── planner_env.py           # loop jerárquico = entorno de la fase 2
│   └── factory.py               # arma planner/agente desde flags ("auto" = checkpoint o fallback)
├── scripts/
│   ├── play.py                  # ▶ jugar una misión con el pipeline completo
│   └── random_episode.py
├── training/                    # dónde se entrena cada modelo                                 → README
│   ├── fase1_habilidades/       # A: PPO del agente de habilidades
│   ├── fase2_sft/               # B: dataset del oráculo · C: SFT
│   ├── fase2_ppo_llm/           # C: PPO multi-turno del LLM
│   └── baseline_ppo_plano/      # PPO plano 1M pasos (comparación)
├── evaluation/                  # D: tasa de éxito por misión, Crafter score, comparación     → README
└── checkpoints/                 # dónde dejar los modelos para que play.py los encuentre       → README
```

## Contratos entre módulos (no romper sin avisar)

1. **Catálogo de habilidades** (`skills/catalog.py`): la lista de habilidades y
   `GOAL_DIM` (hoy 22). Si cambia, hay que reentrenar el agente de la fase 1.
2. **Checkpoint del agente**: `save_skill_policy` / `load_skill_policy` en
   `skills/policy.py`, en `checkpoints/skills/skill_policy.pt`.
3. **Prompt del LLM**: `state_to_text` + `build_messages`. El mismo texto en
   el dataset de SFT, en PPO, en evaluación y en el juego.
4. **Formato de respuesta**: `Razonamiento: ...\nHabilidad: verbo(args)`, lo lee `parse_skill`.
5. **`PlannerEnv`**: un paso = un turno del LLM. Recompensas en `RewardConfig`.

## Equipo (de IDEA.md) → dónde trabaja cada uno

| Rol | Carpeta | Primer entregable |
|---|---|---|
| A: habilidades PPO | `training/fase1_habilidades`, `crafter_hrl/skills` | `recolectar(madera)` entrenada reemplazando a la scripteada |
| B: traductor, oráculo, dataset | `crafter_hrl/planner`, `training/fase2_sft/generar_dataset.py` | dataset de SFT de las misiones de train |
| C: SFT y PPO del LLM | `training/fase2_sft`, `training/fase2_ppo_llm` | SFT que supere al zero-shot |
| D: integración, evaluación, demo | `evaluation`, `scripts`, `training/baseline_ppo_plano` | tabla de comparación y video |

## Cosas que encontramos armando la base (revisar con el equipo)

- **La mesa cuesta más de 1 madera** y cada `recolectar(madera)` da +1. El plan de
  ejemplo de IDEA.md (`recolectar(madera) → colocar(mesa) → ...`) en realidad
  necesita varias `recolectar(madera)`: la espada de piedra son ~11 turnos, no 5.
  El horno cuesta varias piedras. Los costos se leen de `crafter.constants` (no están hardcodeados).
- **`fabricar` exige la mesa a 1 casilla** (3×3 alrededor del jugador). La
  habilidad tiene que incluir caminar hasta la mesa; el oráculo coloca otra mesa
  si la anterior quedó lejos.
- **`dormir` dura hasta que la energía llega al máximo**, no un paso: tiene timeout
  de 300 en vez de 100, y se considera cumplida con el logro `wake_up`.
- **Interrupción por daño**: sin ella, un zombi que aparece en medio de una
  habilidad de 100 pasos mata al jugador sin que el LLM pueda reaccionar.
  `PlannerEnv` corta la habilidad cuando el jugador recibe daño y le devuelve el
  turno al LLM (`interrupt_on_damage=True`).
- **El éxito de una habilidad se mide con los contadores de logros**
  (`info["achievements"]`), no con el inventario: el inventario satura en 9 y se
  consume al fabricar.
- **"Construí una casa" casi no se puede hacer en Crafter**: no podés mirar hacia una
  celda caminable sin moverte a ella, así que no podés taparte la espalda. La
  reemplazamos por `equipo_piedra` (composición de dos misiones de train). Ver
  `crafter_hrl/missions/README.md`.
- **Craftax ≠ Crafter**: si entrenan las habilidades en Craftax (JAX), la política
  no se puede cargar tal cual en Crafter (otro render, otra API). O todo corre en
  Craftax-Classic, o hay que validar la transferencia. Con numba, Crafter
  da ~350 pasos/s por entorno en CPU (medido). Con un entorno por núcleo, 5M de
  pasos deberían llevar del orden de 1-3 h (estimado, no medido).
