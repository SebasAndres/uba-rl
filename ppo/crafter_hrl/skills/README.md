# `crafter_hrl/skills` — Habilidades (nivel bajo)

Responsable: **A** (con B para el catálogo).

## Qué hay

| Archivo | Contenido |
|---|---|
| `catalog.py` | **Contrato**: las 19 habilidades, `encode()` → `goal_vec` (22 dims = verbo 7 + objeto 12 + material 3), chequeo de éxito, requisitos de inventario, timeouts. |
| `skill_env.py` | `SkillEnv`: entorno gymnasium de la **fase 1**. Obs `{"image", "goal", "time"}` (`time` = fracción del timeout que queda), 17 acciones, recompensa +1 al cumplir / −0.01 por paso, timeout 100 (300 para dormir). Sortea sólo habilidades factibles y regala inventario con prob. `boost_prob` para practicar las avanzadas. |
| `policy.py` | `SkillAgent` (CNN Nature + [goal_vec + fracción de timeout + acción anterior] → LSTM → actor/crítico; el estado de la LSTM lo lleva quien llama), `save_skill_policy`/`load_skill_policy`, y `PPOSkillPolicy` (el agente congelado que usa el juego). |
| `scripted.py` | `ScriptedSkillPolicy`: BFS sobre el mapa semántico. Fallback mientras no haya agente entrenado y baseline de la ablación. |
| `executor.py` | `run_skill()`: ejecuta una habilidad hasta cumplirla, morir, timeout o `stop_fn`. |

Habilidades: `recolectar(madera|piedra|carbon|hierro|diamante|brote)`, `beber()`,
`comer(vaca)`, `dormir()`, `pelear(zombi)`, `colocar(mesa|horno|piedra)`,
`fabricar(pico|espada, madera|piedra|hierro)`.

Éxito = sube el contador del logro correspondiente (`collect_wood`, `place_table`,
`make_stone_sword`, `wake_up`, `defeat_zombie`, ...).

## Qué hay que hacer (fase 1)

Ver `training/fase1_habilidades/README.md` para el entrenamiento. Acá:

- [ ] Revisar la arquitectura de `SkillAgent` (¿FiLM en vez de concatenar el goal?). Ya tiene LSTM + acción anterior.
      Si cambia, mantener `save/load_skill_policy` compatibles.
- [ ] Decidir el curriculum en `SkillEnv` (pesos por habilidad, `boost`, mundo persistente o no).
- [ ] ¿Penalizar la muerte en `SkillEnv`? Hoy `death_penalty=0` (IDEA.md no lo define).
- [ ] Mejorar las scripteadas si hace falta para la ablación (hoy no huyen de esqueletos ni de la lava al minar).

## Cómo probar

```bash
python -c "from crafter_hrl.skills.skill_env import SkillEnv; e=SkillEnv(seed=0); print(e.reset())"
python scripts/play.py --skills scripted --planner oracle
python scripts/play.py --skills ppo --skill-ckpt checkpoints/skills/skill_policy.pt
```
