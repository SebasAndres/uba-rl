# Fase 1 — Agente de habilidades con PPO (sin LLM)

**Responsable: A.** Salida: `checkpoints/skills/skill_policy.pt` (lo carga `play.py --skills ppo`).

## Qué se entrena

Una sola política condicionada al objetivo:
- **Input**: imagen 64×64×3 + `goal_vec` (22 dims, `crafter_hrl/skills/catalog.py`).
- **Output**: 1 de las 17 acciones por paso.
- **Entorno**: `SkillEnv` (`crafter_hrl/skills/skill_env.py`), ya implementado:
  un episodio = un intento de una habilidad; +1 al cumplirla, −0.01 por paso,
  timeout 100 (300 para dormir).
- **Red**: `SkillAgent` (`crafter_hrl/skills/policy.py`), ya implementada.

## Qué hay que hacer

1. [ ] **Completar el loop de PPO** en `train_skills.py` (los 4 TODOs). Es
   `cleanrl/cleanrl/ppo_atari.py` L202-L320 con dos buffers de observación
   (`images`, `goals`) en vez de uno. Prestar atención a: rollout, GAE, ratio, clip, crítico.
2. [ ] **Primer hito (IDEA.md, paso 4)**: entrenar sólo `recolectar(madera)`
   (`SkillEnv(skills=[Skill("recolectar", "madera")])`) y reemplazar la
   scripteada en `play.py`. Validar que la curva de éxito sube antes de seguir.
3. [ ] Agregar habilidades de a poco (curriculum): madera → beber/comer → colocar
   → fabricar → piedra/carbón → hierro/diamante. Loguear **éxito por habilidad**
   (ya está el diccionario `success_by_skill`).
4. [ ] Ajustar `boost_prob` / `DEFAULT_BOOST` para que las habilidades avanzadas
   aparezcan lo suficiente.
5. [ ] Evaluar cada habilidad por separado (tasa de éxito y pasos promedio) contra
   la scripteada: eso es la ablación "scripteadas vs RL".
6. [ ] Congelar: guardar el checkpoint final y no tocar más el catálogo.

## Detalles a decidir

- **Truncation por timeout**: CleanRL la trata como `done`. Como el timeout es
  parte de la tarea (no del entorno), es razonable; bootstrapear `V(s)` es una mejora.
- **¿Una red o una por habilidad?** IDEA.md dice una sola condicionada al objetivo
  (comparte la percepción). Si alguna habilidad no aprende, se puede probar aparte.
- **Habilidades difíciles**: `recolectar(diamante)`, `pelear(zombi)` (aparecen
  sobre todo de noche), `recolectar(brote)` (10% de probabilidad por golpe al pasto).
  Puede que haya que aceptar tasas bajas o dejarlas scripteadas.
- **Velocidad**: ~350 pasos/s por entorno con numba. Usar `AsyncVectorEnv` con un
  entorno por núcleo. En Colab hay 2 núcleos: puede convenir entrenar en una PC.

## Cómo correr

```bash
python training/fase1_habilidades/train_skills.py --num-envs 8 --total-timesteps 2000000
tensorboard --logdir runs/
python scripts/play.py --skills ppo
```
