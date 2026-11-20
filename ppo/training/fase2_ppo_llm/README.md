# Fase 2b — PPO multi-turno del LLM (agente congelado)

**Responsable: C.** Entrada: `checkpoints/llm/sft/`. Salida: `checkpoints/llm/ppo/`.

## El entorno ya está: `crafter_hrl.PlannerEnv`

```python
env = PlannerEnv(build_skill_policy("ppo"), missions=missions_by_split("train"))
texto, info = env.reset()
texto, r, terminated, truncated, turn = env.step(respuesta_del_llm)
```

Un paso = un turno. Recompensas (`RewardConfig`): +1 misión cumplida, −0.5 muere
o pide algo inválido, −0.02 por decisión, +0.2 opcional por logro nuevo
(`prerequisite`). La habilidad se corta si el jugador recibe daño (el LLM recupera el turno).

## Qué hay que hacer (`train_llm_ppo.py`)

1. [ ] **Rollout**: generar con muestreo, guardar por turno `prompt_ids`,
   `response_ids`, logprobs por token, `V(s)` y recompensa.
2. [ ] **Value head**: `Linear(hidden, 1)` sobre el último token del prompt (el estado del turno).
3. [ ] **GAE a nivel de turno**: la recompensa va en el último token de cada turno;
   GAE reparte el crédito entre turnos; todos los tokens del turno comparten `A_t`.
4. [ ] **Update PPO**: ratio por token (enmascarando el prompt), clip, value loss,
   penalización KL contra el modelo SFT para que no se rompa el formato.
5. [ ] Loguear éxito por misión, retorno, decisiones, inválidas y KL. Guardar el
   adapter + value head.
6. [ ] Evaluar en **test** (generalización) contra el SFT.

**Alternativa sin crítico: GRPO.** Varias rollouts de la misma misión y semilla,
ventaja = retorno normalizado dentro del grupo. Más simple (sin value head) y se
puede usar `trl` como referencia. Si el PPO multi-turno se complica, es el plan B.

## Costos

- Un episodio son 10-30 turnos (más con el agente real) y 1000-2000 pasos de
  Crafter: el cuello de botella es la generación del LLM, no Crafter.
- Batch de episodios en paralelo: generar todos los prompts del turno juntos (batch del LLM).
- Desarrollar con `--skills scripted` (más rápido y determinista) y pasar a `ppo` al final.
