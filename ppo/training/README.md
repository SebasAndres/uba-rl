# `training/` — Dónde se entrena cada modelo

Orden (cada paso usa el anterior congelado):

```
fase1_habilidades/   PPO del agente de habilidades       → checkpoints/skills/skill_policy.pt
        │ (congelado)
fase2_sft/           dataset del oráculo + SFT con LoRA   → checkpoints/llm/sft/
        │ (inicialización)
fase2_ppo_llm/       PPO multi-turno del LLM              → checkpoints/llm/ppo/

baseline_ppo_plano/  PPO plano 1M pasos (sólo para comparar)
```

**Nadie queda bloqueado:** el SFT y el PPO del LLM se pueden desarrollar desde el
día 1 con `--skills scripted`. Cuando A tenga el agente, se cambia el flag a `ppo`
(y conviene reentrenar/afinar el LLM, porque el agente real falla distinto que el scripteado).

| Carpeta | Script | Estado | Responsable |
|---|---|---|---|
| `fase1_habilidades` | `train_skills.py` | esqueleto: setup listo, loop PPO con TODOs | A |
| `fase2_sft` | `generar_dataset.py` | **funciona** | B |
| `fase2_sft` | `train_sft.py` | esqueleto (receta TRL en el docstring) | C |
| `fase2_ppo_llm` | `train_llm_ppo.py` | esqueleto: entorno listo, PPO con TODOs | C |
| `baseline_ppo_plano` | `train_flat_ppo.py` | esqueleto: wrapper listo, loop con TODO | D |

Logs: TensorBoard en `runs/`. Checkpoints: ver `checkpoints/README.md` (y
copiarlos a Drive si se entrena en Colab).
