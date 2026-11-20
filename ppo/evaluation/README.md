# `evaluation/` — Evaluación y comparación

**Responsable: D.**

`evaluate.py` corre N episodios por misión con una configuración (planner +
habilidades) y guarda un JSON en `runs/eval/` con: éxito, retorno, decisiones,
pasos, muertes e inválidas por misión, tasas de logros y Crafter score.

```bash
python evaluation/evaluate.py --planner oracle   --skills scripted --episodes 20
python evaluation/evaluate.py --planner zeroshot --skills scripted --split test
python evaluation/evaluate.py --planner llm --adapter checkpoints/llm/sft --skills ppo
python evaluation/evaluate.py --planner llm --adapter checkpoints/llm/ppo --skills ppo
```

## Métricas

- **Tasa de éxito por misión**, separada en train y test (la métrica principal).
- **Crafter score**: `exp(mean(ln(1 + s_i))) − 1` sobre los 22 logros (s_i en %).
  Ojo: acá se calcula sobre episodios de misión (cortos y dirigidos), no sobre
  episodios libres de 10k pasos como en el paper: sirve para comparar entre
  nuestras configuraciones, no contra la tabla del paper.
- Decisiones y pasos promedio, % de muertes, respuestas inválidas por episodio.

## Qué hay que hacer

- [ ] Tabla de comparación (IDEA.md):

| Config | Planner | Habilidades |
|---|---|---|
| PPO plano 1M | — | — (`training/baseline_ppo_plano`) |
| Oráculo (techo scripteado) | oracle | scripted / ppo |
| LLM zero-shot | zeroshot | ppo |
| LLM SFT | llm + sft | ppo |
| LLM SFT + PPO | llm + ppo | ppo |
| Ablación | mejor LLM | scripted vs ppo |

- [ ] Script que lea los JSON de `runs/eval/` y arme la tabla y gráficos.
- [ ] Intervalos de confianza (≥ 50 episodios por misión para la versión final).
- [ ] Demo: videos con `scripts/play.py --video` de las misiones de test.
