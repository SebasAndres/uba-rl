# Fase 2a — Dataset del oráculo + SFT del LLM

**Responsables: B (dataset) y C (SFT).** Salida: `checkpoints/llm/sft/` (adapter LoRA).

Pipeline del LLM (IDEA.md): **zero-shot (baseline) → SFT con planes del oráculo → PPO**.
El SFT le enseña el formato y un plan razonable; el PPO después mejora sobre eso.

## 1. Dataset (`generar_dataset.py`, ya funciona)

```bash
python training/fase2_sft/generar_dataset.py --episodes-per-mission 200   # → data/sft/train.jsonl
```

- Corre oráculo + habilidades scripteadas en las misiones de **train** solamente.
- Un ejemplo por turno, formato conversacional `prompt` / `completion`.
- Por defecto guarda sólo episodios exitosos (`--keep-failed` para todos).
- Semillas de train desde 0; la evaluación usa 1000+ (no solapar).

**B — qué falta:**
- [ ] Razonamientos más ricos que las plantillas actuales del oráculo.
- [ ] Diversidad: parafraseos de la misión, distintos órdenes válidos del plan.
- [ ] Decidir si generar con `--skills ppo` cuando esté el agente (el historial
      va a tener los fallos reales del agente, y el LLM aprende a reintentar).
- [ ] Estadísticas del dataset: # ejemplos por misión y por habilidad.

## 2. SFT (`train_sft.py`, esqueleto)

**C — qué falta:**
- [ ] Implementar con TRL `SFTTrainer` + LoRA (la receta está en el docstring).
      Qwen2.5-0.5B-Instruct entra en una T4 sin cuantizar; 1.5B con `--load-in-4bit`.
- [ ] Loss sólo sobre la respuesta (`completion_only_loss`), no sobre el prompt.
- [ ] Evaluar: `python evaluation/evaluate.py --planner llm --adapter checkpoints/llm/sft`
      en train **y** test. Comparar con zero-shot y con el oráculo.
- [ ] Mirar el % de respuestas inválidas (columna `invalid_per_episode`).
