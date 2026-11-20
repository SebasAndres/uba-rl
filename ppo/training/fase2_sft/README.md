# Fase 2a — Dataset del oráculo + SFT del LLM

**Responsables: B (dataset) y C (SFT).** Salida: `checkpoints/llm/sft/` (adapter LoRA).

Pipeline del LLM (IDEA.md): **zero-shot (baseline) → SFT con planes del oráculo → PPO**.
El SFT le enseña el formato y un plan razonable; el PPO después mejora sobre eso.

## 1. Dataset (por implementar: `data/sft/train.jsonl`)

**Idea:** jugar con `PlannerEnv` usando el oráculo como profesor y guardar un
ejemplo por turno, con el MISMO prompt que va a ver el LLM:

```json
{"prompt": [{"role": "system", ...}, {"role": "user", "content": <state_to_text>}],
 "completion": [{"role": "assistant", "content": "Razonamiento: ...\nHabilidad: ..."}],
 "mission": "espada_piedra", "seed": 3, "turn": 5}
```

- Prompt con `build_messages(text)`, respuesta con `OraclePlanner.plan(text, info).text`
  (ya trae el razonamiento encadenado: misión → faltante → ... → decisión).
- Sólo misiones de **train**; semillas desde 0 (la evaluación usa 1000+).
- **Perturbaciones (DAgger)**: en ~15% de los turnos ejecutar otra habilidad (al azar
  o cortada) en vez de la del oráculo, pero etiquetar con lo que el oráculo diría
  en el estado que resulta. Así entran historiales con ✗ / interrumpidas y el
  LLM aprende a recuperarse. La fila del turno perturbado no se guarda.
- Guardar también episodios fallidos (la etiqueta sigue siendo correcta), salvo
  los últimos turnos antes de morir.
- Balance: deduplicar prompts idénticos; `recolectar(madera)` es ~40% de las etiquetas.
- Parafraseos de la misión (3-4 por misión) para no memorizar el enunciado.
- Separar algunas semillas como validación offline: % habilidad == oráculo y % inválidas.
- Tamaño: ~9 turnos/episodio → 200 episodios/misión ≈ 9k ejemplos.
- Generar con habilidades scripteadas ahora; regenerar con `--skills ppo` cuando
  esté el agente (falla distinto).

Si cambia `state_to_text`, `SYSTEM_PROMPT` o el oráculo, hay que regenerar el dataset.

## 2. SFT (`train_sft.py`, esqueleto)

**C — qué falta:**
- [ ] Implementar con TRL `SFTTrainer` + LoRA (la receta está en el docstring).
      Qwen2.5-0.5B-Instruct entra en una T4 sin cuantizar; 1.5B con `--load-in-4bit`.
- [ ] Loss sólo sobre la respuesta (`completion_only_loss`), no sobre el prompt.
- [ ] Evaluar: `python evaluation/evaluate.py --planner llm --adapter checkpoints/llm/sft`
      en train **y** test. Comparar con zero-shot y con el oráculo.
- [ ] Mirar el % de respuestas inválidas (columna `invalid_per_episode`).
