# `checkpoints/` — Dónde dejar los modelos

`scripts/play.py` y `evaluation/evaluate.py` (por defecto `--planner llm --skills ppo`)
buscan exactamente estas rutas:

```
checkpoints/
├── skills/
│   └── skill_policy.pt      # fase 1: save_skill_policy(agent, ...)  (crafter_hrl/skills/policy.py)
└── llm/
    ├── dummy/               # LoRA sin entrenar (= modelo base), creado por scripts/crear_dummies.py
    ├── sft/                 # fase 2a: adapter LoRA + tokenizer (trainer.save_model)
    └── ppo/                 # fase 2b: adapter LoRA (+ value head, no se usa para jugar)
```

Antes de entrenar, crear los dummies (requiere `uv sync --extra llm`):

```bash
uv run python scripts/crear_dummies.py           # no pisa checkpoints existentes (usar --force para pisarlos)
```

- Adapter del LLM, si no se pasa `--adapter`: `llm/ppo` → `llm/sft` → `llm/dummy`.
- Agente de habilidades: `skills/skill_policy.pt` (o `--skill-ckpt`).
- **No hay fallback automático** a las versiones scripteadas: si falta un
  checkpoint, el script falla. Para jugar sin modelos usar
  `--planner oracle --skills scripted`.
- No se commitean (están en `.gitignore`). Compartirlos por Drive.
- En Colab: montar Drive y pasar las rutas con `--skill-ckpt` / `--adapter`, o
  hacer un symlink de `checkpoints/` a una carpeta de Drive.
- Si cambia `GOAL_DIM` (catálogo de habilidades), `load_skill_policy` falla a
  propósito: hay que reentrenar la fase 1.
