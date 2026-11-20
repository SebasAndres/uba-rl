# `checkpoints/` — Dónde dejar los modelos entrenados

`scripts/play.py` y `evaluation/evaluate.py` en modo `auto` buscan exactamente estas rutas:

```
checkpoints/
├── skills/
│   └── skill_policy.pt      # fase 1: save_skill_policy(agent, ...)  (crafter_hrl/skills/policy.py)
└── llm/
    ├── sft/                 # fase 2a: adapter LoRA + tokenizer (trainer.save_model)
    └── ppo/                 # fase 2b: adapter LoRA (+ value head, no se usa para jugar)
```

Prioridad del planner en `auto`: `llm/ppo` → `llm/sft` → oráculo. Habilidades:
`skills/skill_policy.pt` → scripteadas.

- No se commitean (están en `.gitignore`). Compartirlos por Drive.
- En Colab: montar Drive y pasar las rutas con `--skill-ckpt` / `--adapter`, o
  hacer un symlink de `checkpoints/` a una carpeta de Drive.
- Si cambia `GOAL_DIM` (catálogo de habilidades), `load_skill_policy` falla a
  propósito: hay que reentrenar la fase 1.
