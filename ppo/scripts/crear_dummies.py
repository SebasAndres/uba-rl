"""Crea los checkpoints "dummy" (sin entrenar) de los dos modelos.

    python scripts/crear_dummies.py

  checkpoints/skills/skill_policy.pt   SkillAgent con pesos aleatorios (init ortogonal de CleanRL)
  checkpoints/llm/dummy/               Qwen2.5 + LoRA recién inicializado (= modelo base)

Con esto `scripts/play.py` corre el loop completo con los modelos reales, en
el mismo formato que van a tener entrenados. Después:
  - training/fase1_habilidades  sobrescribe checkpoints/skills/skill_policy.pt
  - training/fase2_sft          escribe checkpoints/llm/sft   (play.py lo prefiere al dummy)
  - training/fase2_ppo_llm      escribe checkpoints/llm/ppo   (play.py lo prefiere al sft)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crafter_hrl.factory import LLM_DUMMY, SKILL_CKPT  # noqa: E402
from crafter_hrl.planner.llm import DEFAULT_BASE_MODEL, create_lora_adapter  # noqa: E402


def main(args):
    import torch
    from crafter_hrl.skills.policy import SkillAgent, save_skill_policy

    if SKILL_CKPT.exists() and not args.force:
        print(f"[agente] ya existe {SKILL_CKPT} (usar --force para pisarlo; ¿es uno entrenado?)")
    else:
        torch.manual_seed(args.seed)
        save_skill_policy(SkillAgent(), SKILL_CKPT, dummy=True, seed=args.seed)
        print(f"[agente] dummy guardado en {SKILL_CKPT}")

    if LLM_DUMMY.exists() and not args.force:
        print(f"[llm]    ya existe {LLM_DUMMY}")
    else:
        print(f"[llm]    descargando {args.base_model} (la primera vez, ~1 GB) y creando LoRA...")
        create_lora_adapter(LLM_DUMMY, args.base_model)
        print(f"[llm]    adapter dummy guardado en {LLM_DUMMY}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    main(p.parse_args())
