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


def create_skill_dummy(seed: int, force: bool) -> None:
    """Guarda un agente de habilidades con pesos aleatorios (si no existe uno).

    Parameters
    ----------
    seed : int
        Semilla de la inicialización.
    force : bool
        Pisar el checkpoint aunque ya exista (¡puede ser uno entrenado!).
    """
    import torch
    from crafter_hrl.skills.policy import SkillAgent, save_skill_policy

    if SKILL_CKPT.exists() and not force:
        print(f"[agente] ya existe {SKILL_CKPT} (usar --force para pisarlo; ¿es uno entrenado?)")
        return
    torch.manual_seed(seed)
    save_skill_policy(SkillAgent(), SKILL_CKPT, dummy=True, seed=seed)
    print(f"[agente] dummy guardado en {SKILL_CKPT}")


def create_llm_dummy(base_model: str, force: bool) -> None:
    """Guarda un adapter LoRA sin entrenar (si no existe uno).

    Parameters
    ----------
    base_model : str
        Modelo base de Hugging Face (se descarga la primera vez).
    force : bool
        Pisar el adapter aunque ya exista.
    """
    if LLM_DUMMY.exists() and not force:
        print(f"[llm]    ya existe {LLM_DUMMY}")
        return
    print(f"[llm]    descargando {base_model} (la primera vez, ~1 GB) y creando LoRA...")
    create_lora_adapter(LLM_DUMMY, base_model)
    print(f"[llm]    adapter dummy guardado en {LLM_DUMMY}")


def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos.

    Returns
    -------
    argparse.Namespace
        `base_model`, `seed` y `force`.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="pisar los checkpoints existentes")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Crea los dummies del agente de habilidades y del LLM.

    Parameters
    ----------
    args : argparse.Namespace
        Argumentos de la línea de comandos.
    """
    create_skill_dummy(args.seed, args.force)
    create_llm_dummy(args.base_model, args.force)


if __name__ == "__main__":
    main(parse_args())
