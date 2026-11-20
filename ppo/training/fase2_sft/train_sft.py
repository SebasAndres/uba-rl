"""FASE 2a — SFT del LLM planificador con LoRA sobre el dataset del oráculo.

ESQUELETO. Plan sugerido (TRL reciente; dataset conversacional prompt/completion):

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("json", data_files=args.data)["train"].remove_columns(
        ["mission", "seed", "turn", "episode_success"])
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                             target_modules="all-linear", task_type="CAUSAL_LM")
    cfg = SFTConfig(output_dir=args.out, num_train_epochs=2, per_device_train_batch_size=8,
                    gradient_accumulation_steps=2, learning_rate=2e-4, bf16=True,  # fp16=True en T4 (Colab)
                    completion_only_loss=True,  # loss sólo sobre la respuesta, no el prompt
                    logging_steps=10, save_strategy="epoch", report_to="tensorboard")
    trainer = SFTTrainer(model=args.base_model, train_dataset=ds, args=cfg, peft_config=peft_config)
    trainer.train()
    trainer.save_model(args.out)          # -> checkpoints/llm/sft (adapter LoRA + tokenizer)

Después:
    python evaluation/evaluate.py --planner llm --adapter checkpoints/llm/sft --split all

IMPORTANTE: el prompt tiene que salir de crafter_hrl.planner.llm.build_messages y
crafter_hrl.planner.translator.state_to_text (el dataset ya lo usa). Si se cambia
el formato del traductor, hay que regenerar el dataset.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crafter_hrl.planner.llm import DEFAULT_BASE_MODEL  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos.

    Returns
    -------
    argparse.Namespace
        `base_model`, `data`, `out` y `load_in_4bit`.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--data", default=str(ROOT / "data/sft/train.jsonl"))
    p.add_argument("--out", default=str(ROOT / "checkpoints/llm/sft"))
    p.add_argument("--load-in-4bit", action="store_true", help="QLoRA si no entra en la GPU")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    """Entrena el adapter LoRA con SFT (sin implementar: ver el docstring del módulo).

    Parameters
    ----------
    args : argparse.Namespace
        Argumentos de la línea de comandos.

    Raises
    ------
    NotImplementedError
        Siempre, hasta que se implemente el entrenamiento.
    """
    raise NotImplementedError("TODO(C): implementar el SFT siguiendo el docstring.")


if __name__ == "__main__":
    main(parse_args())
