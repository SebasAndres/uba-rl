"""Planificador LLM: Qwen2.5 + adapter LoRA.

El juego SIEMPRE carga base + adapter, así el flujo es idéntico antes y después
de entrenar; lo único que cambia es qué adapter se carga:

  - dummy:  checkpoints/llm/dummy  LoRA recién inicializado (B=0 => igual al modelo base)
  - SFT:    checkpoints/llm/sft    (training/fase2_sft)
  - PPO:    checkpoints/llm/ppo    (training/fase2_ppo_llm)

El dummy se crea con `python scripts/crear_dummies.py`. SFT y PPO tienen que
usar `LORA_CONFIG` para que los adapters sean intercambiables.
"""

from pathlib import Path

from ..missions.missions import Mission
from .base import PlannerOutput
from .parser import parse_skill
from .translator import SYSTEM_PROMPT

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Contrato del adapter: usar ESTA config en crear_dummies, SFT y PPO.
LORA_CONFIG = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                   task_type="CAUSAL_LM")


def build_messages(state_text: str) -> list[dict]:
    """Prompt en formato chat. Usar ESTA función también en SFT y PPO."""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state_text}]


def create_lora_adapter(out_dir: str | Path, base_model: str = DEFAULT_BASE_MODEL) -> Path:
    """Crea y guarda un adapter LoRA sin entrenar (el LLM 'dummy')."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
    return out_dir


class LLMPlanner:
    name = "llm"

    def __init__(self, base_model: str = DEFAULT_BASE_MODEL, adapter: str | Path | None = None,
                 device: str | None = None, load_in_4bit: bool = False,
                 max_new_tokens: int = 48, temperature: float = 0.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or base_model)
        kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kwargs["dtype"] = torch.bfloat16 if self.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        if adapter is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        if not load_in_4bit:
            model = model.to(self.device)
        self.model = model.eval()
        self.name = f"llm[{Path(adapter).name if adapter else 'base'}]"

    def reset(self, mission: Mission) -> None:
        pass

    def plan(self, text: str, info: dict) -> PlannerOutput:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            build_messages(text), tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        sample = self.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=sample,
                temperature=self.temperature if sample else None, top_p=None, top_k=None,
                pad_token_id=self.tokenizer.eos_token_id)
        completion = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return PlannerOutput(completion.strip(), parse_skill(completion))
