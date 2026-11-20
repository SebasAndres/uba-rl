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
from typing import Any

from ..missions.missions import Mission
from .base import PlannerOutput
from .parser import parse_skill
from .translator import SYSTEM_PROMPT

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Contrato del adapter: usar ESTA config en crear_dummies, SFT y PPO.
LORA_CONFIG = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                   task_type="CAUSAL_LM")


def build_messages(state_text: str) -> list[dict]:
    """Arma el prompt en formato chat. Usar ESTA función también en SFT y PPO.

    Parameters
    ----------
    state_text : str
        Estado traducido a texto (`state_to_text`).

    Returns
    -------
    list of dict
        Mensajes [system, user] para `apply_chat_template`.
    """
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state_text}]


def create_lora_adapter(out_dir: str | Path, base_model: str = DEFAULT_BASE_MODEL) -> Path:
    """Crea y guarda un adapter LoRA sin entrenar (el LLM 'dummy').

    Parameters
    ----------
    out_dir : str or Path
        Carpeta donde guardar el adapter y el tokenizer.
    base_model : str, optional
        Modelo base de Hugging Face.

    Returns
    -------
    Path
        La carpeta del adapter.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
    return out_dir


def _load_model(base_model: str, adapter: str | Path | None, device: str, load_in_4bit: bool) -> Any:
    """Carga el modelo base, le aplica el adapter LoRA y lo pone en modo eval.

    Parameters
    ----------
    base_model : str
        Modelo base de Hugging Face.
    adapter : str or Path or None
        Carpeta del adapter LoRA; None usa el modelo base solo.
    device : str
        "cuda" o "cpu".
    load_in_4bit : bool
        Cuantizar a 4 bits con bitsandbytes (queda en el dispositivo que elija HF).

    Returns
    -------
    transformers.PreTrainedModel
        Modelo listo para generar.
    """
    import torch
    from transformers import AutoModelForCausalLM

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True)}
    else:
        kwargs = {"dtype": torch.bfloat16 if device == "cuda" else torch.float32}
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    if not load_in_4bit:
        model = model.to(device)
    return model.eval()


class LLMPlanner:
    """Planificador de nivel alto basado en un LLM con adapter LoRA.

    Parameters
    ----------
    base_model : str, optional
        Modelo base de Hugging Face.
    adapter : str or Path or None, optional
        Carpeta del adapter LoRA (dummy, SFT o PPO).
    device : str or None, optional
        Dispositivo; por defecto cuda si está disponible.
    load_in_4bit : bool, optional
        Cuantizar a 4 bits.
    max_new_tokens : int, optional
        Largo máximo de la respuesta.
    temperature : float, optional
        Temperatura de muestreo; 0 = greedy.
    """

    name = "llm"

    def __init__(self, base_model: str = DEFAULT_BASE_MODEL, adapter: str | Path | None = None,
                 device: str | None = None, load_in_4bit: bool = False,
                 max_new_tokens: int = 48, temperature: float = 0.0) -> None:
        import torch
        from transformers import AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or base_model)
        self.model = _load_model(base_model, adapter, self.device, load_in_4bit)
        self.name = f"llm[{Path(adapter).name if adapter else 'base'}]"

    def reset(self, mission: Mission) -> None:
        """No hace nada: la misión ya viene en el texto de cada turno.

        Parameters
        ----------
        mission : Mission
            Misión del episodio.
        """

    def plan(self, text: str, info: dict) -> PlannerOutput:
        """Genera el plan del turno a partir del estado en texto.

        Parameters
        ----------
        text : str
            Estado traducido a texto.
        info : dict
            `info` de Crafter (no se usa).

        Returns
        -------
        PlannerOutput
            Respuesta del LLM y la habilidad parseada (None si no es válida).
        """
        import torch

        prompt = self.tokenizer.apply_chat_template(build_messages(text), tokenize=False,
                                                    add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=do_sample,
                temperature=self.temperature if do_sample else None, top_p=None, top_k=None,
                pad_token_id=self.tokenizer.eos_token_id)
        prompt_len = inputs["input_ids"].shape[1]
        completion = self.tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
        return PlannerOutput(completion.strip(), parse_skill(completion))
