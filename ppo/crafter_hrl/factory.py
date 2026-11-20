"""Construye agente de habilidades y planificador a partir de flags/checkpoints.

Por defecto se usan los MODELOS (agente PPO + LLM) cargados desde checkpoints/.
Antes de entrenar, esos checkpoints son dummies (pesos aleatorios / LoRA sin
entrenar) creados con `python scripts/crear_dummies.py`. Entrenar = sobrescribir.

Las versiones scripteadas (oráculo, habilidades con BFS) quedan como opción
explícita: generan el dataset de SFT y sirven de baseline/ablación.
"""

from pathlib import Path

from .planner.base import Planner
from .skills.executor import SkillPolicy

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
SKILL_CKPT = CKPT / "skills" / "skill_policy.pt"
LLM_DUMMY = CKPT / "llm" / "dummy"
LLM_SFT = CKPT / "llm" / "sft"
LLM_PPO = CKPT / "llm" / "ppo"

_HINT = "Corré primero:  python scripts/crear_dummies.py"


def build_skill_policy(kind: str = "ppo", ckpt: str | Path | None = None, device: str | None = None,
                       seed: int = 0, greedy: bool = False) -> SkillPolicy:
    """Construye el agente de habilidades (nivel bajo).

    Parameters
    ----------
    kind : str, optional
        "ppo" (red entrenada con RL) o "scripted" (BFS + reglas).
    ckpt : str or Path or None, optional
        Checkpoint del agente PPO; por defecto `SKILL_CKPT`.
    device : str or None, optional
        Dispositivo del agente PPO.
    seed : int, optional
        Semilla del agente scripteado.
    greedy : bool, optional
        Si el agente PPO toma la acción más probable en vez de muestrear.

    Returns
    -------
    SkillPolicy
        Agente listo para `run_skill`.

    Raises
    ------
    FileNotFoundError
        Si `kind="ppo"` y no existe el checkpoint.
    ValueError
        Si `kind` no es válido.
    """
    if kind == "scripted":
        from .skills.scripted import ScriptedSkillPolicy
        return ScriptedSkillPolicy(seed=seed)
    if kind == "ppo":
        from .skills.policy import PPOSkillPolicy
        ckpt = Path(ckpt) if ckpt else SKILL_CKPT
        if not ckpt.exists():
            raise FileNotFoundError(f"No existe el checkpoint del agente {ckpt}. {_HINT}")
        return PPOSkillPolicy(ckpt, device=device, greedy=greedy)
    raise ValueError(kind)


def resolve_adapter(adapter: str | Path | None = None) -> Path:
    """Elige el adapter LoRA: el pasado por flag, o el más entrenado que exista.

    Parameters
    ----------
    adapter : str or Path or None, optional
        Carpeta explícita; si es None se busca ppo > sft > dummy.

    Returns
    -------
    Path
        Carpeta del adapter.

    Raises
    ------
    FileNotFoundError
        Si no hay ningún adapter en checkpoints/llm.
    """
    if adapter:
        return Path(adapter)
    for path in (LLM_PPO, LLM_SFT, LLM_DUMMY):
        if path.exists():
            return path
    raise FileNotFoundError(f"No hay adapters del LLM en {CKPT / 'llm'}. {_HINT}")


def build_planner(kind: str = "llm", base_model: str | None = None, adapter: str | Path | None = None,
                  load_in_4bit: bool = False, temperature: float = 0.0) -> Planner:
    """Construye el planificador (nivel alto).

    Parameters
    ----------
    kind : str, optional
        "llm" o "oracle" (reglas scripteadas).
    base_model : str or None, optional
        Modelo base de Hugging Face; por defecto `DEFAULT_BASE_MODEL`.
    adapter : str or Path or None, optional
        Carpeta del adapter LoRA (ver `resolve_adapter`).
    load_in_4bit : bool, optional
        Cuantizar el LLM a 4 bits.
    temperature : float, optional
        Temperatura de muestreo del LLM (0 = greedy).

    Returns
    -------
    Planner
        Planificador listo para `plan(text, info)`.

    Raises
    ------
    ValueError
        Si `kind` no es válido.
    """
    if kind == "oracle":
        from .planner.oracle import OraclePlanner
        return OraclePlanner()
    if kind == "llm":
        from .planner.llm import DEFAULT_BASE_MODEL, LLMPlanner
        return LLMPlanner(base_model or DEFAULT_BASE_MODEL, adapter=resolve_adapter(adapter),
                          load_in_4bit=load_in_4bit, temperature=temperature)
    raise ValueError(kind)
