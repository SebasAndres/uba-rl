"""Entorno de la FASE 2 (y del juego/evaluación): el loop jerárquico.

Un paso de este entorno = UN turno del LLM:

    texto del LLM -> parser -> Skill -> goal_vec -> agente congelado ejecuta
    (hasta cumplir o timeout) -> estado a texto -> próximo turno

    obs_texto, info = env.reset(mission)
    obs_texto, r, terminated, truncated, info = env.step(respuesta_del_llm)

Recompensas (IDEA.md):
    +1    misión cumplida (termina el episodio)
    -0.5  muere (termina) o pide algo inválido (sigue)
    -0.02 por decisión
    +0.2  (opcional, `prerequisite`) primera vez que se desbloquea un logro en el episodio
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .env import CrafterEnv
from .env.constants import ES
from .missions import MISSIONS, Mission
from .planner.parser import parse_skill
from .planner.translator import HistoryEntry, state_to_text
from .skills.catalog import Skill, is_feasible, missing_requirements
from .skills.executor import SkillPolicy, SkillResult, run_skill

StepCallback = Callable[[CrafterEnv, int, dict], None]


@dataclass
class RewardConfig:
    """Recompensas del LLM planificador (ver docstring del módulo)."""

    success: float = 1.0
    death: float = -0.5
    invalid: float = -0.5
    per_decision: float = -0.02
    prerequisite: float = 0.0  # poner 0.2 para activar el shaping opcional


@dataclass
class TurnInfo:
    """Detalle de un turno (lo que devuelve `PlannerEnv.step` como info).

    Attributes
    ----------
    llm_text : str
        Respuesta cruda del LLM.
    skill : str or None
        Habilidad parseada, o None si no se pudo parsear.
    valid : bool
        Si la habilidad era parseable y factible.
    skill_result : SkillResult or None
        Resultado de la ejecución del agente (None si fue inválida).
    mission_success : bool
        Si la misión quedó cumplida.
    died : bool
        Si el jugador murió.
    env_steps : int
        Pasos primitivos acumulados en el episodio.
    crafter_info : dict
        `info` de Crafter después del turno.
    """

    llm_text: str
    skill: str | None
    valid: bool
    skill_result: SkillResult | None
    mission_success: bool
    died: bool
    env_steps: int
    crafter_info: dict = field(repr=False, default_factory=dict)


def _unlocked_achievements(info: dict) -> set[str]:
    """Logros con contador > 0.

    Parameters
    ----------
    info : dict
        `info` de Crafter.

    Returns
    -------
    set of str
        Nombres de los logros desbloqueados.
    """
    return {name for name, count in info["achievements"].items() if count > 0}


class PlannerEnv:
    """Entorno a nivel de turno: la acción es el texto del LLM, el agente de habilidades la ejecuta.

    Parameters
    ----------
    skill_policy : SkillPolicy
        Agente de habilidades (nivel bajo) que ejecuta cada habilidad pedida.
    missions : list of Mission or None, optional
        Misiones a sortear en `reset` si no se pasa una.
    seed : int or None, optional
        Semilla de Crafter y del sorteo de misiones.
    max_env_steps : int, optional
        Pasos primitivos máximos del episodio de Crafter.
    max_decisions : int, optional
        Turnos máximos del LLM.
    rewards : RewardConfig or None, optional
        Recompensas.
    check_feasibility : bool, optional
        Si es True, pedir algo sin los materiales cuenta como inválido.
    interrupt_on_damage : bool, optional
        Cortar la habilidad si el jugador recibe daño (para que el LLM reaccione).
    list_skills : str, optional
        "factibles" o "todas": qué habilidades se listan en el prompt.
    on_env_step : callable or None, optional
        Callback `(env, action, info)` en cada paso primitivo (ej. render).
    """

    def __init__(self, skill_policy: SkillPolicy, missions: list[Mission] | None = None,
                 seed: int | None = None, max_env_steps: int = 2000, max_decisions: int = 150,
                 rewards: RewardConfig | None = None, check_feasibility: bool = True,
                 interrupt_on_damage: bool = True, list_skills: str = "factibles",
                 on_env_step: StepCallback | None = None) -> None:
        self.crafter = CrafterEnv(seed=seed, length=max_env_steps)
        self.skill_policy = skill_policy
        self.missions = missions or list(MISSIONS.values())
        self.max_decisions = max_decisions
        self.rewards = rewards or RewardConfig()
        self.check_feasibility = check_feasibility
        self.interrupt_on_damage = interrupt_on_damage
        self.list_skills = list_skills
        self.on_env_step = on_env_step
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ API
    def reset(self, mission: Mission | str | None = None) -> tuple[str, dict]:
        """Empieza un episodio nuevo.

        Parameters
        ----------
        mission : Mission or str or None, optional
            Misión (o su key); si es None se sortea de `self.missions`.

        Returns
        -------
        text : str
            Estado inicial en texto (entrada del LLM en el turno 1).
        info : dict
            `info` inicial de Crafter.
        """
        if isinstance(mission, str):
            mission = MISSIONS[mission]
        self.mission = mission or self.missions[self.rng.integers(len(self.missions))]
        self.obs, self.info = self.crafter.reset()
        self.history: list[HistoryEntry] = []
        self.decisions = 0
        self.unlocked = _unlocked_achievements(self.info)
        return self.text(), self.info

    def text(self) -> str:
        """Estado actual traducido a texto.

        Returns
        -------
        str
            Texto que lee el LLM.
        """
        return state_to_text(self.info, self.mission.text, self.history, list_skills=self.list_skills)

    def step(self, llm_text: str) -> tuple[str, float, bool, bool, TurnInfo]:
        """Juega un turno: parsea la habilidad del LLM y la hace ejecutar al agente.

        Parameters
        ----------
        llm_text : str
            Respuesta del LLM (razonamiento + habilidad).

        Returns
        -------
        text : str
            Estado nuevo en texto (entrada del próximo turno).
        reward : float
            Recompensa del turno.
        terminated : bool
            Misión cumplida o jugador muerto.
        truncated : bool
            Se agotaron los pasos de Crafter o las decisiones.
        turn : TurnInfo
            Detalle del turno.
        """
        self.decisions += 1
        skill = parse_skill(llm_text)
        valid = skill is not None and (not self.check_feasibility or is_feasible(skill, self.info["inventory"]))
        reward = self.rewards.per_decision
        if valid:
            result = self._execute(skill)
            reward += self.rewards.prerequisite * self._count_new_achievements()
        else:
            result = None
            reward += self.rewards.invalid
            self._record_invalid(skill)

        success = self.mission.check(self.info)
        died = result is not None and result.terminated
        world_over = result is not None and result.truncated
        if success:
            reward += self.rewards.success
        if died:
            reward += self.rewards.death
        terminated = success or died
        truncated = not terminated and (world_over or self.decisions >= self.max_decisions)
        turn = TurnInfo(llm_text, str(skill) if skill else None, valid, result, success, died,
                        self.crafter.num_steps, self.info)
        return self.text(), reward, terminated, truncated, turn

    # ------------------------------------------------------------------ helpers
    def _make_stop_fn(self) -> Callable[[dict], bool]:
        """Corta la habilidad si se cumplió la misión o (opcional) si el jugador recibe daño.

        Sin la interrupción por daño, un zombi que aparece a mitad de una habilidad
        de 100 pasos mata al jugador sin que el LLM pueda reaccionar.

        Returns
        -------
        callable
            Función `stop(info) -> bool` con memoria de la última vida vista.
        """
        last = {"health": self.info["inventory"]["health"]}

        def stop(info: dict) -> bool:
            """True si la misión se cumplió o (si está activado) el jugador perdió vida."""
            hurt = info["inventory"]["health"] < last["health"]
            last["health"] = info["inventory"]["health"]
            return self.mission.check(info) or (self.interrupt_on_damage and hurt)

        return stop

    def _execute(self, skill: Skill) -> SkillResult:
        """Hace que el agente de habilidades ejecute `skill` y registra el resultado.

        Parameters
        ----------
        skill : Skill
            Habilidad factible pedida por el LLM.

        Returns
        -------
        SkillResult
            Resultado de la ejecución.
        """
        result = run_skill(self.crafter, self.skill_policy, skill, self.obs, self.info,
                           stop_fn=self._make_stop_fn(), on_step=self.on_env_step)
        self.obs, self.info = result.obs, result.info
        cut_by_damage = result.stopped_early and not result.success and not self.mission.check(self.info)
        label = str(skill) + (" [interrumpida: daño]" if cut_by_damage else "")
        self.history.append(HistoryEntry(label, result.success))
        return result

    def _record_invalid(self, skill: Skill | None) -> None:
        """Agrega al historial un pedido inválido, con los materiales que faltaban.

        Parameters
        ----------
        skill : Skill or None
            Habilidad parseada (no factible), o None si no se pudo parsear.
        """
        label = f"inválida: {skill}" if skill else "inválida"
        if skill is not None:
            missing = missing_requirements(skill, self.info["inventory"])
            if missing:
                label += f" (falta {', '.join(ES[item] for item in missing)})"
        self.history.append(HistoryEntry(label, False))

    def _count_new_achievements(self) -> int:
        """Cuenta (y marca) los logros desbloqueados por primera vez en el episodio.

        Returns
        -------
        int
            Cantidad de logros nuevos.
        """
        new = _unlocked_achievements(self.info) - self.unlocked
        self.unlocked |= new
        return len(new)
