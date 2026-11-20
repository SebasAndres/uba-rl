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
from .skills.catalog import is_feasible, missing_requirements
from .skills.executor import SkillPolicy, SkillResult, run_skill


@dataclass
class RewardConfig:
    success: float = 1.0
    death: float = -0.5
    invalid: float = -0.5
    per_decision: float = -0.02
    prerequisite: float = 0.0  # poner 0.2 para activar el shaping opcional


@dataclass
class TurnInfo:
    llm_text: str
    skill: str | None
    valid: bool
    skill_result: SkillResult | None
    mission_success: bool
    died: bool
    env_steps: int
    crafter_info: dict = field(repr=False, default_factory=dict)


class PlannerEnv:
    def __init__(
        self,
        skill_policy: SkillPolicy,
        missions: list[Mission] | None = None,
        seed: int | None = None,
        max_env_steps: int = 2000,
        max_decisions: int = 150,
        rewards: RewardConfig | None = None,
        check_feasibility: bool = True,
        interrupt_on_damage: bool = True,
        list_skills: str = "factibles",
        on_env_step: Callable[[CrafterEnv, int, dict], None] | None = None,
    ):
        self.interrupt_on_damage = interrupt_on_damage
        self.list_skills = list_skills
        self.crafter = CrafterEnv(seed=seed, length=max_env_steps)
        self.skill_policy = skill_policy
        self.missions = missions or list(MISSIONS.values())
        self.max_decisions = max_decisions
        self.rewards = rewards or RewardConfig()
        self.check_feasibility = check_feasibility
        self.on_env_step = on_env_step
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ API
    def reset(self, mission: Mission | str | None = None):
        if isinstance(mission, str):
            mission = MISSIONS[mission]
        self.mission = mission or self.missions[self.rng.integers(len(self.missions))]
        self.obs, self.info = self.crafter.reset()
        self.history: list[HistoryEntry] = []
        self.decisions = 0
        self.unlocked = {k for k, v in self.info["achievements"].items() if v > 0}
        return self.text(), self.info

    def text(self) -> str:
        return state_to_text(self.info, self.mission.text, self.history, list_skills=self.list_skills)

    def _make_stop_fn(self):
        """Corta la habilidad si se cumplió la misión o (opcional) si el jugador recibe daño.

        Sin la interrupción por daño, un zombi que aparece a mitad de una habilidad
        de 100 pasos mata al jugador sin que el LLM pueda reaccionar.
        """
        last = {"health": self.info["inventory"]["health"]}

        def stop(info: dict) -> bool:
            hurt = info["inventory"]["health"] < last["health"]
            last["health"] = info["inventory"]["health"]
            return self.mission.check(info) or (self.interrupt_on_damage and hurt)

        return stop

    def step(self, llm_text: str):
        self.decisions += 1
        reward = self.rewards.per_decision
        skill = parse_skill(llm_text)
        valid = skill is not None and (not self.check_feasibility or is_feasible(skill, self.info["inventory"]))
        result = None
        died = truncated = False

        if not valid:
            reward += self.rewards.invalid
            label = f"inválida: {skill}" if skill else "inválida"
            if skill is not None:
                faltan = missing_requirements(skill, self.info["inventory"])
                label += f" (falta {', '.join(ES[k] for k in faltan)})" if faltan else ""
            self.history.append(HistoryEntry(label, False))
        else:
            result = run_skill(self.crafter, self.skill_policy, skill, self.obs, self.info,
                               stop_fn=self._make_stop_fn(), on_step=self.on_env_step)
            self.obs, self.info = result.obs, result.info
            died, truncated = result.terminated, result.truncated
            cut = result.stopped_early and not result.success and not self.mission.check(self.info)
            self.history.append(HistoryEntry(str(skill) + (" [interrumpida: daño]" if cut else ""),
                                             result.success))
            new = {k for k, v in self.info["achievements"].items() if v > 0} - self.unlocked
            reward += self.rewards.prerequisite * len(new)
            self.unlocked |= new

        success = self.mission.check(self.info)
        if success:
            reward += self.rewards.success
        if died:
            reward += self.rewards.death
        terminated = success or died
        truncated = not terminated and (truncated or self.decisions >= self.max_decisions)
        info = TurnInfo(llm_text, str(skill) if skill else None, valid, result, success, died,
                        self.crafter.num_steps, self.info)
        return self.text(), reward, terminated, truncated, info
