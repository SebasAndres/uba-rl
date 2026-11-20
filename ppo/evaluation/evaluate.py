"""Evaluación: tasa de éxito por misión + Crafter score, para una configuración.

    python evaluation/evaluate.py --episodes 20                       # LLM + agente (checkpoints)
    python evaluation/evaluate.py --adapter checkpoints/llm/sft --split test
    python evaluation/evaluate.py --planner oracle --skills scripted  # referencia scripteada

Guarda un JSON en runs/eval/ para armar después la tabla comparativa
(PPO plano vs zero-shot vs SFT vs SFT+PPO, scripteadas vs PPO).
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crafter_hrl import PlannerEnv, TurnInfo  # noqa: E402
from crafter_hrl.env.constants import ACHIEVEMENTS  # noqa: E402
from crafter_hrl.factory import build_planner, build_skill_policy  # noqa: E402
from crafter_hrl.missions import MISSIONS, Mission, missions_by_split  # noqa: E402
from crafter_hrl.planner import Planner  # noqa: E402
from crafter_hrl.skills import SkillPolicy  # noqa: E402


def crafter_score(success_rates: dict[str, float]) -> float:
    """Media geométrica (con +1) de las tasas de éxito de los 22 logros, en %.

        S = exp(mean(ln(1 + s_i))) - 1,  s_i en [0, 100]  (Hafner, 2021).

    Parameters
    ----------
    success_rates : dict of str to float
        Fracción de episodios (en [0, 1]) en que se desbloqueó cada logro.

    Returns
    -------
    float
        Crafter score en %.
    """
    s = np.array([100.0 * success_rates.get(a, 0.0) for a in ACHIEVEMENTS])
    return float(np.exp(np.mean(np.log(1 + s))) - 1)


def select_missions(args: argparse.Namespace) -> list[Mission]:
    """Misiones a evaluar: las pasadas con --mission, o las del split elegido.

    Parameters
    ----------
    args : argparse.Namespace
        Argumentos de la línea de comandos.

    Returns
    -------
    list of Mission
        Misiones a evaluar.
    """
    if args.mission:
        return [MISSIONS[key] for key in args.mission]
    if args.split == "all":
        return list(MISSIONS.values())
    return missions_by_split(args.split)


def play_episode(env: PlannerEnv, planner: Planner, mission: Mission) -> tuple[dict, dict]:
    """Juega un episodio completo y resume cómo le fue.

    Parameters
    ----------
    env : PlannerEnv
        Entorno jerárquico (ya construido con su semilla).
    planner : Planner
        Planificador de nivel alto.
    mission : Mission
        Misión a jugar.

    Returns
    -------
    row : dict
        Métricas del episodio: success, died, return, decisions, env_steps, invalid.
    final_info : dict
        `info` de Crafter al terminar (para los logros).
    """
    text, info = env.reset(mission)
    planner.reset(mission)
    episode_return, done, invalid = 0.0, False, 0
    turn: TurnInfo | None = None
    while not done:
        plan = planner.plan(text, info)
        text, reward, terminated, truncated, turn = env.step(plan.text)
        info, done = turn.crafter_info, terminated or truncated
        episode_return += reward
        invalid += not turn.valid
    row = {"success": turn.mission_success, "died": turn.died, "return": episode_return,
           "decisions": env.decisions, "env_steps": turn.env_steps, "invalid": invalid}
    return row, info


def summarize_mission(mission: Mission, rows: list[dict]) -> dict:
    """Promedia las métricas de los episodios de una misión.

    Parameters
    ----------
    mission : Mission
        Misión evaluada.
    rows : list of dict
        Métricas por episodio (de `play_episode`).

    Returns
    -------
    dict
        split, success_rate, mean_return, mean_decisions, mean_env_steps,
        death_rate e invalid_per_episode.
    """
    def mean(key: str) -> float:
        """Promedio de la métrica `key` sobre los episodios."""
        return float(np.mean([r[key] for r in rows]))

    return {
        "split": mission.split, "success_rate": mean("success"),
        "mean_return": mean("return"), "mean_decisions": mean("decisions"),
        "mean_env_steps": mean("env_steps"), "death_rate": mean("died"),
        "invalid_per_episode": mean("invalid"),
    }


def print_mission_summary(mission: Mission, summary: dict) -> None:
    """Imprime una línea con el resumen de una misión.

    Parameters
    ----------
    mission : Mission
        Misión evaluada.
    summary : dict
        Resumen de `summarize_mission`.
    """
    print(f"{mission.key:15s} [{mission.split:5s}] éxito {summary['success_rate']:5.0%}  "
          f"decisiones {summary['mean_decisions']:5.1f}  "
          f"pasos {summary['mean_env_steps']:6.1f}  "
          f"muertes {summary['death_rate']:4.0%}")


def evaluate_missions(args: argparse.Namespace, skills: SkillPolicy, planner: Planner,
                      missions: list[Mission]) -> tuple[dict, dict[str, float]]:
    """Juega `args.episodes` episodios por misión y junta las métricas.

    Parameters
    ----------
    args : argparse.Namespace
        Argumentos (semillas, episodios, límites).
    skills : SkillPolicy
        Agente de habilidades.
    planner : Planner
        Planificador de nivel alto.
    missions : list of Mission
        Misiones a evaluar.

    Returns
    -------
    per_mission : dict
        Resumen por key de misión.
    achievement_rates : dict of str to float
        Fracción de episodios en que se desbloqueó cada logro.
    """
    per_mission = {}
    achieved: dict[str, int] = defaultdict(int)
    n_episodes = 0
    for mission in missions:
        rows = []
        for ep in range(args.episodes):
            env = PlannerEnv(skills, seed=args.seed + ep, max_env_steps=args.max_env_steps,
                             max_decisions=args.max_decisions)
            row, final_info = play_episode(env, planner, mission)
            rows.append(row)
            for achievement, count in final_info["achievements"].items():
                achieved[achievement] += count > 0
            n_episodes += 1
        per_mission[mission.key] = summarize_mission(mission, rows)
        print_mission_summary(mission, per_mission[mission.key])
    return per_mission, {a: achieved[a] / n_episodes for a in ACHIEVEMENTS}


def save_results(args: argparse.Namespace, run_name: str, per_mission: dict,
                 rates: dict[str, float], score: float) -> Path:
    """Guarda la configuración y los resultados en un JSON.

    Parameters
    ----------
    args : argparse.Namespace
        Configuración de la evaluación.
    run_name : str
        Prefijo del archivo (planificador_agente).
    per_mission : dict
        Resumen por misión.
    rates : dict of str to float
        Tasas de logros.
    score : float
        Crafter score.

    Returns
    -------
    Path
        Archivo escrito.
    """
    out = Path(args.out) / f"{run_name}_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "missions": per_mission,
                               "achievement_rates": rates, "crafter_score": score}, indent=2))
    return out


def run(args: argparse.Namespace) -> None:
    """Evalúa una configuración (planificador + agente) y guarda los resultados.

    Parameters
    ----------
    args : argparse.Namespace
        Argumentos de la línea de comandos.
    """
    skills = build_skill_policy(args.skills, args.skill_ckpt, seed=args.seed)
    planner = build_planner(args.planner, args.base_model, args.adapter, args.load_in_4bit)
    per_mission, rates = evaluate_missions(args, skills, planner, select_missions(args))
    score = crafter_score(rates)
    print(f"\nCrafter score (sobre los episodios de misión): {score:.1f}%")
    out = save_results(args, f"{planner.name}_{skills.name}", per_mission, rates, score)
    print(f"Resultados en {out}")


def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos.

    Returns
    -------
    argparse.Namespace
        Argumentos parseados.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planner", default="llm", choices=["llm", "oracle"])
    p.add_argument("--base-model", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--skills", default="ppo", choices=["ppo", "scripted"])
    p.add_argument("--skill-ckpt", default=None)
    p.add_argument("--split", default="all", choices=["train", "test", "all"])
    p.add_argument("--mission", nargs="*", default=None, help="misiones puntuales (pisa --split)")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000, help="semillas de evaluación: seed..seed+episodes")
    p.add_argument("--max-env-steps", type=int, default=2000)
    p.add_argument("--max-decisions", type=int, default=150)
    p.add_argument("--out", default="runs/eval")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
