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

from crafter_hrl import PlannerEnv  # noqa: E402
from crafter_hrl.env.constants import ACHIEVEMENTS  # noqa: E402
from crafter_hrl.factory import build_planner, build_skill_policy  # noqa: E402
from crafter_hrl.missions import MISSIONS, missions_by_split  # noqa: E402


def crafter_score(success_rates: dict[str, float]) -> float:
    """Media geométrica (con +1) de las tasas de éxito de los 22 logros, en %.

    S = exp(mean(ln(1 + s_i))) - 1,  s_i en [0, 100]  (Hafner, 2021).
    """
    s = np.array([100.0 * success_rates.get(a, 0.0) for a in ACHIEVEMENTS])
    return float(np.exp(np.mean(np.log(1 + s))) - 1)


def run(args):
    skills = build_skill_policy(args.skills, args.skill_ckpt, seed=args.seed)
    planner = build_planner(args.planner, args.base_model, args.adapter, args.load_in_4bit)
    missions = (list(MISSIONS.values()) if args.split == "all" else missions_by_split(args.split)) \
        if not args.mission else [MISSIONS[m] for m in args.mission]

    per_mission = {}
    achieved = defaultdict(int)
    n_total = 0
    for mission in missions:
        rows = []
        for ep in range(args.episodes):
            env = PlannerEnv(skills, seed=args.seed + ep, max_env_steps=args.max_env_steps,
                             max_decisions=args.max_decisions)
            text, info = env.reset(mission)
            planner.reset(mission)
            ret, done, invalid = 0.0, False, 0
            while not done:
                out = planner.plan(text, info)
                text, r, term, trunc, turn = env.step(out.text)
                info, ret, done = turn.crafter_info, ret + r, term or trunc
                invalid += not turn.valid
            rows.append({"success": turn.mission_success, "died": turn.died, "return": ret,
                         "decisions": env.decisions, "env_steps": turn.env_steps, "invalid": invalid})
            for a, c in info["achievements"].items():
                achieved[a] += c > 0
            n_total += 1
        succ = np.mean([r["success"] for r in rows])
        per_mission[mission.key] = {
            "split": mission.split, "success_rate": float(succ),
            "mean_return": float(np.mean([r["return"] for r in rows])),
            "mean_decisions": float(np.mean([r["decisions"] for r in rows])),
            "mean_env_steps": float(np.mean([r["env_steps"] for r in rows])),
            "death_rate": float(np.mean([r["died"] for r in rows])),
            "invalid_per_episode": float(np.mean([r["invalid"] for r in rows])),
        }
        print(f"{mission.key:15s} [{mission.split:5s}] éxito {succ:5.0%}  "
              f"decisiones {per_mission[mission.key]['mean_decisions']:5.1f}  "
              f"pasos {per_mission[mission.key]['mean_env_steps']:6.1f}  "
              f"muertes {per_mission[mission.key]['death_rate']:4.0%}")

    rates = {a: achieved[a] / n_total for a in ACHIEVEMENTS}
    score = crafter_score(rates)
    print(f"\nCrafter score (sobre los episodios de misión): {score:.1f}%")
    out = Path(args.out) / f"{planner.name}_{skills.name}_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "missions": per_mission,
                               "achievement_rates": rates, "crafter_score": score}, indent=2))
    print(f"Resultados en {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
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
    run(p.parse_args())
