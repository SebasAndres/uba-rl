"""Genera el dataset de SFT del LLM con el oráculo (misiones de TRAIN).

Corre oráculo + habilidades scripteadas en muchas semillas y guarda, por cada
turno, el par (prompt, respuesta del oráculo) en formato chat (JSONL):

    {"prompt": [{"role": "system", ...}, {"role": "user", "content": <estado>}],
     "completion": [{"role": "assistant", "content": "Razonamiento: ...\nHabilidad: ..."}],
     "mission": "espada_piedra", "seed": 3, "turn": 5, "episode_success": true}

Por defecto sólo guarda episodios exitosos (filtrado tipo rejection sampling).

    python training/fase2_sft/generar_dataset.py --episodes-per-mission 200
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crafter_hrl import PlannerEnv  # noqa: E402
from crafter_hrl.factory import build_skill_policy  # noqa: E402
from crafter_hrl.missions import missions_by_split  # noqa: E402
from crafter_hrl.planner.llm import build_messages  # noqa: E402
from crafter_hrl.planner.oracle import OraclePlanner  # noqa: E402


def main(args):
    skills = build_skill_policy(args.skills, seed=args.seed)
    oracle = OraclePlanner()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_eps = n_ok = 0
    with out.open("w") as f:
        for mission in missions_by_split("train"):
            for ep in range(args.episodes_per_mission):
                seed = args.seed + ep
                env = PlannerEnv(skills, seed=seed, max_env_steps=args.max_env_steps)
                text, info = env.reset(mission)
                oracle.reset(mission)
                rows, done = [], False
                while not done:
                    out_ = oracle.plan(text, info)
                    rows.append({"prompt": build_messages(text),
                                 "completion": [{"role": "assistant", "content": out_.text}],
                                 "mission": mission.key, "seed": seed, "turn": env.decisions})
                    text, _, term, trunc, turn = env.step(out_.text)
                    info, done = turn.crafter_info, term or trunc
                n_eps += 1
                n_ok += turn.mission_success
                if turn.mission_success or args.keep_failed:
                    for r in rows:
                        r["episode_success"] = turn.mission_success
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_rows += len(rows)
            print(f"{mission.key}: listo ({n_ok}/{n_eps} exitosos acumulados)")
    print(f"{n_rows} ejemplos de {n_eps} episodios -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-per-mission", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="semillas de train: NO solapar con las de evaluación (1000+)")
    p.add_argument("--skills", default="scripted", choices=["scripted", "ppo"])
    p.add_argument("--max-env-steps", type=int, default=2000)
    p.add_argument("--keep-failed", action="store_true")
    p.add_argument("--out", default=str(ROOT / "data/sft/train.jsonl"))
    main(p.parse_args())
