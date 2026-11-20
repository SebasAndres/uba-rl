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
from typing import TextIO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crafter_hrl import PlannerEnv  # noqa: E402
from crafter_hrl.factory import build_skill_policy  # noqa: E402
from crafter_hrl.missions import Mission, missions_by_split  # noqa: E402
from crafter_hrl.planner.llm import build_messages  # noqa: E402
from crafter_hrl.planner.oracle import OraclePlanner  # noqa: E402


def record_episode(env: PlannerEnv, oracle: OraclePlanner, mission: Mission, seed: int) -> tuple[list[dict], bool]:
    """Juega un episodio con el oráculo y guarda un ejemplo de SFT por turno.

    Parameters
    ----------
    env : PlannerEnv
        Entorno jerárquico (ya construido con `seed`).
    oracle : OraclePlanner
        Planificador que hace de "profesor".
    mission : Mission
        Misión del episodio.
    seed : int
        Semilla del episodio (se guarda en cada fila).

    Returns
    -------
    rows : list of dict
        Ejemplos {prompt, completion, mission, seed, turn}.
    success : bool
        Si se cumplió la misión.
    """
    text, info = env.reset(mission)
    oracle.reset(mission)
    rows, done = [], False
    while not done:
        plan = oracle.plan(text, info)
        rows.append({"prompt": build_messages(text),
                     "completion": [{"role": "assistant", "content": plan.text}],
                     "mission": mission.key, "seed": seed, "turn": env.decisions})
        text, _, terminated, truncated, turn = env.step(plan.text)
        info, done = turn.crafter_info, terminated or truncated
    return rows, turn.mission_success


def write_rows(f: TextIO, rows: list[dict], success: bool) -> None:
    """Escribe los ejemplos de un episodio en JSONL.

    Parameters
    ----------
    f : TextIO
        Archivo abierto para escritura.
    rows : list of dict
        Ejemplos del episodio.
    success : bool
        Si el episodio fue exitoso (se agrega como `episode_success`).
    """
    for row in rows:
        row["episode_success"] = success
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(args: argparse.Namespace) -> None:
    """Genera el dataset para todas las misiones de train.

    Parameters
    ----------
    args : argparse.Namespace
        Argumentos de la línea de comandos.
    """
    skills = build_skill_policy(args.skills, seed=args.seed)
    oracle = OraclePlanner()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_episodes = n_success = 0
    with out.open("w") as f:
        for mission in missions_by_split("train"):
            for ep in range(args.episodes_per_mission):
                seed = args.seed + ep
                env = PlannerEnv(skills, seed=seed, max_env_steps=args.max_env_steps)
                rows, success = record_episode(env, oracle, mission, seed)
                n_episodes += 1
                n_success += success
                if success or args.keep_failed:
                    write_rows(f, rows, success)
                    n_rows += len(rows)
            print(f"{mission.key}: listo ({n_success}/{n_episodes} exitosos acumulados)")
    print(f"{n_rows} ejemplos de {n_episodes} episodios -> {out}")


def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos.

    Returns
    -------
    argparse.Namespace
        Argumentos parseados.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes-per-mission", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="semillas de train: NO solapar con las de evaluación (1000+)")
    p.add_argument("--skills", default="scripted", choices=["scripted", "ppo"])
    p.add_argument("--max-env-steps", type=int, default=2000)
    p.add_argument("--keep-failed", action="store_true", help="guardar también episodios fallidos")
    p.add_argument("--out", default=str(ROOT / "data/sft/train.jsonl"))
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
