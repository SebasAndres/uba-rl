"""Paso 1 de "primeros pasos": un episodio aleatorio para inspeccionar obs e info.

    python scripts/random_episode.py --steps 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crafter_hrl.env import CrafterEnv  # noqa: E402
from crafter_hrl.env.constants import ACTIONS  # noqa: E402
from crafter_hrl.planner.translator import state_to_text  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = CrafterEnv(seed=args.seed)
    obs, info = env.reset()
    print(f"obs: {obs.shape} {obs.dtype} | acciones ({len(ACTIONS)}): {ACTIONS}")
    print(f"info keys: {list(info)} | semantic {info['semantic'].shape} | player_pos {info['player_pos']}")
    rng = np.random.default_rng(args.seed)
    total = 0.0
    for t in range(args.steps):
        obs, r, term, trunc, info = env.step(rng.integers(len(ACTIONS)))
        total += r
        if r > 0.5:
            print(f"t={t}: logro nuevo! {[k for k, v in info['achievements'].items() if v]}")
        if term or trunc:
            print(f"Fin del episodio en t={t} ({'murió' if term else 'límite'})")
            break
    print(f"\nRetorno: {total:.2f}\n\nEstado traducido a texto:\n")
    print(state_to_text(info, "(misión de ejemplo)", []))
