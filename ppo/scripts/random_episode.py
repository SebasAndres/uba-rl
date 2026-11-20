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

NEW_ACHIEVEMENT_REWARD = 0.5  # Crafter da +1 por logro nuevo (±0.1 son cambios de vida)


def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos.

    Returns
    -------
    argparse.Namespace
        `steps` y `seed`.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def describe_spaces(obs: np.ndarray, info: dict) -> None:
    """Imprime la forma de la observación, las acciones y las claves de `info`.

    Parameters
    ----------
    obs : np.ndarray
        Observación inicial.
    info : dict
        `info` inicial de Crafter.
    """
    print(f"obs: {obs.shape} {obs.dtype} | acciones ({len(ACTIONS)}): {ACTIONS}")
    print(f"info keys: {list(info)} | semantic {info['semantic'].shape} | player_pos {info['player_pos']}")


def play_random(env: CrafterEnv, steps: int, seed: int) -> tuple[float, dict]:
    """Juega acciones uniformes al azar e informa los logros nuevos.

    Parameters
    ----------
    env : CrafterEnv
        Entorno ya reseteado.
    steps : int
        Pasos máximos.
    seed : int
        Semilla del muestreo de acciones.

    Returns
    -------
    total_reward : float
        Suma de recompensas de Crafter.
    info : dict
        Último `info`.
    """
    rng = np.random.default_rng(seed)
    total_reward, info = 0.0, env.last_info
    for t in range(steps):
        _, reward, terminated, truncated, info = env.step(rng.integers(len(ACTIONS)))
        total_reward += reward
        if reward > NEW_ACHIEVEMENT_REWARD:
            print(f"t={t}: logro nuevo! {[k for k, v in info['achievements'].items() if v]}")
        if terminated or truncated:
            print(f"Fin del episodio en t={t} ({'murió' if terminated else 'límite'})")
            break
    return total_reward, info


def main() -> None:
    """Juega un episodio aleatorio y muestra el estado final traducido a texto."""
    args = parse_args()
    env = CrafterEnv(seed=args.seed)
    obs, info = env.reset()
    describe_spaces(obs, info)
    total_reward, info = play_random(env, args.steps, args.seed)
    print(f"\nRetorno: {total_reward:.2f}\n\nEstado traducido a texto:\n")
    print(state_to_text(info, "(misión de ejemplo)", []))


if __name__ == "__main__":
    main()
