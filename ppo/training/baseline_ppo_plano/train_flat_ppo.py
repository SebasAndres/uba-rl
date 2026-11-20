"""BASELINE — PPO plano en Crafter (sin LLM, sin habilidades), 1M de pasos.

Es la fila de comparación "PPO plano" de la evaluación: un agente PPO que ve
sólo la imagen y optimiza la recompensa nativa de Crafter (+1 por logro nuevo,
±0.1 por cambio de vida). Se mide con el Crafter score.

ESQUELETO: el wrapper gymnasium está hecho; el loop es el de
../../../cleanrl/cleanrl/ppo_atari.py casi sin cambios (sacar los wrappers de
Atari: NoopReset, FireReset, frame-skip, grayscale; dejar la imagen RGB 64x64).
Referencias: paper de Crafter (Hafner, 2021), PPO ~ 4.6% de score a 1M pasos.

    python training/baseline_ppo_plano/train_flat_ppo.py
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crafter_hrl.env import CrafterEnv  # noqa: E402
from crafter_hrl.env.constants import NUM_ACTIONS  # noqa: E402


class FlatCrafter(gym.Env):
    """Crafter con API gymnasium. info["achievements"] al final del episodio para el score."""

    def __init__(self, seed: int | None = None, length: int = 10000):
        self.crafter = CrafterEnv(seed=seed, length=length)
        self.observation_space = gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.crafter.reset()
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.crafter.step(action)
        out = {"achievements": info["achievements"]} if terminated or truncated else {}
        return obs, reward, terminated, truncated, out


if __name__ == "__main__":
    # TODO(D o A): copiar el loop de ppo_atari.py con:
    #   envs = gym.vector.AsyncVectorEnv([lambda i=i: FlatCrafter(seed=i) for i in range(16)])
    #   red: la misma CNN que SkillAgent pero sin goal_vec (o SkillAgent con goal en cero)
    #   total_timesteps = 1_000_000
    # y al final guardar las tasas de logros por episodio para evaluation/evaluate.crafter_score.
    raise NotImplementedError("Completar con el loop de cleanrl/ppo_atari.py")
