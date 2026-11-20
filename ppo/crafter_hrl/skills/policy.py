"""Agente de habilidades PPO (fase 1): red + formato de checkpoint.

La MISMA clase la usan el script de entrenamiento (training/fase1_habilidades)
y el script de juego (scripts/play.py). Si se cambia la arquitectura, hay que
mantener `save_skill_policy` / `load_skill_policy` compatibles.

Arquitectura base (a decidir/ajustar por A): CNN estilo Nature (como
cleanrl/ppo_atari.py) sobre la imagen 64x64x3, concatenada con el goal_vec,
y dos cabezas: actor (17 logits) y crítico (1 valor).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..env.constants import NUM_ACTIONS
from .catalog import GOAL_DIM, Skill, encode


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SkillAgent(nn.Module):
    def __init__(self, goal_dim: int = GOAL_DIM, num_actions: int = NUM_ACTIONS, hidden: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(                                    # 3x64x64
            layer_init(nn.Conv2d(3, 32, 8, stride=4)), nn.ReLU(),     # 32x15x15
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),    # 64x6x6
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),    # 64x4x4
            nn.Flatten(),
        )
        self.goal = nn.Sequential(layer_init(nn.Linear(goal_dim, 64)), nn.ReLU())
        self.trunk = nn.Sequential(layer_init(nn.Linear(64 * 4 * 4 + 64, hidden)), nn.ReLU())
        self.actor = layer_init(nn.Linear(hidden, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def features(self, image: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # image: (B, 64, 64, 3) uint8 -> (B, 3, 64, 64) float en [0, 1]
        x = image.permute(0, 3, 1, 2).float() / 255.0
        return self.trunk(torch.cat([self.cnn(x), self.goal(goal)], dim=1))

    def get_value(self, image, goal):
        return self.critic(self.features(image, goal))

    def get_action_and_value(self, image, goal, action=None):
        h = self.features(image, goal)
        dist = Categorical(logits=self.actor(h))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(h)


# ---------------------------------------------------------------- checkpoints
def save_skill_policy(agent: SkillAgent, path: str | Path, **metadata) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": agent.state_dict(), "goal_dim": GOAL_DIM,
                "num_actions": NUM_ACTIONS, "metadata": metadata}, path)


def load_skill_policy(path: str | Path, device: str = "cpu") -> SkillAgent:
    ckpt = torch.load(path, map_location=device)
    if ckpt["goal_dim"] != GOAL_DIM:
        raise ValueError(f"El checkpoint usa goal_dim={ckpt['goal_dim']} pero el catálogo "
                         f"actual tiene {GOAL_DIM}: ¿cambió skills/catalog.py?")
    agent = SkillAgent(ckpt["goal_dim"], ckpt["num_actions"]).to(device)
    agent.load_state_dict(ckpt["state_dict"])
    agent.eval()
    return agent


class PPOSkillPolicy:
    """Adaptador: agente PPO congelado con la misma interfaz que ScriptedSkillPolicy."""

    name = "ppo"

    def __init__(self, path: str | Path, device: str | None = None, greedy: bool = False):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.greedy = greedy
        self.agent = load_skill_policy(path, device)

    def reset(self, skill: Skill) -> None:
        pass  # si la política fuera recurrente (LSTM), acá se resetea el estado

    @torch.no_grad()
    def act(self, env, obs: np.ndarray, info: dict, skill: Skill) -> int:
        image = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        goal = torch.as_tensor(encode(skill), device=self.device).unsqueeze(0)
        logits = self.agent.actor(self.agent.features(image, goal))
        if self.greedy:
            return int(logits.argmax(dim=-1).item())
        return int(Categorical(logits=logits).sample().item())
