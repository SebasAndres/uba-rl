"""FASE 1 — Entrenar el agente de habilidades con PPO (sin LLM).

ESQUELETO: el setup (entornos, red, buffers, logging, checkpoint) está hecho;
el loop de PPO está marcado con TODO. Referencia línea por línea:
    ../../../cleanrl/cleanrl/ppo_atari.py   (loop: L202-L320)
Diferencias con ppo_atari.py:
    - la observación es un Dict {"image", "goal"}: hay dos buffers en vez de uno,
    - la red es crafter_hrl.skills.policy.SkillAgent (CNN + goal_vec),
    - el checkpoint se guarda con save_skill_policy (lo lee scripts/play.py).

Uso:
    python training/fase1_habilidades/train_skills.py --total-timesteps 2000000
    tensorboard --logdir runs/
"""

import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crafter_hrl.skills.catalog import GOAL_DIM  # noqa: E402
from crafter_hrl.skills.policy import SkillAgent, save_skill_policy  # noqa: E402
from crafter_hrl.skills.skill_env import SkillEnv  # noqa: E402


@dataclass
class Args:
    exp_name: str = "skills_ppo"
    seed: int = 1
    cuda: bool = True
    save_path: str = str(ROOT / "checkpoints/skills/skill_policy.pt")
    save_every: int = 50  # iteraciones

    # entorno
    num_envs: int = 16
    async_envs: bool = True         # AsyncVectorEnv (subprocesos): Crafter es CPU-bound
    boost_prob: float = 0.3         # prob. de regalar inventario en el reset (curriculum)
    step_penalty: float = 0.01

    # PPO (punto de partida: los de ppo_atari.py; ajustar)
    total_timesteps: int = 5_000_000
    learning_rate: float = 2.5e-4
    num_steps: int = 256
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # calculados en runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(seed: int, args: Args):
    def thunk():
        return SkillEnv(seed=seed, boost_prob=args.boost_prob, step_penalty=args.step_penalty)
    return thunk


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(str(ROOT / "runs" / run_name))
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    VecEnv = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    envs = VecEnv([make_env(args.seed + i, args) for i in range(args.num_envs)])

    agent = SkillAgent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Buffers del rollout (num_steps, num_envs, ...)
    images = torch.zeros((args.num_steps, args.num_envs, 64, 64, 3), dtype=torch.uint8, device=device)
    goals = torch.zeros((args.num_steps, args.num_envs, GOAL_DIM), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # Métrica principal: tasa de éxito por habilidad (ventana móvil).
    success_by_skill: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_image = torch.as_tensor(next_obs["image"], device=device)
    next_goal = torch.as_tensor(next_obs["goal"], device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        # TODO(A) 1. Annealing del learning rate (ppo_atari.py L204-L207).

        # TODO(A) 2. Rollout de num_steps pasos (ppo_atari.py L209-L232):
        #   - guardar images[step], goals[step], dones[step]
        #   - agent.get_action_and_value(next_image, next_goal) sin gradiente
        #   - envs.step(...) ; next_done = terminated | truncated
        #   - loguear éxito por habilidad: infos["skill"], infos["success"] cuando termina
        #     (con gymnasium 0.29 vienen en infos["final_info"]):
        #         success_by_skill[skill].append(success)
        #   OJO: SkillEnv trunca por timeout de la habilidad. Tratar truncation como
        #   done es lo que hace CleanRL (simplificación); bootstrapear V(s) en
        #   truncation es una mejora posible.

        # TODO(A) 3. GAE (ppo_atari.py L234-L246) -> advantages, returns.

        # TODO(A) 4. Aplanar el batch y hacer update_epochs de PPO con minibatches
        #   (ppo_atari.py L248-L320): ratio, clip, value loss (clipeada), entropía,
        #   clip_grad_norm, early stop por target_kl. Loguear losses, approx_kl,
        #   clipfrac, explained_variance.
        raise NotImplementedError("Completar el loop de PPO (ver TODOs).")

        # Logging de la tasa de éxito por habilidad.
        for skill, hist in success_by_skill.items():
            writer.add_scalar(f"skills/success/{skill}", np.mean(hist), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if iteration % args.save_every == 0 or iteration == args.num_iterations:
            save_skill_policy(agent, args.save_path, global_step=global_step, args=vars(args))

    envs.close()
    writer.close()
