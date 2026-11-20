"""FASE 1 — Entrenar el agente de habilidades con PPO (sin LLM).

ESQUELETO: el setup (entornos, red, buffers, logging, checkpoint) está hecho;
el loop de PPO está marcado con TODO. Referencia (la red es recurrente):
    ../../../cleanrl/cleanrl/ppo_atari_lstm.py
Diferencias con ppo_atari_lstm.py:
    - la observación es un Dict {"image", "goal", "time"} y además la red recibe
      la acción anterior: hay un buffer por cada entrada en vez de uno,
    - la red es crafter_hrl.skills.policy.SkillAgent (CNN + [goal_vec | tiempo |
      acción anterior] -> LSTM); `encode` + `get_states` reemplazan a `get_states`
      de CleanRL, y `get_action_and_value` también devuelve el estado nuevo,
    - el checkpoint se guarda con save_skill_policy (lo lee scripts/play.py).

Estado de la LSTM (ver docstring de crafter_hrl/skills/policy.py):
    - `next_lstm_state` lleva (h, c) de cada entorno entre pasos del rollout,
    - `initial_lstm_state` es la copia al empezar el rollout: el update vuelve a
      pasar la secuencia entera desde ahí, con los mismos reinicios (dones),
    - por eso los minibatches se arman con ENTORNOS completos, no pasos sueltos.

Uso:
    python training/fase1_habilidades/train_skills.py --total-timesteps 2000000
    tensorboard --logdir runs/
"""

import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crafter_hrl.skills.catalog import GOAL_DIM  # noqa: E402
from crafter_hrl.env.constants import NUM_ACTIONS  # noqa: E402
from crafter_hrl.skills.policy import SkillAgent, prev_action_vec, save_skill_policy  # noqa: E402,F401
from crafter_hrl.skills.skill_env import SkillEnv  # noqa: E402


@dataclass
class Args:
    """Hiperparámetros de la fase 1 (se leen de la línea de comandos con tyro)."""

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
    num_minibatches: int = 4  # con LSTM: minibatch = num_envs / num_minibatches entornos completos
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


@dataclass
class RolloutBuffers:
    """Buffers del rollout, todos con forma (num_steps, num_envs, ...)."""

    images: torch.Tensor
    goals: torch.Tensor
    times: torch.Tensor
    prev_actions: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor

    @classmethod
    def allocate(cls, num_steps: int, num_envs: int, device: torch.device) -> "RolloutBuffers":
        """Reserva los buffers en `device`.

        Parameters
        ----------
        num_steps : int
            Pasos por rollout.
        num_envs : int
            Entornos en paralelo.
        device : torch.device
            Dispositivo.

        Returns
        -------
        RolloutBuffers
            Buffers en cero.
        """
        shape = (num_steps, num_envs)
        return cls(
            images=torch.zeros((*shape, 64, 64, 3), dtype=torch.uint8, device=device),
            goals=torch.zeros((*shape, GOAL_DIM), device=device),
            times=torch.zeros((*shape, 1), device=device),
            prev_actions=torch.zeros((*shape, NUM_ACTIONS), device=device),
            actions=torch.zeros(shape, device=device),
            logprobs=torch.zeros(shape, device=device),
            rewards=torch.zeros(shape, device=device),
            dones=torch.zeros(shape, device=device),
            values=torch.zeros(shape, device=device),
        )


def make_env(seed: int, args: Args) -> Callable[[], SkillEnv]:
    """Fábrica de entornos para `gym.vector`.

    Parameters
    ----------
    seed : int
        Semilla del entorno.
    args : Args
        Hiperparámetros (curriculum, penalización por paso).

    Returns
    -------
    callable
        Función sin argumentos que construye un `SkillEnv`.
    """
    def thunk() -> SkillEnv:
        """Construye el entorno (se llama dentro de cada subproceso)."""
        return SkillEnv(seed=seed, boost_prob=args.boost_prob, step_penalty=args.step_penalty)
    return thunk


def finalize_args(args: Args) -> Args:
    """Calcula los tamaños derivados (batch, minibatch, iteraciones).

    Parameters
    ----------
    args : Args
        Hiperparámetros leídos de la línea de comandos.

    Returns
    -------
    Args
        Los mismos args, con los campos derivados completos.

    Raises
    ------
    ValueError
        Si los entornos no se pueden repartir en minibatches de entornos completos.
    """
    if args.num_envs % args.num_minibatches != 0:
        raise ValueError("num_envs tiene que ser múltiplo de num_minibatches (minibatches por entorno).")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    return args


def make_writer(args: Args) -> SummaryWriter:
    """Crea el writer de tensorboard y loguea los hiperparámetros.

    Parameters
    ----------
    args : Args
        Hiperparámetros.

    Returns
    -------
    SummaryWriter
        Writer en runs/<exp_name>__<seed>__<timestamp>.
    """
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(str(ROOT / "runs" / run_name))
    table = "\n".join(f"|{k}|{v}|" for k, v in vars(args).items())
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + table)
    return writer


def log_iteration(writer: SummaryWriter, success_by_skill: dict[str, deque],
                  global_step: int, start_time: float) -> None:
    """Loguea la tasa de éxito por habilidad y los pasos por segundo.

    Parameters
    ----------
    writer : SummaryWriter
        Writer de tensorboard.
    success_by_skill : dict of str to deque
        Ventana móvil de éxitos por habilidad.
    global_step : int
        Pasos de entorno acumulados.
    start_time : float
        Momento de inicio del entrenamiento.
    """
    for skill, history in success_by_skill.items():
        writer.add_scalar(f"skills/success/{skill}", np.mean(history), global_step)
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


def train(args: Args) -> None:
    """Entrena el agente de habilidades con PPO (el loop está marcado con TODOs).

    Parameters
    ----------
    args : Args
        Hiperparámetros (ya con `finalize_args`).
    """
    writer = make_writer(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    VecEnv = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    envs = VecEnv([make_env(args.seed + i, args) for i in range(args.num_envs)])
    agent = SkillAgent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)  # noqa: F841
    buffers = RolloutBuffers.allocate(args.num_steps, args.num_envs, device)  # noqa: F841
    success_by_skill: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))  # métrica principal

    global_step, start_time = 0, time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_image = torch.as_tensor(next_obs["image"], device=device)  # noqa: F841
    next_goal = torch.as_tensor(next_obs["goal"], device=device)  # noqa: F841
    next_time = torch.as_tensor(next_obs["time"], device=device)  # noqa: F841
    next_done = torch.zeros(args.num_envs, device=device)  # noqa: F841
    # Lo que la red NO guarda y lleva el loop: acción anterior (ceros = "recién empiezo") y estado LSTM.
    next_prev_action = torch.zeros(args.num_envs, NUM_ACTIONS, device=device)  # noqa: F841
    next_lstm_state = agent.initial_state(args.num_envs, device)  # noqa: F841

    for iteration in range(1, args.num_iterations + 1):
        # TODO(A) 0. initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        # TODO(A) 1. Annealing del learning rate (como ppo_atari_lstm.py).

        # TODO(A) 2. Rollout de num_steps pasos (como ppo_atari_lstm.py):
        #   - guardar buffers.images/goals/times/prev_actions/dones[step] (next_*)
        #   - sin gradiente:
        #       action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
        #           next_image, next_goal, next_time, next_prev_action, next_lstm_state, next_done)
        #   - envs.step(...) ; next_done = terminated | truncated
        #   - next_prev_action = prev_action_vec(action, next_done)   # ceros si empieza otra habilidad
        #   - loguear éxito por habilidad: infos["skill"], infos["success"] cuando termina
        #     (con gymnasium 0.29 vienen en infos["final_info"]):
        #         success_by_skill[skill].append(success)
        #   Timeout: como la red ve la fracción de tiempo restante ("time"), el
        #   timeout es parte de la tarea (también corta la habilidad en el juego)
        #   y tratarlo como done, como hace CleanRL, es correcto: no hace falta
        #   bootstrapear. La única truncation "externa" es el fin del mundo de
        #   Crafter (world_length=10000 pasos), que es rara.

        # TODO(A) 3. GAE -> advantages, returns. Valor del último estado:
        #   agent.get_value(next_image, next_goal, next_time, next_prev_action, next_lstm_state, next_done)

        # TODO(A) 4. Update PPO por minibatches de ENTORNOS completos (como ppo_atari_lstm.py):
        #   envs_per_batch = num_envs // num_minibatches ; mezclar los índices de entorno,
        #   tomar TODOS los pasos de esos entornos (orden paso a paso, ver get_states) y re-evaluar:
        #       agent.get_action_and_value(b_images[mb], b_goals[mb], b_times[mb], b_prev_actions[mb],
        #           (initial_lstm_state[0][:, mb_envs], initial_lstm_state[1][:, mb_envs]),
        #           b_dones[mb], b_actions.long()[mb])
        #   Después: ratio, clip, value loss (clipeada), entropía, clip_grad_norm,
        #   early stop por target_kl. Loguear losses, approx_kl, clipfrac, explained_variance.
        raise NotImplementedError("Completar el loop de PPO (ver TODOs).")

        log_iteration(writer, success_by_skill, global_step, start_time)
        if iteration % args.save_every == 0 or iteration == args.num_iterations:
            save_skill_policy(agent, args.save_path, global_step=global_step, args=vars(args))

    envs.close()
    writer.close()


if __name__ == "__main__":
    train(finalize_args(tyro.cli(Args)))
