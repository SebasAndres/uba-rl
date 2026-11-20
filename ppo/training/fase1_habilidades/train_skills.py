"""FASE 1 — Entrenar el agente de habilidades con PPO (sin LLM).

PPO recurrente basado en ../../../cleanrl/cleanrl/ppo_atari_lstm.py, con:
    - observación Dict {"image", "goal", "time"} + acción anterior: un buffer por entrada,
    - red crafter_hrl.skills.policy.SkillAgent (CNN + [goal_vec | tiempo | acción anterior] -> LSTM),
    - checkpoint con save_skill_policy (lo lee scripts/play.py).

Estado de la LSTM (ver docstring de crafter_hrl/skills/policy.py):
    - `state.lstm_state` lleva (h, c) de cada entorno entre pasos del rollout,
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
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crafter_hrl.skills.catalog import GOAL_DIM  # noqa: E402
from crafter_hrl.env.constants import NUM_ACTIONS  # noqa: E402
from crafter_hrl.skills.policy import LSTMState, SkillAgent, prev_action_vec, save_skill_policy  # noqa: E402
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
    boost_prob: float = 0.3         # prob. de regalar inventario aleatorio en el reset
    step_penalty: float = 0.01
    shaping_coef: float = 0.1       # shaping por distancia BFS al objetivo (0 = sin shaping)
    use_curriculum: bool = True     # etapas + muestreo ponderado por (1 - éxito)
    start_stage: int = 0
    advance_threshold: float = 0.6  # éxito medio de lo desbloqueado para pasar de etapa
    prereq_prob: float = 0.5        # prob. de regalar los prerequisitos de la habilidad sorteada

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
        Hiperparámetros (curriculum, shaping, penalización por paso).

    Returns
    -------
    callable
        Función sin argumentos que construye un `SkillEnv`.
    """
    def thunk() -> SkillEnv:
        """Construye el entorno (se llama dentro de cada subproceso)."""
        return SkillEnv(seed=seed, boost_prob=args.boost_prob, step_penalty=args.step_penalty,
                        shaping_coef=args.shaping_coef, gamma=args.gamma,
                        use_curriculum=args.use_curriculum, start_stage=args.start_stage,
                        advance_threshold=args.advance_threshold, prereq_prob=args.prereq_prob)
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


@dataclass
class LoopState:
    """Lo que el loop lleva entre pasos (y entre rollouts) y la red no guarda."""

    image: torch.Tensor
    goal: torch.Tensor
    time: torch.Tensor
    prev_action: torch.Tensor  # one-hot de la acción anterior (ceros = "recién empiezo")
    done: torch.Tensor         # 1 donde la observación actual es la primera de una habilidad nueva
    lstm_state: LSTMState

    @classmethod
    def initial(cls, obs: dict[str, np.ndarray], agent: SkillAgent, num_envs: int,
                device: torch.device) -> "LoopState":
        """Estado al empezar: memoria vacía y acción anterior en cero.

        Parameters
        ----------
        obs : dict
            Observación de `envs.reset`.
        agent : SkillAgent
            Red (para el estado inicial de la LSTM).
        num_envs : int
            Entornos en paralelo.
        device : torch.device
            Dispositivo.

        Returns
        -------
        LoopState
            Estado inicial.
        """
        image, goal, time_left = obs_to_tensors(obs, device)
        return cls(image, goal, time_left, torch.zeros(num_envs, NUM_ACTIONS, device=device),
                   torch.zeros(num_envs, device=device), agent.initial_state(num_envs, device))


def obs_to_tensors(obs: dict[str, np.ndarray], device: torch.device
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convierte la observación vectorizada en tensores.

    Parameters
    ----------
    obs : dict
        {"image": (N, 64, 64, 3) uint8, "goal": (N, GOAL_DIM), "time": (N, 1)}.
    device : torch.device
        Dispositivo.

    Returns
    -------
    tuple of torch.Tensor
        (image uint8, goal float, time float (N, 1)).
    """
    return (torch.as_tensor(obs["image"], device=device),
            torch.as_tensor(obs["goal"], dtype=torch.float32, device=device),
            torch.as_tensor(obs["time"], dtype=torch.float32, device=device).reshape(-1, 1))


def per_env_infos(infos: dict, num_envs: int) -> list[dict]:
    """Separa los `infos` vectorizados (gymnasium 0.29) en un dict por entorno.

    Para los entornos que terminaron devuelve `final_info` (el info del último
    paso de la habilidad); para el resto, las claves presentes en ese entorno.

    Parameters
    ----------
    infos : dict
        Infos de `envs.step`.
    num_envs : int
        Entornos en paralelo.

    Returns
    -------
    list of dict
        Un info por entorno.
    """
    final_mask = infos.get("_final_info", np.zeros(num_envs, dtype=bool))
    out = []
    for i in range(num_envs):
        if final_mask[i]:
            out.append(infos["final_info"][i])
            continue
        out.append({k: v[i] for k, v in infos.items()
                    if not k.startswith("_") and k not in ("final_info", "final_observation")
                    and infos.get(f"_{k}", np.ones(num_envs, dtype=bool))[i]})
    return out


def record_infos(env_infos: list[dict], done: np.ndarray, success_by_skill: dict[str, deque],
                 stats: dict[str, list]) -> None:
    """Acumula éxito por habilidad (al terminar) y shaping/etapa/muertes del rollout.

    Parameters
    ----------
    env_infos : list of dict
        Un info por entorno (ver `per_env_infos`).
    done : np.ndarray
        (N,) bool, entornos que terminaron en este paso.
    success_by_skill : dict of str to deque
        Ventana móvil de éxitos por habilidad (se modifica in-place).
    stats : dict of str to list
        Acumuladores del rollout (se modifica in-place).
    """
    for info, finished in zip(env_infos, done):
        if "shaping" in info:
            stats["shaping"].append(float(info["shaping"]))
        if "stage" in info:
            stats["stage"].append(float(info["stage"]))
        if finished and "success" in info:
            success_by_skill[str(info["skill"])].append(float(info["success"]))
            stats["success"].append(float(info["success"]))
            stats["died"].append(float(info["died"]))


def rollout_step(envs: gym.vector.VectorEnv, agent: SkillAgent, buffers: RolloutBuffers,
                 state: LoopState, step: int, device: torch.device
                 ) -> tuple[LoopState, list[dict], np.ndarray]:
    """Un paso del rollout: guarda la entrada, actúa sin gradiente y avanza los entornos.

    Parameters
    ----------
    envs : gym.vector.VectorEnv
        Entornos vectorizados.
    agent : SkillAgent
        Red.
    buffers : RolloutBuffers
        Buffers del rollout (se escriben en `step`).
    state : LoopState
        Estado actual del loop.
    step : int
        Índice del paso dentro del rollout.
    device : torch.device
        Dispositivo.

    Returns
    -------
    state : LoopState
        Estado siguiente.
    env_infos : list of dict
        Info de cada entorno (ver `per_env_infos`).
    done : np.ndarray
        (N,) bool, entornos que terminaron.
    """
    buffers.images[step], buffers.goals[step], buffers.times[step] = state.image, state.goal, state.time
    buffers.prev_actions[step], buffers.dones[step] = state.prev_action, state.done
    with torch.no_grad():
        action, logprob, _, value, lstm_state = agent.get_action_and_value(
            state.image, state.goal, state.time, state.prev_action, state.lstm_state, state.done)
    buffers.values[step], buffers.actions[step], buffers.logprobs[step] = value.flatten(), action, logprob
    obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
    done_np = np.logical_or(terminated, truncated)  # timeout = parte de la tarea (ver docstring de train)
    buffers.rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
    done = torch.as_tensor(done_np, dtype=torch.float32, device=device)
    image, goal, time_left = obs_to_tensors(obs, device)
    next_state = LoopState(image, goal, time_left, prev_action_vec(action, done), done, lstm_state)
    return next_state, per_env_infos(infos, len(done_np)), done_np


def collect_rollout(envs: gym.vector.VectorEnv, agent: SkillAgent, buffers: RolloutBuffers,
                    state: LoopState, args: Args, device: torch.device,
                    success_by_skill: dict[str, deque]) -> tuple[LoopState, dict[str, list]]:
    """Juega `num_steps` pasos en todos los entornos y llena los buffers.

    Parameters
    ----------
    envs : gym.vector.VectorEnv
        Entornos vectorizados.
    agent : SkillAgent
        Red.
    buffers : RolloutBuffers
        Buffers a llenar.
    state : LoopState
        Estado al empezar el rollout.
    args : Args
        Hiperparámetros.
    device : torch.device
        Dispositivo.
    success_by_skill : dict of str to deque
        Ventana móvil de éxitos por habilidad (se modifica in-place).

    Returns
    -------
    state : LoopState
        Estado al terminar el rollout.
    stats : dict of str to list
        Shaping, etapa, éxitos y muertes vistos en el rollout.
    """
    stats: dict[str, list] = defaultdict(list)
    for step in range(args.num_steps):
        state, env_infos, done = rollout_step(envs, agent, buffers, state, step, device)
        record_infos(env_infos, done, success_by_skill, stats)
    return state, stats


def compute_gae(agent: SkillAgent, buffers: RolloutBuffers, state: LoopState,
                args: Args) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcula ventajas (GAE) y retornos, bootstrapeando con el valor del último estado.

    Parameters
    ----------
    agent : SkillAgent
        Red.
    buffers : RolloutBuffers
        Buffers del rollout.
    state : LoopState
        Estado al terminar el rollout.
    args : Args
        Hiperparámetros (gamma, gae_lambda, num_steps).

    Returns
    -------
    advantages, returns : torch.Tensor
        Cada uno (num_steps, num_envs).
    """
    with torch.no_grad():
        next_value = agent.get_value(state.image, state.goal, state.time, state.prev_action,
                                     state.lstm_state, state.done).reshape(-1)
    advantages = torch.zeros_like(buffers.rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            next_nonterminal, next_values = 1.0 - state.done, next_value
        else:
            next_nonterminal, next_values = 1.0 - buffers.dones[t + 1], buffers.values[t + 1]
        delta = buffers.rewards[t] + args.gamma * next_values * next_nonterminal - buffers.values[t]
        last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + buffers.values


def flatten_batch(buffers: RolloutBuffers, advantages: torch.Tensor,
                  returns: torch.Tensor) -> dict[str, torch.Tensor]:
    """Aplana (num_steps, num_envs, ...) -> (num_steps * num_envs, ...), orden paso a paso.

    Parameters
    ----------
    buffers : RolloutBuffers
        Buffers del rollout.
    advantages : torch.Tensor
        Ventajas (num_steps, num_envs).
    returns : torch.Tensor
        Retornos (num_steps, num_envs).

    Returns
    -------
    dict of str to torch.Tensor
        Un tensor plano por buffer, más "advantages" y "returns".
    """
    batch = {}
    for field in fields(buffers):
        tensor = getattr(buffers, field.name)
        batch[field.name] = tensor.reshape(-1, *tensor.shape[2:])
    batch["advantages"], batch["returns"] = advantages.reshape(-1), returns.reshape(-1)
    return batch


def value_loss(new_value: torch.Tensor, returns: torch.Tensor, old_value: torch.Tensor,
               args: Args) -> torch.Tensor:
    """Pérdida del crítico (clipeada alrededor del valor viejo si `clip_vloss`).

    Parameters
    ----------
    new_value : torch.Tensor
        Valores re-evaluados (M,).
    returns : torch.Tensor
        Retornos objetivo (M,).
    old_value : torch.Tensor
        Valores del rollout (M,).
    args : Args
        Hiperparámetros (clip_vloss, clip_coef).

    Returns
    -------
    torch.Tensor
        Escalar.
    """
    unclipped = (new_value - returns) ** 2
    if not args.clip_vloss:
        return 0.5 * unclipped.mean()
    clipped_value = old_value + torch.clamp(new_value - old_value, -args.clip_coef, args.clip_coef)
    return 0.5 * torch.max(unclipped, (clipped_value - returns) ** 2).mean()


def minibatch_loss(agent: SkillAgent, batch: dict[str, torch.Tensor], mb_inds: torch.Tensor,
                   mb_envs: torch.Tensor, initial_lstm_state: LSTMState,
                   args: Args) -> tuple[torch.Tensor, dict[str, float]]:
    """Re-evalúa las secuencias de un minibatch de entornos y arma la pérdida de PPO.

    Parameters
    ----------
    agent : SkillAgent
        Red.
    batch : dict of str to torch.Tensor
        Batch aplanado (ver `flatten_batch`).
    mb_inds : torch.Tensor
        Índices planos de TODOS los pasos de esos entornos, en orden paso a paso.
    mb_envs : torch.Tensor
        Entornos del minibatch.
    initial_lstm_state : LSTMState
        Estado de la LSTM al empezar el rollout.
    args : Args
        Hiperparámetros.

    Returns
    -------
    loss : torch.Tensor
        Pérdida total.
    stats : dict of str to float
        policy_loss, value_loss, entropy, approx_kl, clipfrac.
    """
    _, new_logprob, entropy, new_value, _ = agent.get_action_and_value(
        batch["images"][mb_inds], batch["goals"][mb_inds], batch["times"][mb_inds],
        batch["prev_actions"][mb_inds],
        (initial_lstm_state[0][:, mb_envs], initial_lstm_state[1][:, mb_envs]),
        batch["dones"][mb_inds], batch["actions"].long()[mb_inds])
    log_ratio = new_logprob - batch["logprobs"][mb_inds]
    ratio = log_ratio.exp()
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean().item()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
    adv = batch["advantages"][mb_inds]
    if args.norm_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
    v_loss = value_loss(new_value.view(-1), batch["returns"][mb_inds], batch["values"][mb_inds], args)
    entropy_mean = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_mean + args.vf_coef * v_loss
    return loss, {"policy_loss": pg_loss.item(), "value_loss": v_loss.item(),
                  "entropy": entropy_mean.item(), "approx_kl": approx_kl, "clipfrac": clipfrac}


def ppo_update(agent: SkillAgent, optimizer: optim.Optimizer, batch: dict[str, torch.Tensor],
               initial_lstm_state: LSTMState, args: Args) -> dict[str, float]:
    """Update de PPO por minibatches de ENTORNOS completos (la LSTM necesita secuencias enteras).

    Parameters
    ----------
    agent : SkillAgent
        Red.
    optimizer : optim.Optimizer
        Optimizador.
    batch : dict of str to torch.Tensor
        Batch aplanado (ver `flatten_batch`).
    initial_lstm_state : LSTMState
        Estado de la LSTM al empezar el rollout.
    args : Args
        Hiperparámetros.

    Returns
    -------
    dict of str to float
        Promedio de las estadísticas de cada minibatch.
    """
    device = batch["rewards"].device
    envs_per_batch = args.num_envs // args.num_minibatches
    flat_inds = torch.arange(args.batch_size, device=device).reshape(args.num_steps, args.num_envs)
    history: list[dict[str, float]] = []
    for _ in range(args.update_epochs):
        env_order = torch.randperm(args.num_envs, device=device)
        for start in range(0, args.num_envs, envs_per_batch):
            mb_envs = env_order[start:start + envs_per_batch]
            loss, stats = minibatch_loss(agent, batch, flat_inds[:, mb_envs].reshape(-1),
                                         mb_envs, initial_lstm_state, args)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            history.append(stats)
        if args.target_kl is not None and history[-1]["approx_kl"] > args.target_kl:
            break
    return {key: float(np.mean([s[key] for s in history])) for key in history[0]}


def anneal_lr(optimizer: optim.Optimizer, iteration: int, args: Args) -> None:
    """Baja el learning rate linealmente hasta 0 (si `anneal_lr`).

    Parameters
    ----------
    optimizer : optim.Optimizer
        Optimizador.
    iteration : int
        Iteración actual (empieza en 1).
    args : Args
        Hiperparámetros.
    """
    if args.anneal_lr:
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        optimizer.param_groups[0]["lr"] = frac * args.learning_rate


def log_update(writer: SummaryWriter, update_stats: dict[str, float], rollout_stats: dict[str, list],
               batch: dict[str, torch.Tensor], lr: float, global_step: int) -> None:
    """Loguea pérdidas, KL, clipfrac, varianza explicada y estadísticas del rollout.

    Parameters
    ----------
    writer : SummaryWriter
        Writer de tensorboard.
    update_stats : dict of str to float
        Salida de `ppo_update`.
    rollout_stats : dict of str to list
        Salida de `collect_rollout`.
    batch : dict of str to torch.Tensor
        Batch aplanado (para varianza explicada y recompensa media).
    lr : float
        Learning rate actual.
    global_step : int
        Pasos de entorno acumulados.
    """
    writer.add_scalar("charts/learning_rate", lr, global_step)
    for key, value in update_stats.items():
        writer.add_scalar(f"losses/{key}", value, global_step)
    y_pred, y_true = batch["values"].cpu().numpy(), batch["returns"].cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    writer.add_scalar("rollout/reward_mean", batch["rewards"].mean().item(), global_step)
    for key, values in rollout_stats.items():  # shaping, stage, success, died
        if values:
            writer.add_scalar(f"rollout/{key}", np.mean(values), global_step)


def train(args: Args) -> None:
    """Entrena el agente de habilidades con PPO recurrente.

    Timeout: como la red ve la fracción de tiempo restante ("time"), el timeout
    es parte de la tarea y se trata como done (sin bootstrap), como en CleanRL.

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
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    buffers = RolloutBuffers.allocate(args.num_steps, args.num_envs, device)
    success_by_skill: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))  # métrica principal
    state = LoopState.initial(envs.reset(seed=args.seed)[0], agent, args.num_envs, device)
    global_step, start_time = 0, time.time()
    for iteration in range(1, args.num_iterations + 1):
        initial_lstm_state = (state.lstm_state[0].clone(), state.lstm_state[1].clone())
        anneal_lr(optimizer, iteration, args)
        state, rollout_stats = collect_rollout(envs, agent, buffers, state, args, device, success_by_skill)
        global_step += args.batch_size
        batch = flatten_batch(buffers, *compute_gae(agent, buffers, state, args))
        update_stats = ppo_update(agent, optimizer, batch, initial_lstm_state, args)
        log_update(writer, update_stats, rollout_stats, batch, optimizer.param_groups[0]["lr"], global_step)
        log_iteration(writer, success_by_skill, global_step, start_time)
        if iteration % args.save_every == 0 or iteration == args.num_iterations:
            save_skill_policy(agent, args.save_path, global_step=global_step, args=vars(args))
    envs.close()
    writer.close()


if __name__ == "__main__":
    train(finalize_args(tyro.cli(Args)))
