"""Agente de habilidades PPO (fase 1): red + formato de checkpoint.

La MISMA clase la usan el script de entrenamiento (training/fase1_habilidades)
y el script de juego (scripts/play.py). Si se cambia la arquitectura, hay que
subir `ARCH_VERSION` y mantener `save_skill_policy` / `load_skill_policy` compatibles.

Arquitectura (como cleanrl/ppo_atari_lstm.py, condicionada a la habilidad):

    imagen 64x64x3 ──► CNN Nature ─────────────────────────────┐
    [goal_vec | fracción de timeout | acción anterior] ──► MLP ─┴► Linear(512) ► LSTM(128) ─┬► actor (17)
                                                                                            └► crítico (1)

  - Fracción de timeout restante: hace observable el límite de la habilidad
    (sin ella el mismo frame vale distinto en el paso 5 que en el 95;
    Pardo et al. 2018, "Time Limits in RL").
  - Acción anterior (one-hot de 17): la LSTM ve los frames pero no qué hizo el
    agente entre uno y otro. En el primer paso de cada habilidad es un vector
    de CEROS (no la acción 0, que es `noop`): así marca "recién empiezo".
  - LSTM: memoria dentro de la habilidad (por dónde pasó, qué salió de la vista).
    El estado (h, c) NO lo guarda la red: entra y sale de `get_states` y lo
    lleva quien llama (el loop de entrenamiento o `PPOSkillPolicy`). Se
    reinicia a cero al empezar cada habilidad multiplicando por (1 - done).
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..env.constants import NUM_ACTIONS
from .catalog import GOAL_DIM, Skill, encode, time_left

CNN_OUT_DIM = 64 * 4 * 4
GOAL_EMBED_DIM = 64
TIME_DIM = 1        # fracción de timeout restante
LSTM_DIM = 128
# 1: CNN + goal_vec | 2: + fracción de tiempo | 3: + acción anterior + LSTM
ARCH_VERSION = 3

LSTMState = tuple[torch.Tensor, torch.Tensor]  # (h, c), cada uno (1, B, LSTM_DIM)


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Inicialización ortogonal de CleanRL.

    Parameters
    ----------
    layer : nn.Module
        Capa lineal o convolucional.
    std : float, optional
        Ganancia de la inicialización ortogonal.
    bias_const : float, optional
        Valor constante del bias.

    Returns
    -------
    nn.Module
        La misma capa, inicializada in-place.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def prev_action_vec(action: torch.Tensor, done: torch.Tensor, num_actions: int = NUM_ACTIONS) -> torch.Tensor:
    """Codifica la acción recién tomada como entrada "acción anterior" del próximo paso.

    Parameters
    ----------
    action : torch.Tensor
        Acciones tomadas (B,).
    done : torch.Tensor
        (B,) con 1 donde el episodio terminó con esa acción: el próximo paso
        es el primero de una habilidad nueva y su acción anterior es ceros.
    num_actions : int, optional
        Largo del one-hot.

    Returns
    -------
    torch.Tensor
        One-hot (B, num_actions) float, con filas en cero donde `done` es 1.
    """
    one_hot = nn.functional.one_hot(action.long(), num_actions).float()
    return one_hot * (1.0 - done.float()).unsqueeze(1)


class SkillAgent(nn.Module):
    """Red actor-crítico recurrente condicionada a la habilidad.

    Parameters
    ----------
    goal_dim : int, optional
        Largo del goal_vec.
    num_actions : int, optional
        Cantidad de acciones primitivas (logits del actor y largo de la acción anterior).
    hidden : int, optional
        Tamaño de la capa previa a la LSTM.
    lstm_dim : int, optional
        Tamaño del estado de la LSTM.
    """

    def __init__(self, goal_dim: int = GOAL_DIM, num_actions: int = NUM_ACTIONS,
                 hidden: int = 512, lstm_dim: int = LSTM_DIM) -> None:
        super().__init__()
        self.num_actions, self.lstm_dim = num_actions, lstm_dim
        self.cnn = nn.Sequential(                                    # 3x64x64
            layer_init(nn.Conv2d(3, 32, 8, stride=4)), nn.ReLU(),     # 32x15x15
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),    # 64x6x6
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),    # 64x4x4
            nn.Flatten(),
        )
        context_dim = goal_dim + TIME_DIM + num_actions
        self.context = nn.Sequential(layer_init(nn.Linear(context_dim, GOAL_EMBED_DIM)), nn.ReLU())
        self.trunk = nn.Sequential(layer_init(nn.Linear(CNN_OUT_DIM + GOAL_EMBED_DIM, hidden)), nn.ReLU())
        self.lstm = nn.LSTM(hidden, lstm_dim)
        for name, param in self.lstm.named_parameters():  # init de ppo_atari_lstm.py
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(lstm_dim, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(lstm_dim, 1), std=1.0)

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> LSTMState:
        """Estado (h, c) en cero: memoria vacía.

        Parameters
        ----------
        batch_size : int
            Cantidad de secuencias en paralelo (entornos).
        device : torch.device or str, optional
            Dispositivo.

        Returns
        -------
        LSTMState
            (h, c), cada uno (1, batch_size, lstm_dim).
        """
        zeros = torch.zeros(1, batch_size, self.lstm_dim, device=device)
        return zeros, zeros.clone()

    def encode(self, image: torch.Tensor, goal: torch.Tensor, time: torch.Tensor,
               prev_action: torch.Tensor) -> torch.Tensor:
        """Features de cada paso, antes de la LSTM (no dependen del estado).

        Parameters
        ----------
        image : torch.Tensor
            Imágenes (N, 64, 64, 3) uint8.
        goal : torch.Tensor
            goal_vecs (N, goal_dim).
        time : torch.Tensor
            Fracción de timeout restante (N, 1), en [0, 1].
        prev_action : torch.Tensor
            Acción anterior en one-hot (N, num_actions); ceros en el primer paso.

        Returns
        -------
        torch.Tensor
            Features (N, hidden).
        """
        pixels = image.permute(0, 3, 1, 2).float() / 255.0  # -> (N, 3, 64, 64) en [0, 1]
        context = self.context(torch.cat([goal, time, prev_action], dim=1))
        return self.trunk(torch.cat([self.cnn(pixels), context], dim=1))

    def get_states(self, x: torch.Tensor, state: LSTMState, done: torch.Tensor) -> tuple[torch.Tensor, LSTMState]:
        """Pasa una secuencia de features por la LSTM, reiniciando el estado donde `done`.

        Parameters
        ----------
        x : torch.Tensor
            Features de `encode`, (T * B, hidden) ordenadas paso a paso
            (los B entornos del paso 0, después los del paso 1, ...).
        state : LSTMState
            Estado (h, c) antes del primer paso, cada uno (1, B, lstm_dim).
        done : torch.Tensor
            (T * B,) con 1 donde ese paso es el PRIMERO de una habilidad nueva:
            ahí el estado se pone en cero antes de usarlo.

        Returns
        -------
        outputs : torch.Tensor
            Salidas de la LSTM (T * B, lstm_dim), en el mismo orden que `x`.
        state : LSTMState
            Estado después del último paso (para seguir en la próxima llamada).
        """
        batch_size = state[0].shape[1]
        x = x.reshape(-1, batch_size, x.shape[-1])
        done = done.reshape(-1, batch_size)
        outputs = []
        for x_t, done_t in zip(x, done):
            keep = (1.0 - done_t).view(1, -1, 1)
            out, state = self.lstm(x_t.unsqueeze(0), (keep * state[0], keep * state[1]))
            outputs.append(out)
        return torch.flatten(torch.cat(outputs), 0, 1), state

    def get_value(self, image: torch.Tensor, goal: torch.Tensor, time: torch.Tensor,
                  prev_action: torch.Tensor, state: LSTMState, done: torch.Tensor) -> torch.Tensor:
        """Valor V del crítico (ver `encode` y `get_states` para los argumentos).

        Returns
        -------
        torch.Tensor
            Valores (T * B, 1).
        """
        hidden, _ = self.get_states(self.encode(image, goal, time, prev_action), state, done)
        return self.critic(hidden)

    def get_action_and_value(self, image: torch.Tensor, goal: torch.Tensor, time: torch.Tensor,
                             prev_action: torch.Tensor, state: LSTMState, done: torch.Tensor,
                             action: torch.Tensor | None = None
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, LSTMState]:
        """Muestrea (o evalúa) acciones y calcula log-prob, entropía, valor y el estado nuevo.

        Sirve para un paso del rollout (T = 1) y para re-evaluar una secuencia
        entera en el update (T = num_steps, desde el estado inicial guardado).
        Ver `encode` y `get_states` para los argumentos de entrada.

        Parameters
        ----------
        action : torch.Tensor or None, optional
            Acciones a evaluar (T * B,); si es None se muestrean.

        Returns
        -------
        action : torch.Tensor
            Acciones (T * B,).
        log_prob : torch.Tensor
            Log-probabilidad de cada acción (T * B,).
        entropy : torch.Tensor
            Entropía de la distribución (T * B,).
        value : torch.Tensor
            Valores del crítico (T * B, 1).
        state : LSTMState
            Estado después del último paso.
        """
        hidden, state = self.get_states(self.encode(image, goal, time, prev_action), state, done)
        dist = Categorical(logits=self.actor(hidden))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden), state


# ---------------------------------------------------------------- checkpoints
def save_skill_policy(agent: SkillAgent, path: str | Path, **metadata: Any) -> None:
    """Guarda pesos + dimensiones + metadata en un checkpoint.

    Parameters
    ----------
    agent : SkillAgent
        Red a guardar.
    path : str or Path
        Archivo destino (se crean las carpetas que falten).
    **metadata : Any
        Datos extra (ej. global_step, args, dummy=True).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": agent.state_dict(), "goal_dim": GOAL_DIM, "num_actions": NUM_ACTIONS,
                "lstm_dim": agent.lstm_dim, "arch_version": ARCH_VERSION, "metadata": metadata}, path)


def load_skill_policy(path: str | Path, device: str = "cpu") -> SkillAgent:
    """Carga un checkpoint guardado con `save_skill_policy`.

    Parameters
    ----------
    path : str or Path
        Archivo del checkpoint.
    device : str, optional
        Dispositivo donde cargar la red.

    Returns
    -------
    SkillAgent
        Red en modo eval.

    Raises
    ------
    ValueError
        Si el checkpoint se entrenó con otro catálogo de habilidades (otro GOAL_DIM)
        o con otra versión de la arquitectura.
    """
    ckpt = torch.load(path, map_location=device)
    if ckpt["goal_dim"] != GOAL_DIM:
        raise ValueError(f"El checkpoint usa goal_dim={ckpt['goal_dim']} pero el catálogo "
                         f"actual tiene {GOAL_DIM}: ¿cambió skills/catalog.py?")
    version = ckpt.get("arch_version", 2 if "time_dim" in ckpt else 1)
    if version != ARCH_VERSION:
        raise ValueError(f"El checkpoint es de la arquitectura v{version} y la actual es v{ARCH_VERSION} "
                         "(ver policy.py): reentrenar, o regenerar el dummy con "
                         "`python scripts/crear_dummies.py --force`.")
    agent = SkillAgent(ckpt["goal_dim"], ckpt["num_actions"], lstm_dim=ckpt["lstm_dim"]).to(device)
    agent.load_state_dict(ckpt["state_dict"])
    agent.eval()
    return agent


class PPOSkillPolicy:
    """Adaptador: agente PPO congelado con la misma interfaz que `ScriptedSkillPolicy`.

    Lleva entre llamadas a `act` lo que la red no guarda: el estado de la LSTM,
    la acción anterior y los pasos dados (para la fracción de tiempo). Todo se
    reinicia en `reset`, que `run_skill` llama al empezar cada habilidad.

    Parameters
    ----------
    path : str or Path
        Checkpoint del agente.
    device : str or None, optional
        Dispositivo; por defecto cuda si está disponible.
    greedy : bool, optional
        Si es True toma la acción más probable en vez de muestrear.
    """

    name = "ppo"

    def __init__(self, path: str | Path, device: str | None = None, greedy: bool = False) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.greedy = greedy
        self.agent = load_skill_policy(path, self.device)
        self.reset(None)

    def reset(self, skill: Skill | None) -> None:
        """Memoria vacía: estado de la LSTM en cero, acción anterior en cero y paso 0.

        Parameters
        ----------
        skill : Skill or None
            Habilidad que se va a ejecutar (no se usa).
        """
        self._state = self.agent.initial_state(1, self.device)
        self._prev_action = torch.zeros(1, self.agent.num_actions, device=self.device)
        self._step = 0

    @torch.no_grad()
    def act(self, env: Any, obs: np.ndarray, info: dict, skill: Skill) -> int:
        """Elige una acción primitiva (un paso de la LSTM) y actualiza la memoria.

        El tiempo se calcula igual que en `SkillEnv`: `time_left(skill, pasos ya tomados)`.

        Parameters
        ----------
        env : CrafterEnv
            Entorno (no se usa: el agente sólo ve píxeles).
        obs : np.ndarray
            Imagen RGB (64, 64, 3).
        info : dict
            `info` de Crafter (no se usa).
        skill : Skill
            Habilidad pedida por el LLM; se codifica como goal_vec.

        Returns
        -------
        int
            Índice de la acción primitiva.
        """
        image = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        goal = torch.as_tensor(encode(skill), device=self.device).unsqueeze(0)
        time = torch.as_tensor(time_left(skill, self._step), device=self.device).unsqueeze(0)
        x = self.agent.encode(image, goal, time, self._prev_action)
        no_reset = torch.zeros(1, device=self.device)  # el reinicio ya lo hizo `reset`
        hidden, self._state = self.agent.get_states(x, self._state, no_reset)
        logits = self.agent.actor(hidden)
        action = logits.argmax(dim=-1) if self.greedy else Categorical(logits=logits).sample()
        self._prev_action = prev_action_vec(action, no_reset)
        self._step += 1
        return int(action.item())
