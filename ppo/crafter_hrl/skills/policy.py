"""Agente de habilidades PPO (fase 1): red + formato de checkpoint.

La MISMA clase la usan el script de entrenamiento (training/fase1_habilidades)
y el script de juego (scripts/play.py). Si se cambia la arquitectura, hay que
mantener `save_skill_policy` / `load_skill_policy` compatibles.

Arquitectura base (a decidir/ajustar por A): CNN estilo Nature (como
cleanrl/ppo_atari.py) sobre la imagen 64x64x3, concatenada con un MLP sobre
[goal_vec | fracción de timeout restante], y dos cabezas: actor (17 logits) y
crítico (1 valor).

La fracción de tiempo hace que el timeout de la habilidad sea observable: sin
ella el mismo frame vale distinto en el paso 5 que en el 95 y el crítico no
puede distinguirlo (Pardo et al. 2018, "Time Limits in RL").
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
TIME_DIM = 1  # fracción de timeout restante


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


class SkillAgent(nn.Module):
    """Red actor-crítico condicionada a la habilidad: CNN(imagen) + MLP(goal_vec, tiempo).

    Parameters
    ----------
    goal_dim : int, optional
        Largo del goal_vec.
    num_actions : int, optional
        Cantidad de acciones primitivas (logits del actor).
    hidden : int, optional
        Tamaño de la capa oculta compartida.
    """

    def __init__(self, goal_dim: int = GOAL_DIM, num_actions: int = NUM_ACTIONS, hidden: int = 512) -> None:
        super().__init__()
        self.cnn = nn.Sequential(                                    # 3x64x64
            layer_init(nn.Conv2d(3, 32, 8, stride=4)), nn.ReLU(),     # 32x15x15
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),    # 64x6x6
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),    # 64x4x4
            nn.Flatten(),
        )
        self.goal = nn.Sequential(layer_init(nn.Linear(goal_dim + TIME_DIM, GOAL_EMBED_DIM)), nn.ReLU())
        self.trunk = nn.Sequential(layer_init(nn.Linear(CNN_OUT_DIM + GOAL_EMBED_DIM, hidden)), nn.ReLU())
        self.actor = layer_init(nn.Linear(hidden, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def features(self, image: torch.Tensor, goal: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Representación compartida por actor y crítico.

        Parameters
        ----------
        image : torch.Tensor
            Imágenes (B, 64, 64, 3) uint8.
        goal : torch.Tensor
            goal_vecs (B, goal_dim).
        time : torch.Tensor
            Fracción de timeout restante (B, 1), en [0, 1].

        Returns
        -------
        torch.Tensor
            Features (B, hidden).
        """
        pixels = image.permute(0, 3, 1, 2).float() / 255.0  # -> (B, 3, 64, 64) en [0, 1]
        return self.trunk(torch.cat([self.cnn(pixels), self.goal(torch.cat([goal, time], dim=1))], dim=1))

    def get_value(self, image: torch.Tensor, goal: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Valor V(s, g) del crítico.

        Parameters
        ----------
        image : torch.Tensor
            Imágenes (B, 64, 64, 3) uint8.
        goal : torch.Tensor
            goal_vecs (B, goal_dim).
        time : torch.Tensor
            Fracción de timeout restante (B, 1), en [0, 1].

        Returns
        -------
        torch.Tensor
            Valores (B, 1).
        """
        return self.critic(self.features(image, goal, time))

    def get_action_and_value(self, image: torch.Tensor, goal: torch.Tensor, time: torch.Tensor,
                             action: torch.Tensor | None = None
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Muestrea (o evalúa) una acción y calcula su log-prob, la entropía y el valor.

        Parameters
        ----------
        image : torch.Tensor
            Imágenes (B, 64, 64, 3) uint8.
        goal : torch.Tensor
            goal_vecs (B, goal_dim).
        time : torch.Tensor
            Fracción de timeout restante (B, 1), en [0, 1].
        action : torch.Tensor or None, optional
            Acciones a evaluar; si es None se muestrean.

        Returns
        -------
        action : torch.Tensor
            Acciones (B,).
        log_prob : torch.Tensor
            Log-probabilidad de cada acción (B,).
        entropy : torch.Tensor
            Entropía de la distribución (B,).
        value : torch.Tensor
            Valores del crítico (B, 1).
        """
        hidden = self.features(image, goal, time)
        dist = Categorical(logits=self.actor(hidden))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden)


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
    torch.save({"state_dict": agent.state_dict(), "goal_dim": GOAL_DIM,
                "num_actions": NUM_ACTIONS, "time_dim": TIME_DIM, "metadata": metadata}, path)


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
        o con una arquitectura sin la entrada de tiempo.
    """
    ckpt = torch.load(path, map_location=device)
    if ckpt["goal_dim"] != GOAL_DIM:
        raise ValueError(f"El checkpoint usa goal_dim={ckpt['goal_dim']} pero el catálogo "
                         f"actual tiene {GOAL_DIM}: ¿cambió skills/catalog.py?")
    if ckpt.get("time_dim") != TIME_DIM:
        raise ValueError("El checkpoint es de antes de agregar la fracción de tiempo como entrada: "
                         "reentrenar, o regenerar el dummy con `python scripts/crear_dummies.py --force`.")
    agent = SkillAgent(ckpt["goal_dim"], ckpt["num_actions"]).to(device)
    agent.load_state_dict(ckpt["state_dict"])
    agent.eval()
    return agent


class PPOSkillPolicy:
    """Adaptador: agente PPO congelado con la misma interfaz que `ScriptedSkillPolicy`.

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
        self._step = 0

    def reset(self, skill: Skill) -> None:
        """Reinicia el contador de pasos de la habilidad (con un LSTM, acá se resetea el estado).

        Parameters
        ----------
        skill : Skill
            Habilidad que se va a ejecutar.
        """
        self._step = 0

    @torch.no_grad()
    def act(self, env: Any, obs: np.ndarray, info: dict, skill: Skill) -> int:
        """Elige una acción primitiva a partir de la imagen, el goal_vec y el tiempo restante.

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
        self._step += 1
        logits = self.agent.actor(self.agent.features(image, goal, time))
        if self.greedy:
            return int(logits.argmax(dim=-1).item())
        return int(Categorical(logits=logits).sample().item())
