"""Reward shaping basado en potencial para la fase 1 (Ng et al., 1999).

    F(s, s') = coef * (gamma * Phi(s') - Phi(s)),   Phi(s) = 1 - d(s) / (max_radius + 1)

donde d(s) es el largo del camino BFS (sobre celdas caminables) hasta quedar al
lado del objetivo de la habilidad. Phi está en [0, 1]: 1 = al lado, 0 = no hay
objetivo alcanzable dentro de `max_radius`.

Por qué es seguro: al ser basado en potencial no cambia la política óptima, sólo
densifica la señal. Al terminar el episodio (éxito, muerte o timeout) se usa
Phi(s') = 0, como pide la teoría para episodios finitos; como Phi >= 0, eso
equivale a una pequeña penalización por terminar lejos y NO premia morir.

Usa información privilegiada (mapa semántico completo), pero sólo en el reward:
el agente sigue viendo únicamente píxeles.
"""

from ..env.perception import find_path_to_adjacent, name_to_ids
from .catalog import Skill, stations_needed

DEFAULT_MAX_RADIUS = 20


def shaping_targets(skill: Skill) -> list[str]:
    """Objetivos del mapa a los que tiene que acercarse la habilidad.

    Parameters
    ----------
    skill : Skill
        Habilidad en ejecución.

    Returns
    -------
    list of str
        Nombres internos de Crafter; vacío si la habilidad no tiene shaping
        (dormir, colocar, recolectar(brote): el pasto está en todos lados).
    """
    if skill.verb == "fabricar":
        return stations_needed(skill)
    if skill.verb in ("dormir", "colocar") or (skill.verb == "recolectar" and skill.obj == "brote"):
        return []
    return [skill.target] if skill.target else []


def bfs_distance(info: dict, target: str, max_radius: int = DEFAULT_MAX_RADIUS) -> int:
    """Pasos BFS hasta quedar al lado de `target` (max_radius + 1 si no se alcanza).

    Parameters
    ----------
    info : dict
        `info` de Crafter (usa "semantic" y "player_pos").
    target : str
        Material u objeto destino (ej. "tree").
    max_radius : int, optional
        Distancia Manhattan máxima del BFS.

    Returns
    -------
    int
        Largo del camino, en [0, max_radius + 1].
    """
    path, _ = find_path_to_adjacent(info["semantic"], info["player_pos"], name_to_ids(target), max_radius)
    return max_radius + 1 if path is None else min(len(path), max_radius + 1)


def potential(skill: Skill, info: dict, max_radius: int = DEFAULT_MAX_RADIUS) -> float:
    """Potencial Phi(s) en [0, 1] (con varias estaciones, la más lejana manda).

    Parameters
    ----------
    skill : Skill
        Habilidad en ejecución.
    info : dict
        `info` de Crafter.
    max_radius : int, optional
        Distancia Manhattan máxima del BFS.

    Returns
    -------
    float
        1 - d / (max_radius + 1); 0 si la habilidad no tiene shaping.
    """
    targets = shaping_targets(skill)
    if not targets:
        return 0.0
    dist = max(bfs_distance(info, t, max_radius) for t in targets)
    return 1.0 - dist / (max_radius + 1)


def shaping_reward(phi_prev: float, phi_next: float, gamma: float, coef: float, done: bool) -> float:
    """Término de shaping F = coef * (gamma * Phi(s') - Phi(s)), con Phi(terminal) = 0.

    Parameters
    ----------
    phi_prev : float
        Potencial antes del paso.
    phi_next : float
        Potencial después del paso.
    gamma : float
        Descuento (tiene que coincidir con el de PPO).
    coef : float
        Escala del shaping.
    done : bool
        Si el episodio de la habilidad terminó en este paso.

    Returns
    -------
    float
        Recompensa extra del paso.
    """
    return coef * (gamma * (0.0 if done else phi_next) - phi_prev)
