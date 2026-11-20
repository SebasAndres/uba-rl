"""Curriculum de la fase 1: qué habilidades se practican y con qué frecuencia.

Dos mecanismos:
  1. **Etapas**: se desbloquean grupos de habilidades de fácil a difícil
     (madera -> beber/comer -> mesa/herramientas de madera -> piedra/carbón ->
     hierro/diamante/resto). Se avanza cuando la tasa de éxito media de las
     habilidades ya desbloqueadas supera `advance_threshold`.
  2. **Muestreo por dificultad**: dentro de lo desbloqueado, cada habilidad se
     sortea con peso (1 - éxito) + `min_weight`: las que peor andan aparecen más.

Los prerequisitos (inventario, mesa/horno cerca) los regala `SkillEnv`, que es
quien puede editar el mundo. Cada entorno del vector lleva su propio
curriculum (con las estadísticas que ve él).
"""

from collections import deque

import numpy as np

from .catalog import Skill

STAGES: list[list[Skill]] = [
    [Skill("recolectar", "madera")],
    [Skill("beber"), Skill("comer", "vaca")],
    [Skill("colocar", "mesa"), Skill("fabricar", "pico", "madera"), Skill("fabricar", "espada", "madera")],
    [Skill("recolectar", "piedra"), Skill("recolectar", "carbon"), Skill("colocar", "piedra"),
     Skill("colocar", "horno"), Skill("fabricar", "pico", "piedra"), Skill("fabricar", "espada", "piedra")],
    [Skill("recolectar", "hierro"), Skill("recolectar", "diamante"), Skill("recolectar", "brote"),
     Skill("dormir"), Skill("pelear", "zombi"),
     Skill("fabricar", "pico", "hierro"), Skill("fabricar", "espada", "hierro")],
]


class Curriculum:
    """Etapas + muestreo ponderado por (1 - éxito).

    Parameters
    ----------
    skills : list of Skill
        Habilidades permitidas (las de las etapas que no estén acá se ignoran).
    start_stage : int, optional
        Etapa inicial (índice en `STAGES`).
    advance_threshold : float, optional
        Éxito medio de lo desbloqueado necesario para pasar de etapa.
    min_episodes : int, optional
        Intentos mínimos de cada habilidad de la etapa actual antes de avanzar
        (las de etapas anteriores ya lo cumplieron).
    window : int, optional
        Tamaño de la ventana móvil de éxitos por habilidad.
    min_weight : float, optional
        Peso mínimo de muestreo (para no olvidar las que ya salen).
    """

    def __init__(self, skills: list[Skill], start_stage: int = 0, advance_threshold: float = 0.6,
                 min_episodes: int = 10, window: int = 50, min_weight: float = 0.1) -> None:
        self.allowed = set(skills)
        self.stages = [[s for s in stage if s in self.allowed] for stage in STAGES]
        leftover = [s for s in skills if not any(s in stage for stage in self.stages)]
        if leftover:  # habilidades que no están en ninguna etapa: van al final
            self.stages.append(leftover)
        self.stages = [stage for stage in self.stages if stage]
        self.stage = min(start_stage, len(self.stages) - 1)
        self.advance_threshold = advance_threshold
        self.min_episodes = min_episodes
        self.min_weight = min_weight
        self.history: dict[Skill, deque] = {s: deque(maxlen=window) for s in skills}

    def unlocked(self) -> list[Skill]:
        """Habilidades de la etapa actual y las anteriores.

        Returns
        -------
        list of Skill
            Habilidades que se pueden sortear.
        """
        return [s for stage in self.stages[: self.stage + 1] for s in stage]

    def success_rate(self, skill: Skill) -> float | None:
        """Tasa de éxito en la ventana, o None si todavía no se intentó.

        Parameters
        ----------
        skill : Skill
            Habilidad.

        Returns
        -------
        float or None
            Fracción de éxitos.
        """
        history = self.history[skill]
        return float(np.mean(history)) if history else None

    def weights(self, skills: list[Skill]) -> np.ndarray:
        """Probabilidades de muestreo: (1 - éxito) + min_weight, normalizadas.

        Parameters
        ----------
        skills : list of Skill
            Candidatas.

        Returns
        -------
        np.ndarray
            Probabilidades (suman 1).
        """
        rates = [self.success_rate(s) for s in skills]
        w = np.array([1.0 if r is None else 1.0 - r for r in rates]) + self.min_weight
        return w / w.sum()

    def sample(self, skills: list[Skill], rng: np.random.Generator) -> Skill:
        """Sortea una habilidad de `skills` con los pesos por dificultad.

        Parameters
        ----------
        skills : list of Skill
            Candidatas (no vacía).
        rng : np.random.Generator
            Generador.

        Returns
        -------
        Skill
            Habilidad sorteada.
        """
        return skills[rng.choice(len(skills), p=self.weights(skills))]

    def update(self, skill: Skill, success: bool) -> None:
        """Registra un intento y avanza de etapa si corresponde.

        Parameters
        ----------
        skill : Skill
            Habilidad intentada.
        success : bool
            Si se cumplió.
        """
        if skill not in self.history:
            return
        self.history[skill].append(float(success))
        if self.stage + 1 < len(self.stages) and self._ready_to_advance():
            self.stage += 1

    def _ready_to_advance(self) -> bool:
        """Indica si la etapa actual tiene datos y el éxito medio de lo desbloqueado alcanza.

        Returns
        -------
        bool
            True si se puede pasar a la próxima etapa.
        """
        if any(len(self.history[s]) < self.min_episodes for s in self.stages[self.stage]):
            return False
        rates = [r for r in (self.success_rate(s) for s in self.unlocked()) if r is not None]
        return float(np.mean(rates)) >= self.advance_threshold
