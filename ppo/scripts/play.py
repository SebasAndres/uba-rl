"""Juega una misión con el pipeline jerárquico completo: LLM planificador + agente PPO.

Hay DOS agentes, cada uno en un nivel distinto:

  * LLM planificador (nivel alto, `llm_planner`): lee el estado del juego en texto
    y responde con el PLAN del turno: razonamiento + UNA habilidad a ejecutar,
    ej. "fabricar(pico, madera)". No toca el juego directamente.
  * Agente de habilidades (nivel bajo, `skill_agent`): red PPO de la fase 1. Una
    "habilidad" (skill) es una sub-tarea del catálogo (crafter_hrl/skills/catalog.py):
    recolectar(madera), colocar(mesa), pelear(zombi)... El agente recibe la
    habilidad elegida por el LLM codificada como `goal_vec` y, con eso + la imagen,
    elige acciones primitivas de Crafter (moverse, golpear, ...) hasta cumplirla.

    estado (texto) ──► LLM ──► plan "fabricar(pico, madera)" ──► parser ──► goal_vec
          ▲                                                                  │
          └──── estado nuevo ◄── Crafter ◄── acciones primitivas ◄── agente PPO

Carga los modelos de checkpoints/ (agente: skills/skill_policy.pt; LLM: el
adapter más entrenado que exista, llm/ppo > llm/sft > llm/dummy). Antes de
entrenar, crear los dummies con `python scripts/crear_dummies.py`.

Ejemplos:
    python scripts/play.py                                   # LLM + agente (dummies o entrenados)
    python scripts/play.py --mission base --video runs/base.mp4
    python scripts/play.py --window                          # ventana en vivo (pygame)
    python scripts/play.py --adapter checkpoints/llm/sft     # elegir adapter
    python scripts/play.py --planner oracle --skills scripted   # referencia scripteada (sin modelos)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crafter_hrl import PlannerEnv, TurnInfo  # noqa: E402
from crafter_hrl.factory import build_planner, build_skill_policy  # noqa: E402
from crafter_hrl.missions import MISSIONS, Mission  # noqa: E402
from crafter_hrl.planner import SYSTEM_PROMPT, Planner  # noqa: E402
from crafter_hrl.render import Renderer  # noqa: E402
from crafter_hrl.skills import SkillPolicy  # noqa: E402

INDENT = "   "
WIDTH = 78  # ancho de los separadores


def print_section(title: str, body: str) -> None:
    """Imprime una sección de un turno: título con separador y cuerpo indentado.

    Parameters
    ----------
    title : str
        Título de la sección.
    body : str
        Contenido (puede ser multilínea).
    """
    print(f"\n{INDENT}── {title} " + "─" * max(0, WIDTH - len(INDENT) - len(title) - 4))
    print(indent(body, INDENT * 2))


# ─────────────────────────────────────────────────────────────── argumentos
def add_planner_args(parser: argparse.ArgumentParser) -> None:
    """Agrega los flags del nivel alto (LLM planificador).

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser al que se le agregan los argumentos (se modifica in-place).
    """
    group = parser.add_argument_group("nivel alto: LLM planificador")
    group.add_argument("--planner", default="llm", choices=["llm", "oracle"],
                       help="quién elige la habilidad: llm (modelo) u oracle (plan scripteado)")
    group.add_argument("--base-model", default=None, help="modelo base HF (default Qwen2.5-0.5B-Instruct)")
    group.add_argument("--adapter", default=None, help="carpeta LoRA (default: llm/ppo > llm/sft > llm/dummy)")
    group.add_argument("--temperature", type=float, default=0.7,
                       help="muestreo del LLM (0 = greedy; un LLM greedy sin entrenar repite siempre lo mismo)")
    group.add_argument("--list-skills", default="factibles", choices=["factibles", "todas"],
                       help="qué habilidades se le listan al LLM en el prompt")
    group.add_argument("--load-in-4bit", action="store_true", help="cuantizar el LLM a 4 bits")


def add_skill_args(parser: argparse.ArgumentParser) -> None:
    """Agrega los flags del nivel bajo (agente de habilidades).

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser al que se le agregan los argumentos (se modifica in-place).
    """
    group = parser.add_argument_group("nivel bajo: agente de habilidades")
    group.add_argument("--skills", default="ppo", choices=["ppo", "scripted"],
                       help="quién ejecuta la habilidad: ppo (agente RL) o scripted (BFS)")
    group.add_argument("--skill-ckpt", default=None, help="default checkpoints/skills/skill_policy.pt")


def add_episode_args(parser: argparse.ArgumentParser) -> None:
    """Agrega los flags del episodio y de la visualización.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser al que se le agregan los argumentos (se modifica in-place).
    """
    group = parser.add_argument_group("episodio y visualización")
    group.add_argument("--mission", default="espada_piedra", choices=list(MISSIONS))
    group.add_argument("--seed", type=int, default=0)
    group.add_argument("--max-env-steps", type=int, default=2000, help="pasos primitivos máximos de Crafter")
    group.add_argument("--max-decisions", type=int, default=30, help="turnos del LLM (en CPU, unos segundos c/u)")
    group.add_argument("--video", default=None, help="guardar video (.mp4 o .gif)")
    group.add_argument("--window", action="store_true", help="mostrar el juego en vivo (requiere pygame)")
    group.add_argument("--fps", type=int, default=15)
    group.add_argument("--quiet", action="store_true", help="no imprimir el plan ni el estado de cada turno")


def parse_args() -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos.

    Returns
    -------
    argparse.Namespace
        Argumentos parseados, agrupados en nivel alto, nivel bajo y episodio.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_planner_args(parser)
    add_skill_args(parser)
    add_episode_args(parser)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────── impresión
def indent(text: str, prefix: str = INDENT) -> str:
    """Indenta todas las líneas de un texto multilínea.

    Parameters
    ----------
    text : str
        Texto a indentar.
    prefix : str, optional
        Prefijo que se antepone a cada línea.

    Returns
    -------
    str
        Texto con `prefix` al comienzo de cada línea.
    """
    return prefix + text.replace("\n", "\n" + prefix)


def turn_status(turn: TurnInfo) -> str:
    """Resume en una etiqueta corta cómo terminó la habilidad del turno.

    Parameters
    ----------
    turn : TurnInfo
        Información del turno devuelta por `PlannerEnv.step`.

    Returns
    -------
    str
        "INVÁLIDA" si el LLM pidió algo no parseable o no factible, "✓" si el
        agente cumplió la habilidad, "✗ daño" si se interrumpió por recibir daño
        y "✗ timeout" si se agotaron los pasos de la habilidad.
    """
    if not turn.valid:
        return "INVÁLIDA"
    if turn.skill_result.success:
        return "✓"
    return "✗ daño" if turn.skill_result.stopped_early else "✗ timeout"


def reward_breakdown(env: PlannerEnv, turn: TurnInfo, reward: float) -> str:
    """Descompone la recompensa del turno en sus términos (ver `RewardConfig`).

    Parameters
    ----------
    env : PlannerEnv
        Entorno jerárquico (de ahí salen los valores de cada término).
    turn : TurnInfo
        Información del turno.
    reward : float
        Recompensa total del turno.

    Returns
    -------
    str
        Ej. "-0.52 = -0.02 (decisión) -0.50 (inválida)".
    """
    rewards = env.rewards
    terms = [(rewards.per_decision, "decisión")]
    if not turn.valid:
        terms.append((rewards.invalid, "inválida"))
    if turn.mission_success:
        terms.append((rewards.success, "misión cumplida"))
    if turn.died:
        terms.append((rewards.death, "murió"))
    remainder = reward - sum(value for value, _ in terms)
    if abs(remainder) > 1e-9:
        terms.append((remainder, "logros nuevos"))
    return f"{reward:+.2f} = " + " ".join(f"{value:+.2f} ({name})" for value, name in terms)


def execution_summary(env: PlannerEnv, turn: TurnInfo) -> str:
    """Describe qué hizo el agente de habilidades con el plan del LLM.

    Parameters
    ----------
    env : PlannerEnv
        Entorno jerárquico (de ahí sale el nombre del agente).
    turn : TurnInfo
        Información del turno.

    Returns
    -------
    str
        Una línea: habilidad ejecutada, cantidad de acciones primitivas y resultado.
    """
    if turn.skill is None:
        return "no se ejecutó nada: no se encontró una habilidad válida en la salida del LLM"
    if not turn.valid:
        return f"no se ejecutó {turn.skill}: faltan materiales en el inventario"
    steps = turn.skill_result.steps
    return (f"el agente {env.skill_policy.name} ejecutó {turn.skill} con {steps} acciones "
            f"primitivas → {turn_status(turn)}")


def print_turn(env: PlannerEnv, turn: TurnInfo, reward: float, llm_input: str, plan_text: str,
               state_after: str, quiet: bool) -> None:
    """Imprime un turno: entrada y salida del LLM, ejecución, estado resultante y recompensa.

    Parameters
    ----------
    env : PlannerEnv
        Entorno jerárquico; se usa para el número de decisión y el nombre del agente.
    turn : TurnInfo
        Información del turno devuelta por `PlannerEnv.step`.
    reward : float
        Recompensa del turno (la que recibiría el LLM en la fase 2).
    llm_input : str
        Estado en texto que leyó el LLM para decidir (mensaje de usuario del prompt).
    plan_text : str
        Respuesta cruda del LLM.
    state_after : str
        Estado en texto después de ejecutar la habilidad (= entrada del próximo turno).
    quiet : bool
        Si es True, sólo se imprime la línea de resumen.
    """
    header = (f" TURNO {env.decisions:3d} │ {turn.skill or '(no parseable)':24s} │ "
              f"{turn_status(turn):9s} │ r={reward:+.2f} │ t={turn.env_steps}")
    if quiet:
        print(header)
        return
    print(f"\n\n{'═' * WIDTH}\n{header}\n{'═' * WIDTH}")
    extracted = (f"→ habilidad extraída: {turn.skill or 'ninguna'}\n"
                 "  (si nombra varias, el parser toma la última después de 'Habilidad:')")
    print_section("① Entrada del LLM (system prompt + este texto)", llm_input)
    print_section("② Salida del LLM", f"{plan_text}\n\n{extracted}")
    print_section("③ Ejecución", execution_summary(env, turn))
    print_section("④ Estado tras ejecutar", state_after.split("\nHabilidades:")[0])
    print_section("⑤ Recompensa", reward_breakdown(env, turn, reward))


def print_outcome(env: PlannerEnv, last_turn: TurnInfo, total_reward: float) -> None:
    """Imprime el resultado final del episodio.

    Parameters
    ----------
    env : PlannerEnv
        Entorno jerárquico; se usa para la cantidad de decisiones tomadas.
    last_turn : TurnInfo
        Último turno del episodio.
    total_reward : float
        Suma de las recompensas de todos los turnos.
    """
    if last_turn.mission_success:
        outcome = "MISIÓN CUMPLIDA"
    elif last_turn.died:
        outcome = "MURIÓ"
    else:
        outcome = "SIN TIEMPO"
    print(f"\n{'█' * WIDTH}\n {outcome} en {env.decisions} decisiones / {last_turn.env_steps} pasos."
          f"  Retorno = {total_reward:+.2f}\n{'█' * WIDTH}")


# ─────────────────────────────────────────────────────────────── episodio
def play_turn(env: PlannerEnv, llm_planner: Planner, state_text: str, crafter_info: dict,
              renderer: Renderer) -> tuple[str, float, bool, TurnInfo, str]:
    """Juega UN turno: el LLM elige una habilidad y el agente de RL la ejecuta.

    Parameters
    ----------
    env : PlannerEnv
        Entorno jerárquico que contiene al agente de habilidades.
    llm_planner : Planner
        Planificador de nivel alto (LLM u oráculo).
    state_text : str
        Estado actual en texto: es la entrada del LLM.
    crafter_info : dict
        `info` crudo de Crafter (sólo lo usan los planificadores scripteados).
    renderer : Renderer
        Mantiene viva la ventana mientras el LLM piensa.

    Returns
    -------
    next_state_text : str
        Estado en texto después de ejecutar la habilidad.
    reward : float
        Recompensa del turno.
    done : bool
        True si el episodio terminó (misión cumplida, muerte o límite de tiempo).
    turn : TurnInfo
        Detalle del turno (habilidad, validez, pasos del agente, etc.).
    plan_text : str
        Respuesta cruda del LLM.
    """
    plan = renderer.run_responsive(llm_planner.plan, state_text, crafter_info)
    next_state_text, reward, terminated, truncated, turn = env.step(plan.text)
    return next_state_text, reward, terminated or truncated, turn, plan.text


def run_episode(env: PlannerEnv, llm_planner: Planner, mission: Mission, quiet: bool,
                renderer: Renderer) -> tuple[TurnInfo, float]:
    """Juega una misión completa, turno a turno, hasta que el episodio termine.

    Cada turno = 1 decisión del LLM + 1 habilidad ejecutada por el agente de RL.

    Parameters
    ----------
    env : PlannerEnv
        Entorno jerárquico que contiene al agente de habilidades.
    llm_planner : Planner
        Planificador de nivel alto (LLM u oráculo).
    mission : Mission
        Misión a cumplir.
    quiet : bool
        Si es True, no imprime el prompt inicial ni el detalle de cada turno.
    renderer : Renderer
        Visualización (mantiene viva la ventana mientras el LLM piensa).

    Returns
    -------
    last_turn : TurnInfo
        Último turno jugado (indica si se cumplió la misión o si murió).
    total_reward : float
        Suma de las recompensas de todos los turnos.
    """
    state_text, crafter_info = env.reset(mission)
    llm_planner.reset(mission)
    if not quiet:
        print(f"\nSystem prompt del LLM (igual en todos los turnos):\n{indent(SYSTEM_PROMPT)}")

    total_reward, done = 0.0, False
    while not done:
        llm_input = state_text
        state_text, reward, done, turn, plan_text = play_turn(env, llm_planner, llm_input, crafter_info, renderer)
        crafter_info = turn.crafter_info
        total_reward += reward
        print_turn(env, turn, reward, llm_input, plan_text, state_text, quiet)
    return turn, total_reward


# ─────────────────────────────────────────────────────────────── main
def build_agents(args: argparse.Namespace) -> tuple[SkillPolicy, Planner]:
    """Carga los dos agentes del pipeline a partir de los flags.

    Parameters
    ----------
    args : argparse.Namespace
        Argumentos de la línea de comandos.

    Returns
    -------
    skill_agent : SkillPolicy
        Nivel bajo: agente de RL (PPO, fase 1) o scripteado que ejecuta habilidades.
    llm_planner : Planner
        Nivel alto: LLM (u oráculo) que elige qué habilidad ejecutar en cada turno.
    """
    print("Cargando modelos...", flush=True)
    skill_agent = build_skill_policy(args.skills, args.skill_ckpt, seed=args.seed)
    llm_planner = build_planner(args.planner, args.base_model, args.adapter,
                                args.load_in_4bit, args.temperature)
    return skill_agent, llm_planner


def main() -> None:
    """Carga los agentes, arma el entorno jerárquico y juega una misión."""
    args = parse_args()
    skill_agent, llm_planner = build_agents(args)
    mission = MISSIONS[args.mission]

    with Renderer.from_options(args.video, args.window, args.fps) as renderer:
        # El env envuelve al agente de habilidades: cada env.step(plan_del_llm)
        # lo hace jugar una habilidad completa con acciones primitivas.
        env = PlannerEnv(
            skill_agent,
            seed=args.seed,
            max_env_steps=args.max_env_steps,
            max_decisions=args.max_decisions,
            on_env_step=renderer,
            list_skills=args.list_skills,
        )
        print(f"Misión: {mission.text}   [planificador={llm_planner.name}, "
              f"agente de habilidades={skill_agent.name}, seed={args.seed}]")
        last_turn, total_reward = run_episode(env, llm_planner, mission, args.quiet, renderer)

    print_outcome(env, last_turn, total_reward)


if __name__ == "__main__":
    main()
