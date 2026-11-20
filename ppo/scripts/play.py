"""Juega una misión con el pipeline completo: LLM planificador + agente PPO de habilidades.

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
from crafter_hrl.planner import Planner  # noqa: E402
from crafter_hrl.render import Renderer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mission", default="espada_piedra", choices=list(MISSIONS))
    p.add_argument("--planner", default="llm", choices=["llm", "oracle"])
    p.add_argument("--base-model", default=None, help="modelo base HF (default Qwen2.5-0.5B-Instruct)")
    p.add_argument("--adapter", default=None, help="carpeta LoRA (default: llm/ppo > llm/sft > llm/dummy)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="muestreo del LLM (0 = greedy; un LLM greedy sin entrenar repite siempre lo mismo)")
    p.add_argument("--list-skills", default="factibles", choices=["factibles", "todas"],
                   help="qué habilidades se le listan al LLM en el prompt")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--skills", default="ppo", choices=["ppo", "scripted"])
    p.add_argument("--skill-ckpt", default=None, help="default checkpoints/skills/skill_policy.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-env-steps", type=int, default=2000)
    p.add_argument("--max-decisions", type=int, default=30, help="turnos del LLM (en CPU, unos segundos c/u)")
    p.add_argument("--video", default=None, help="guardar video (.mp4 o .gif)")
    p.add_argument("--window", action="store_true", help="mostrar el juego en vivo (requiere pygame)")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--quiet", action="store_true", help="no imprimir el prompt de cada turno")
    return p.parse_args()


def turn_status(turn: TurnInfo) -> str:
    res = turn.skill_result
    if not turn.valid:
        return "INVÁLIDA"
    if res.success:
        return "✓"
    return "✗ daño" if res.stopped_early else "✗ timeout"


def print_turn(env: PlannerEnv, turn: TurnInfo, reward: float, llm_text: str, obs_text: str, quiet: bool) -> None:
    res = turn.skill_result
    steps = f"{res.steps:3d} pasos" if res else "  0 pasos"
    print(f"\n── Turno {env.decisions:3d} │ {turn.skill or '(no parseable)':28s} │ {turn_status(turn):9s} │ "
          f"{steps} │ r={reward:+.2f} │ t={turn.env_steps}")
    if not quiet:
        print("   LLM: " + llm_text.replace("\n", "\n        "))
        print("   " + obs_text.split("\nHabilidades:")[0].replace("\n", "\n   "))


def run_episode(env: PlannerEnv, planner: Planner, mission: Mission, quiet: bool) -> tuple[TurnInfo, float]:
    """Juega la misión hasta terminar; devuelve el último turno y el retorno total."""
    text, info = env.reset(mission)
    planner.reset(mission)
    
    if not quiet:
        print("\nEntrada del LLM (turno 1):\n   " + text.replace("\n", "\n   "))
    
    total, done = 0.0, False
    while not done:
        out = planner.plan(text, info)
        text, reward, terminated, truncated, turn = env.step(out.text)
        info, total, done = turn.crafter_info, total + reward, terminated or truncated
        print_turn(env, turn, reward, out.text, text, quiet)
        
    return turn, total


def main() -> None:
    args = parse_args()

    skills = build_skill_policy(args.skills, args.skill_ckpt, seed=args.seed)
    
    print("Cargando modelos...", flush=True)
    planner = build_planner(args.planner, args.base_model, args.adapter, args.load_in_4bit, args.temperature)
    
    mission = MISSIONS[args.mission]
    with Renderer.from_options(args.video, args.window, args.fps) as renderer:
        env = PlannerEnv(
                skills, 
                seed=args.seed,
                max_env_steps=args.max_env_steps,
                max_decisions=args.max_decisions, 
                on_env_step=renderer, 
                list_skills=args.list_skills
            )
        print(f"Misión: {mission.text}   [planner={planner.name}, habilidades={skills.name}, seed={args.seed}]")
        turn, total = run_episode(env, planner, mission, args.quiet)

    outcome = "MISIÓN CUMPLIDA" if turn.mission_success else ("MURIÓ" if turn.died else "SIN TIEMPO")
    print(f"\n{outcome} en {env.decisions} decisiones / {turn.env_steps} pasos. Retorno = {total:+.2f}")


if __name__ == "__main__":
    main()
