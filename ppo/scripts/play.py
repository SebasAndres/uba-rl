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

from crafter_hrl import PlannerEnv  # noqa: E402
from crafter_hrl.factory import build_planner, build_skill_policy  # noqa: E402
from crafter_hrl.missions import MISSIONS  # noqa: E402


def parse_args():
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


class Renderer:
    def __init__(self, video: str | None, window: bool, fps: int, size: int = 512):
        self.frames, self.video, self.size, self.fps = [], video, size, fps
        self.screen = None
        if window:
            import pygame
            pygame.init()
            self.pygame = pygame
            self.screen = pygame.display.set_mode((size, size))
            self.clock = pygame.time.Clock()

    def __call__(self, env, action, info):
        if not (self.video or self.screen):
            return
        frame = env.render(self.size)
        if self.video:
            self.frames.append(frame)
        if self.screen is not None:
            surf = self.pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            self.screen.blit(surf, (0, 0))
            self.pygame.display.flip()
            self.pygame.event.pump()
            self.clock.tick(self.fps)

    def close(self):
        if self.video and self.frames:
            import imageio
            Path(self.video).parent.mkdir(parents=True, exist_ok=True)
            try:
                kwargs = {"fps": self.fps} if self.video.endswith(".mp4") else {"duration": 1000 / self.fps}
                imageio.mimsave(self.video, self.frames, **kwargs)
            except ValueError:  # sin backend para mp4 (falta imageio[ffmpeg])
                self.video = str(Path(self.video).with_suffix(".gif"))
                print("Sin soporte mp4 (pip install 'imageio[ffmpeg]'): guardo GIF.")
                imageio.mimsave(self.video, self.frames, duration=1000 / self.fps)
            print(f"\nVideo guardado en {self.video} ({len(self.frames)} frames)")
        if self.screen is not None:
            self.pygame.quit()


def main():
    args = parse_args()
    skills = build_skill_policy(args.skills, args.skill_ckpt, seed=args.seed)
    print("Cargando modelos...", flush=True)
    planner = build_planner(args.planner, args.base_model, args.adapter, args.load_in_4bit, args.temperature)
    renderer = Renderer(args.video, args.window, args.fps)
    env = PlannerEnv(skills, seed=args.seed, max_env_steps=args.max_env_steps,
                     max_decisions=args.max_decisions, on_env_step=renderer, list_skills=args.list_skills)

    mission = MISSIONS[args.mission]
    text, info = env.reset(mission)
    planner.reset(mission)
    print(f"Misión: {mission.text}   [planner={planner.name}, habilidades={skills.name}, seed={args.seed}]")

    total, done = 0.0, False
    while not done:
        if env.decisions == 0 and not args.quiet:
            print("\nEntrada del LLM (turno 1):\n   " + text.replace("\n", "\n   "))
        out = planner.plan(text, info)
        text, reward, terminated, truncated, turn = env.step(out.text)
        info, total, done = turn.crafter_info, total + reward, terminated or truncated
        res = turn.skill_result
        status = ("INVÁLIDA" if not turn.valid else "✓" if res.success
                  else "✗ daño" if res.stopped_early else "✗ timeout")
        steps = f"{res.steps:3d} pasos" if res else "  0 pasos"
        print(f"\n── Turno {env.decisions:3d} │ {turn.skill or '(no parseable)':28s} │ {status:9s} │ "
              f"{steps} │ r={reward:+.2f} │ t={turn.env_steps}")
        if not args.quiet:
            print("   LLM: " + out.text.replace("\n", "\n        "))
            print("   " + text.split("\nHabilidades:")[0].replace("\n", "\n   "))

    outcome = "MISIÓN CUMPLIDA" if turn.mission_success else ("MURIÓ" if turn.died else "SIN TIEMPO")
    print(f"\n{outcome} en {env.decisions} decisiones / {turn.env_steps} pasos. Retorno = {total:+.2f}")
    renderer.close()


if __name__ == "__main__":
    main()
