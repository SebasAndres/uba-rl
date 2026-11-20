"""FASE 2b — PPO multi-turno del LLM planificador (agente de habilidades CONGELADO).

ESQUELETO. Lo que ya está resuelto y hay que usar tal cual:
  - crafter_hrl.PlannerEnv: el entorno a nivel de turno (texto -> recompensa).
  - crafter_hrl.planner.llm.build_messages: el MISMO prompt que en SFT y evaluación.
  - crafter_hrl.factory.build_skill_policy("ppo"): el agente de la fase 1, congelado.

Estructura de una iteración (ver training/fase2_ppo_llm/README.md):

  1. ROLLOUT: N episodios en paralelo (misiones de TRAIN). En cada turno:
       prompt_ids = tokenizar(build_messages(texto))
       response_ids, logprobs = policy.generate(prompt_ids)   # muestreo, temperature ~1
       value = value_head(hidden del último token del prompt)  # V(s_turno)
       texto, r, terminated, truncated, info = env.step(decode(response_ids))
     Guardar por turno: prompt_ids, response_ids, logprobs (por token), value, r, done.

  2. VENTAJAS A NIVEL DE TURNO: GAE sobre la secuencia de turnos del episodio
       delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
       A_t = sum_k (gamma * lambda)^k delta_{t+k}
     La recompensa va en el último token de cada turno; todos los tokens de la
     respuesta del turno t comparten A_t (variante "turn-level"). Alternativa:
     GAE token a token con gamma=1 dentro del turno.

  3. UPDATE PPO por minibatches de turnos:
       ratio = exp(logp_nuevo - logp_viejo) por token (enmascarando el prompt)
       L_clip = -mean(min(ratio*A, clip(ratio, 1-eps, 1+eps)*A))
       L_V    = (V - R)^2  (value head: Linear(hidden, 1) sobre el modelo con LoRA)
       KL contra el modelo SFT (referencia) como penalización: beta * KL
     Sólo se entrenan LoRA + value head.

  4. Loguear: tasa de éxito por misión, retorno, # decisiones, # inválidas, KL.
     Guardar el adapter en checkpoints/llm/ppo (lo levanta scripts/play.py).

Alternativa sin crítico: GRPO (varias rollouts por misión+semilla, ventaja =
retorno normalizado dentro del grupo).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crafter_hrl import PlannerEnv, RewardConfig  # noqa: E402,F401
from crafter_hrl.factory import LLM_SFT, build_skill_policy  # noqa: E402,F401
from crafter_hrl.missions import missions_by_split  # noqa: E402,F401
from crafter_hrl.planner.llm import DEFAULT_BASE_MODEL, build_messages  # noqa: E402,F401

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--init-adapter", default=str(LLM_SFT), help="arrancar desde el SFT")
    p.add_argument("--skills", default="ppo", choices=["ppo", "scripted"])
    p.add_argument("--out", default=str(ROOT / "checkpoints/llm/ppo"))
    p.add_argument("--episodes-per-iter", type=int, default=16)
    p.add_argument("--max-env-steps", type=int, default=1500)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--prerequisite-bonus", type=float, default=0.0, help="0.2 para el shaping opcional")
    args = p.parse_args()

    # Esto ya funciona: el entorno de turnos con el agente congelado.
    skills = build_skill_policy(args.skills)
    env = PlannerEnv(skills, missions=missions_by_split("train"), max_env_steps=args.max_env_steps,
                     rewards=RewardConfig(prerequisite=args.prerequisite_bonus))
    raise NotImplementedError("TODO(C): rollout + GAE por turno + update PPO (ver docstring).")
