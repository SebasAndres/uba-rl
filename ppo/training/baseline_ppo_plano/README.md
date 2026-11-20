# Baseline — PPO plano (sin jerarquía)

**Responsable: D (o A, reusando su loop).** Salida: curva y Crafter score a 1M pasos.

Un agente PPO que ve sólo la imagen y optimiza la recompensa nativa de Crafter.
Responde a "¿para qué sirve la jerarquía?": es la primera fila de la tabla de comparación.

## Qué hay que hacer

- [ ] Completar `train_flat_ppo.py`: `FlatCrafter` (wrapper gymnasium) ya está;
      el loop es el de `cleanrl/cleanrl/ppo_atari.py` sin los wrappers de Atari
      (imagen RGB 64×64, sin frame-stack, sin grayscale).
- [ ] 1M de pasos, varias semillas si da el tiempo.
- [ ] Guardar los logros por episodio y calcular el Crafter score con
      `evaluation.evaluate.crafter_score`.
- [ ] Opcional: medir el éxito de las *misiones* con este agente (¿cuántas veces
      fabrica una espada de piedra sin que se lo pidan?).

Referencia: en el paper de Crafter (Hafner, 2021) PPO saca ~4.6% de score a 1M pasos.
