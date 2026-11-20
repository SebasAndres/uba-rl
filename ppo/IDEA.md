# Contexto del proyecto: LLM planificador + agente PPO en Crafter

## Objetivo
RL jerárquico. Un LLM finetuneado con RL (PPO) recibe una misión en
texto (ej. "fabricá una espada de piedra") y decide qué habilidad
ejecutar en cada turno. Un agente PPO, entrenado antes y congelado,
ejecuta esas habilidades con acciones primitivas en Crafter.
El LLM resuelve el QUÉ y en qué orden; el agente, el CÓMO.

## Entorno
- Crafter (Minecraft 2D, 22 logros, 17 acciones discretas, obs RGB
  64x64). Corre en CPU y funciona en Colab.
- Craftax (versión en JAX) para entrenar las habilidades más rápido
  en GPU.
- La info útil para chequear habilidades y armar el texto está en
  info["inventory"], info["achievements"] y el mapa semántico.

## Fase 1: agente de habilidades (PPO, sin LLM)
- Una sola política condicionada al objetivo.
- Input: imagen 64x64x3 + goal_vec (one-hot de habilidad + objeto,
  ~25 dims).
- Output: 1 de las 17 acciones por paso.
- Habilidades: recolectar(madera|piedra|carbón|hierro|diamante|
  brote), beber(), comer(vaca), dormir(), pelear(zombi),
  colocar(mesa|horno|piedra), fabricar(pico|espada, material).
- Recompensa: +1 si el inventario cambia como pide la habilidad,
  -0.01 por paso, timeout de 100 pasos.
- Se sortean solo habilidades posibles. Para practicar las avanzadas
  se puede editar el inventario en el reset (env._player.inventory).
- Al terminar, la política se congela.

## Fase 2: LLM planificador (PPO, agente congelado)
- Input (texto): misión + estado (vida/comida/agua/energía) +
  inventario + objetos cercanos con dirección y distancia +
  historial + lista de habilidades.
- Output: una línea de razonamiento + una habilidad con sintaxis fija,
  ej. "recolectar(piedra)".
- Loop: LLM -> parser -> (habilidad, objeto) -> encode -> goal_vec ->
  el agente ejecuta hasta cumplir o timeout -> estado a texto -> LLM.
  Un paso por turno, no un plan entero de entrada.
- Recompensa: +1 misión cumplida (chequeo sobre info, ej.
  inventory["stone_sword"] >= 1), -0.5 si muere o pide algo
  inválido, -0.02 por decisión. Opcional: +0.2 por prerrequisitos.
- PPO multi-turno: recompensa en el último token de cada turno,
  value head + GAE para repartir crédito entre turnos.
  Alternativa sin crítico: GRPO.
- Modelo: Qwen2.5 0.5B-1.5B con LoRA, 4-bit si hace falta.
- Pipeline del LLM: zero-shot (baseline) -> SFT con planes de un
  oráculo/scripteados -> PPO.

## Ejemplo: espada de piedra
recolectar(madera) -> colocar(mesa) -> fabricar(pico, madera) ->
recolectar(piedra) -> fabricar(espada, piedra)

## Misiones
- Nativas: espada de piedra, pico de hierro, horno, juntar 5 de carbón.
- Compuestas, con chequeo propio: "construí una casa", "prepará
  una base".
- Separar misiones de train y de test para medir generalización.
  Es el experimento clave que justifica el LLM.

## Evaluación
- Crafter score: media geométrica de la tasa de éxito de los logros.
- Tasa de éxito por misión.
- Comparar: PPO plano (1M de pasos), LLM zero-shot + habilidades,
  LLM SFT + habilidades, LLM SFT+PPO + habilidades.
- Ablación: habilidades scripteadas vs entrenadas con RL.

## Restricciones
- Colab o PC con GPU modesta.
- Guardar checkpoints en Drive.
- Episodios de entrenamiento del LLM de 1000-2000 pasos.
- Entornos en paralelo (SubprocVecEnv).

## Equipo de 4
- A: habilidades PPO.
- B: traductor de estado a texto, oráculo y dataset de SFT.
- C: SFT y PPO del LLM.
- D: integración, evaluación y demo.
Al principio se conectan con habilidades scripteadas para que
nadie quede bloqueado.

## Primeros pasos para probar
1. Instalar crafter, correr un episodio aleatorio e inspeccionar
   obs e info.
2. Escribir estado_a_texto(info) y el parser y encoder de habilidades.
3. Implementar 2-3 habilidades scripteadas y un LLM zero-shot que
   intente la misión "espada de piedra".
4. Entrenar con PPO la primera habilidad condicionada,
   recolectar(madera), y reemplazar la versión scripteada.
