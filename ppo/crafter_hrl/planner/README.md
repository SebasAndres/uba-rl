# `crafter_hrl/planner` — Planificador (nivel alto)

Responsables: **B** (traductor, oráculo) y **C** (LLM).

## Qué hay

| Archivo | Contenido |
|---|---|
| `translator.py` | `state_to_text()`: misión + vitales + inventario + objetos visibles (dirección, distancia) + historial + lista de habilidades. `SYSTEM_PROMPT` y `format_response()`. |
| `parser.py` | `parse_skill()`: toma la última habilidad válida después de `Habilidad:`; tolera tildes, mayúsculas y espacios. Devuelve `None` si es inválida (→ −0.5). |
| `oracle.py` | `OraclePlanner`: resuelve el árbol tecnológico con reglas (recursivo sobre requisitos, re-planifica en cada turno), pelea zombis cercanos, toma/come/duerme con vitales bajos, y tiene una regla propia para `base`. Escribe un razonamiento en plantilla. |
| `llm.py` | `LLMPlanner`: Qwen2.5 (default 0.5B-Instruct) + adapter LoRA opcional, 4-bit opcional. `build_messages()` = el prompt en formato chat. |
| `base.py` | Interfaz común `plan(text, info) → PlannerOutput(text, skill)`. |

Ejemplo de entrada (lo que ve el LLM):

```
Misión: Fabricá una espada de piedra.
Estado: vida 9/9, comida 7/9, agua 7/9, energía 8/9
Inventario: madera 1, piedra 1, pico de madera 1
Cerca: vaca (S, 1), árbol (NO, 2), mesa (N, 4)
Historial: colocar(mesa) ✓, recolectar(madera) ✓, fabricar(pico, madera) ✓, recolectar(piedra) ✓
Habilidades: recolectar(madera), ..., fabricar(espada, hierro)
```

Salida esperada:

```
Razonamiento: Tengo todo para fabricar espada de piedra.
Habilidad: fabricar(espada, piedra)
```

## Qué hay que hacer

**B — traductor y oráculo**
- [ ] Iterar el formato del texto: ¿incluir día/noche, posición de la última mesa,
      habilidades factibles marcadas? Todo cambio obliga a regenerar el dataset de SFT.
- [ ] Razonamientos del oráculo más variados/informativos (hoy son plantillas
      cortas): el SFT aprende a imitarlos.
- [ ] Mejorar el oráculo donde falla (`pico_hierro`, `carbon_5`): buscar hierro,
      cuidar la vida (no salir de noche, hacer espada antes de bajar a cuevas).
- [ ] Reglas del oráculo para cada misión compuesta nueva.

**C — LLM**
- [ ] Medir el zero-shot (`--planner zeroshot`) con 0.5B y 1.5B: % de respuestas
      parseables y éxito por misión. Es la primera fila de la comparación.
- [ ] Decidir `max_new_tokens` / temperatura para entrenamiento (muestreo) vs evaluación (greedy).
