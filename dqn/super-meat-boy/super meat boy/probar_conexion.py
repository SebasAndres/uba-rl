from entorno_meatboy import SuperMeatBoyEnv
import time

print("Inicializando entorno de Super Meat Boy en la PC...")
env = SuperMeatBoyEnv()

print("¡Todo listo! El test de acciones aleatorias comienza en 3 segundos...")
time.sleep(3)

obs, info = env.reset()

for paso in range(50):
    # El agente elige una acción al azar (0, 1, 2 o 3)
    accion_aleatoria = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(accion_aleatoria)
    print(f"Paso {paso} | Acción: {accion_aleatoria} | Recompensa (Avance X): {reward}")

print("¡Test finalizado con éxito! El puente funciona.")