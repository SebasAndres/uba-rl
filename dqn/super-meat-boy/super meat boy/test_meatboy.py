import time
from stable_baselines3 import PPO
from entorno_meatboy import SuperMeatBoyEnv

print("1. Inicializando entorno de prueba...")
env = SuperMeatBoyEnv()

print("2. Cargando el cerebro entrenado (ppo_meatboy_v1)...")
model = PPO.load("ppo_meatboy_v1", env=env)

print("¡El agente tomará el control de Super Meat Boy en 3 segundos!")
print("Quita las manos del teclado y mira la pantalla...")
time.sleep(3)

obs, info = env.reset()
puntuacion_total = 0

# Dejamos que juegue durante 500 pasos continuos para evaluar su comportamiento
for paso in range(500):
    # El modelo predice la mejor acción basándose en la imagen actual (CnnPolicy)
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    puntuacion_total += reward
    
    # Si el episodio termina o muere, reiniciamos
    if terminated or truncated:
        obs, info = env.reset()

print(f"¡Evaluación finalizada! Puntaje de avance acumulado: {puntuacion_total}")