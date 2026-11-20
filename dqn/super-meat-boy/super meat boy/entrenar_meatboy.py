import time
import pydirectinput
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from entorno_meatboy import SuperMeatBoyEnv

# ==========================================
# CALLBACK: SOLO FRENO DE MANO
# ==========================================
class ControlCallback(BaseCallback):
    def _on_rollout_end(self) -> None:
        # Suelta todo cuando PPO se pone a hacer cálculos para que no quede atascado
        pydirectinput.keyUp('right')
        pydirectinput.keyUp('left')
        pydirectinput.keyUp('space')
        pydirectinput.keyUp('shift')
        
    def _on_step(self) -> bool:
        # Ya no abortamos el entrenamiento; dejamos que siga farmeando el nivel
        return True

# ==========================================
# SCRIPT DE ENTRENAMIENTO PRINCIPAL
# ==========================================
print("1. Cargando el entorno...")
env = SuperMeatBoyEnv()
print("¡Entorno cargado!")

print("2. Inicializando el modelo PPO (CnnPolicy)...")
model = PPO(
    "CnnPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0003, 
    n_steps=2048, 
    batch_size=64,
    device="cuda"
)

print("3. ¡Todo listo para entrenar!")
print("Liberando teclas atascadas por seguridad y esperando 5 segundos...")

pydirectinput.keyUp('right')
pydirectinput.keyUp('left')
pydirectinput.keyUp('space')
pydirectinput.keyUp('shift')
time.sleep(5)

print("¡Entrenando sin interrupciones!")

start_time = time.time()


model.learn(total_timesteps=10000, callback=ControlCallback())

end_time = time.time()
elapsed_time = end_time - start_time
minutos = int(elapsed_time // 60)
segundos = elapsed_time % 60

# Freno total de seguridad al terminar
pydirectinput.keyUp('right')
pydirectinput.keyUp('left')
pydirectinput.keyUp('space')
pydirectinput.keyUp('shift')

print(f"\n✨ ¡Proceso finalizado en {minutos} minutos y {segundos:.2f} segundos! ✨")
model.save("ppo_meatboy_v1")
print("¡Modelo guardado con éxito como 'ppo_meatboy_v1.zip'!")