import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import time
import pydirectinput
import ctypes
import math
from ctypes.wintypes import RECT, POINT

pydirectinput.PAUSE = 0.0

class SuperMeatBoyEnv(gym.Env):
    def __init__(self):
        super(SuperMeatBoyEnv, self).__init__()
        
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
        self.sct = mss.mss()
        
        self.plantilla_bg = cv2.imread("plantilla_novia.png")
        if self.plantilla_bg is None:
            raise FileNotFoundError("¡Falta plantilla_novia.png!")
        self.h_bg, self.w_bg = self.plantilla_bg.shape[:-1]
        
        hwnd = ctypes.windll.user32.FindWindowW(None, "Super Meat Boy v1.2.5")
        if hwnd:
            rect = RECT()
            ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
            pt = POINT(0, 0)
            ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
            self.monitor = {
                "top": pt.y,
                "left": pt.x,
                "width": rect.right,
                "height": rect.bottom
            }
        else:
            print("❌ AVISO: Juego no encontrado. Abortando cámara.")
            self.monitor = {"top": 0, "left": 0, "width": 640, "height": 360}
        
        self.pos_anterior_x = 0
        self.pos_anterior_y = 0
        
        self.meta_x = 300  
        self.meta_y = 210  
        self.dist_anterior = 0
        
        self.pasos_actuales = 0
        self.max_pasos_por_episodio = 400
        self.nivel_completado = False
        
        self.enfocar_ventana_juego()

    def enfocar_ventana_juego(self):
        hwnd = ctypes.windll.user32.FindWindowW(None, "Super Meat Boy v1.2.5")
        if hwnd:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            time.sleep(0.5)

    def get_observation_and_position(self):
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        obs = cv2.resize(gray, (84, 84))
        obs = np.expand_dims(obs, axis=0)
        
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        
        # RASTREO DE MEAT BOY (Balanceado para evitar congelamientos)
        mask1 = cv2.inRange(hsv, np.array([0, 160, 120]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 160, 120]), np.array([180, 255, 255]))
        red_mask = mask1 | mask2
        contours_mb, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_x = self.pos_anterior_x
        current_y = self.pos_anterior_y
        
        if contours_mb:
            valid_contours_mb = [c for c in contours_mb if 30 < cv2.contourArea(c) < 2000]
            if valid_contours_mb:
                if self.pos_anterior_x == 0 and self.pos_anterior_y == 0:
                    best_contour = max(valid_contours_mb, key=cv2.contourArea)
                else:
                    best_contour = min(
                        valid_contours_mb, 
                        key=lambda c: math.hypot(
                            (cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]/2) - self.pos_anterior_x,
                            (cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]/2) - self.pos_anterior_y
                        )
                    )
                x, y, w, h = cv2.boundingRect(best_contour)
                current_x = int(x + w/2)
                current_y = int(y + h/2)
                
                cv2.rectangle(bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(bgr, "MB", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        # RASTREO DE BANDAGE GIRL
        res = cv2.matchTemplate(bgr, self.plantilla_bg, cv2.TM_CCOEFF_NORMED)
        umbral = 0.85 
        loc = np.where(res >= umbral)
        
        if len(loc[0]) > 0:
            for pt in zip(*loc[::-1]):
                self.meta_x = int(pt[0] + self.w_bg / 2)
                self.meta_y = int(pt[1] + self.h_bg / 2)
                cv2.rectangle(bgr, pt, (pt[0] + self.w_bg, pt[1] + self.h_bg), (255, 0, 255), 2)
                cv2.putText(bgr, "META", (pt[0], pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                break 

        # DIBUJAR LÍNEA Y MOSTRAR DISTANCIA EN PÍXELES
        dist_actual = math.hypot(self.meta_x - current_x, self.meta_y - current_y)
        
        cv2.line(bgr, (current_x, current_y), (self.meta_x, self.meta_y), (0, 255, 0), 2)
        cv2.putText(bgr, f"Distancia: {dist_actual:.1f} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Ojos de la IA", bgr)
        cv2.waitKey(1)
                
        return obs, current_x, current_y

    def step(self, action):
        self.pasos_actuales += 1
        duracion_accion = 0.05 

        teclas_requeridas = []
        if action == 1: teclas_requeridas = ['right']
        elif action == 2: teclas_requeridas = ['left']
        elif action == 3: teclas_requeridas = ['space']
        elif action == 4: teclas_requeridas = ['right', 'space']
        elif action == 5: teclas_requeridas = ['left', 'space']
        elif action == 6: teclas_requeridas = ['right', 'shift']
        elif action == 7: teclas_requeridas = ['left', 'shift']
        elif action == 8: teclas_requeridas = ['right', 'shift', 'space']
        elif action == 9: teclas_requeridas = ['left', 'shift', 'space']

        for tecla in teclas_requeridas:
            pydirectinput.keyDown(tecla)

        todas_las_teclas = ['right', 'left', 'space', 'shift']
        for tecla in todas_las_teclas:
            if tecla not in teclas_requeridas:
                pydirectinput.keyUp(tecla)

        time.sleep(duracion_accion)
        
        obs, current_x, current_y = self.get_observation_and_position()
        
        dist_actual = math.hypot(self.meta_x - current_x, self.meta_y - current_y)
        
        raw_reward = self.dist_anterior - dist_actual
        reward = np.clip(raw_reward, -5.0, 10.0)
        
        if raw_reward <= 0:
            reward -= 0.5 
            
        self.pos_anterior_x = current_x
        self.pos_anterior_y = current_y
        self.dist_anterior = dist_actual
        
        terminated = False
        truncated = False
        
        # VICTORIA ULTRA PRECISA: 20 píxeles de centro a centro (colisión física)
        if dist_actual <= 10.0:
            print(f"\n🎉 ¡Victoria detectada! (Distancia: {dist_actual:.1f} px)")
            reward += 1000.0
            terminated = True
            self.nivel_completado = True
            
        if self.pasos_actuales >= self.max_pasos_por_episodio:
            truncated = True
            
        if terminated or truncated:
            for tecla in todas_las_teclas:
                pydirectinput.keyUp(tecla)
            
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        pydirectinput.keyUp('right')
        pydirectinput.keyUp('left')
        pydirectinput.keyUp('space')
        pydirectinput.keyUp('shift')
        
        if self.nivel_completado:
            print("⏳ Esperando 4 segundos a que pase la animación...")
            time.sleep(4.0) 
            print("🔄 Apretando 'W' para repetir el nivel 1...")
            pydirectinput.keyDown('w')
            time.sleep(0.2)
            pydirectinput.keyUp('w')
            time.sleep(2.5) 
            print("▶️ ¡Retomando entrenamiento en el mismo nivel!")
            self.nivel_completado = False
        else:
            time.sleep(0.5)
            
        # Amnesia para evitar falsas victorias instantáneas post-reset
        self.pos_anterior_x = 0
        self.pos_anterior_y = 0
        self.meta_x = 9999
        self.meta_y = 9999
        
        self.pasos_actuales = 0
        obs, current_x, current_y = self.get_observation_and_position()
        
        self.pos_anterior_x = current_x
        self.pos_anterior_y = current_y
        self.dist_anterior = math.hypot(self.meta_x - current_x, self.meta_y - current_y)
        
        return obs, {}