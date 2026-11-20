import cv2
import numpy as np
import mss
import ctypes
import math
from ctypes.wintypes import RECT, POINT

hwnd = ctypes.windll.user32.FindWindowW(None, "Super Meat Boy v1.2.5")
if not hwnd:
    print("❌ ERROR: Abre el juego primero.")
    exit()

rect = RECT()
ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
pt = POINT(0, 0)
ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
monitor = {"top": pt.y, "left": pt.x, "width": rect.right, "height": rect.bottom}

with mss.mss() as sct:
    sct_img = sct.grab(monitor)
    frame_inicial = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

print("=========================================================")
print("🎯 MODO SELECCIÓN INTELIGENTE")
print("1. Se abrirá una ventana con el juego congelado.")
print("2. Dibuja un rectángulo con el mouse atrapando a Bandage Girl.")
print("3. Presiona ENTER para confirmar.")
print("=========================================================")

# Seleccionar la región con el mouse
roi = cv2.selectROI("Selecciona a Bandage Girl y presiona ENTER", frame_inicial, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("Selecciona a Bandage Girl y presiona ENTER")

x_roi, y_roi, w_roi, h_roi = roi
if w_roi == 0 or h_roi == 0:
    print("❌ No seleccionaste nada. Saliendo...")
    exit()

# Recortar la imagen y guardarla
plantilla = frame_inicial[int(y_roi):int(y_roi+h_roi), int(x_roi):int(x_roi+w_roi)]
cv2.imwrite("plantilla_novia.png", plantilla)
print("✅ Plantilla memorizada y guardada como 'plantilla_novia.png'. Iniciando radar...")

with mss.mss() as sct:
    while True:
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        
        # 1. RASTREAR A MEAT BOY (Rojo - esto ya funciona perfecto)
        mask1 = cv2.inRange(hsv, np.array([0, 150, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 150, 100]), np.array([180, 255, 255]))
        red_mask = mask1 | mask2
        contours_mb, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mb_x, mb_y = 0, 0
        if contours_mb:
            valid_contours = [c for c in contours_mb if cv2.contourArea(c) > 50]
            if valid_contours:
                best_contour = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(best_contour)
                mb_x = int(x + w/2)
                mb_y = int(y + h/2)
                cv2.rectangle(bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 2. RASTREAR A BANDAGE GIRL (Con la foto/plantilla)
        res = cv2.matchTemplate(bgr, plantilla, cv2.TM_CCOEFF_NORMED)
        umbral = 0.65 # Tolerancia del 65% de coincidencia
        loc = np.where(res >= umbral)
        
        meta_x, meta_y = 0, 0
        for pt_loc in zip(*loc[::-1]):
            cv2.rectangle(bgr, pt_loc, (pt_loc[0] + w_roi, pt_loc[1] + h_roi), (255, 0, 255), 2)
            meta_x = pt_loc[0] + w_roi//2
            meta_y = pt_loc[1] + h_roi//2
            break

        # 3. LÍNEAS Y DISTANCIAS
        if mb_x != 0 and meta_x != 0:
            dist = math.hypot(meta_x - mb_x, meta_y - mb_y)
            cv2.line(bgr, (mb_x, mb_y), (meta_x, meta_y), (0, 255, 0), 2)
            cv2.putText(bgr, f"Distancia: {dist:.1f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if dist < 75:
                cv2.putText(bgr, "¡ZONA DE VICTORIA!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Radar Definitivo", bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()