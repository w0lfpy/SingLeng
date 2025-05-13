import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class HandGestureRecognition:
    def __init__(self):
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_dips = [3, 7, 11, 15, 19]
        self.finger_pips = [2, 6, 10, 14, 18]
        self.finger_mcps = [1, 5, 9, 13, 17]

    def letters_alpha(self, text):
        letters = {
            'Gesto A': 'A', 'Gesto B': 'B', 'Gesto C': 'C',
            'Gesto D': 'D', 'Gesto E': 'E', 'Gesto F': 'F',
            'Gesto G': 'G', 'Gesto H': 'H', 'Gesto I': 'I',
            'Gesto J': 'J', 'Gesto K': 'K', 'Gesto L': 'L',
            'Gesto M': 'M', 'Gesto N': 'N', 'Gesto O': 'O',
            'Gesto P': 'P', 'Gesto Q': 'Q', 'Gesto R': 'R',
            'Gesto S': 'S', 'Gesto T': 'T', 'Gesto U': 'U',
            'Gesto V': 'V', 'Gesto W': 'W', 'Gesto X': 'X',
            'Gesto Y': 'Y'
        }
        return letters.get(text, "Gesto desconocido")

    def get_gesture(self, landmarks):
        if (landmarks[self.finger_tips[0]].y < landmarks[self.finger_pips[0]].y and 
            landmarks[self.finger_tips[0]].x < landmarks[self.finger_pips[1]].x and
            all(landmarks[self.finger_tips[i]].y > landmarks[self.finger_mcps[i]].y + 0.02 for i in range(1, 5)) and 
            all(landmarks[self.finger_tips[0]].x < landmarks[self.finger_tips[i]].x for i in range(1, 5))):
            return "Gesto A"
        elif (landmarks[self.finger_tips[0]].y < landmarks[self.finger_dips[0]].y and 
              landmarks[self.finger_tips[0]].x > landmarks[self.finger_pips[0]].x and 
              landmarks[self.finger_tips[1]].y < landmarks[self.finger_dips[1]].y and 
              landmarks[self.finger_tips[2]].y < landmarks[self.finger_dips[2]].y and 
              landmarks[self.finger_tips[3]].y < landmarks[self.finger_dips[3]].y and 
              landmarks[self.finger_tips[4]].y < landmarks[self.finger_dips[4]].y):
            return "Gesto B"
        elif (landmarks[self.finger_tips[0]].x < landmarks[self.finger_pips[0]].x and 
            landmarks[self.finger_tips[0]].y < landmarks[self.finger_mcps[0]].y and 
            all(landmarks[self.finger_tips[i]].x < landmarks[self.finger_pips[i]].x and
                landmarks[self.finger_tips[i]].y > landmarks[self.finger_pips[i]].y for i in range(1, 5)) and
                abs(landmarks[self.finger_tips[0]].x - landmarks[self.finger_tips[4]].x) > 0.02):
            return "Gesto C"
        elif (landmarks[self.finger_tips[0]].y < landmarks[self.finger_mcps[0]].y and 
              landmarks[self.finger_tips[0]].x < landmarks[self.finger_mcps[0]].x and
              landmarks[self.finger_tips[1]].y < landmarks[self.finger_mcps[1]].y and 
              all(landmarks[self.finger_tips[i]].y > landmarks[self.finger_pips[i]].y for i in range(2, 5)) and 
              all(landmarks[self.finger_tips[1]].x > landmarks[self.finger_tips[i]].x for i in range(2, 5))):
            return "Gesto D"
        elif (landmarks[self.finger_tips[0]].y < landmarks[self.finger_mcps[0]].y and 
            landmarks[self.finger_tips[0]].x > landmarks[self.finger_mcps[0]].x and 
            all(landmarks[self.finger_tips[i]].y > landmarks[self.finger_pips[i]].y for i in range(1, 5)) and
            all(landmarks[self.finger_tips[i]].y < landmarks[self.finger_mcps[i]].y for i in range(1, 5))):
            return "Gesto E"
        elif (landmarks[self.finger_tips[0]].y < landmarks[self.finger_pips[0]].y and 
              landmarks[self.finger_tips[0]].x < landmarks[self.finger_pips[0]].x and 
              landmarks[self.finger_tips[1]].y > landmarks[self.finger_pips[1]].y and 
              landmarks[self.finger_tips[1]].x < landmarks[self.finger_mcps[1]].x and 
              all(landmarks[self.finger_tips[i]].y < landmarks[self.finger_pips[i]].y for i in range(2, 5))):
            return "Gesto F"
        elif (landmarks[self.finger_tips[0]].y > landmarks[self.finger_pips[0]].y and 
              landmarks[self.finger_tips[1]].y < landmarks[self.finger_mcps[1]].y and 
              all(landmarks[self.finger_tips[i]].x > landmarks[self.finger_pips[i]].x for i in range(2, 5))):
            return "Gesto G"
        elif (landmarks[self.finger_tips[0]].y > landmarks[self.finger_pips[0]].y and 
              landmarks[self.finger_tips[1]].y < landmarks[self.finger_mcps[1]].y and 
              landmarks[self.finger_tips[2]].y < landmarks[self.finger_mcps[2]].y and 
              all(landmarks[self.finger_tips[i]].x > landmarks[self.finger_pips[i]].x for i in range(3, 5))):
            return "Gesto H" 
        elif (landmarks[self.finger_tips[0]].x >= landmarks[self.finger_dips[0]].x and 
              landmarks[self.finger_tips[0]].y < landmarks[self.finger_pips[0]].y and 
              all(landmarks[self.finger_tips[i]].y > landmarks[self.finger_pips[i]].y for i in range(1, 4)) and 
              landmarks[self.finger_tips[4]].y < landmarks[self.finger_pips[4]].y):
            return "Gesto I"
        # La 'J' es la misma posición de mano que la I pero con movimiento, no es una posición fija.
        elif (landmarks[self.finger_tips[1]].y < landmarks[self.finger_mcps[1]].y and  
            all(landmarks[self.finger_tips[i]].y > landmarks[self.finger_pips[i]].y for i in range(3, 5))):  
            if (landmarks[self.finger_tips[0]].y < landmarks[self.finger_pips[2]].y and  
                landmarks[self.finger_tips[0]].x > landmarks[self.finger_tips[1]].x and
                landmarks[self.finger_tips[2]].z < landmarks[self.finger_tips[0]].z):  
                return "Gesto K"
            elif (landmarks[self.finger_tips[0]].y > landmarks[self.finger_tips[1]].y and 
                landmarks[self.finger_tips[0]].x < landmarks[self.finger_tips[1]].x and 
                all(landmarks[self.finger_tips[i]].y > landmarks[self.finger_mcps[i]].y for i in range(2, 5))):  
                return "Gesto L"
        elif (all(landmarks[self.finger_tips[i]].y > landmarks[self.finger_pips[i]].y for i in range(1, 5)) and
            landmarks[self.finger_tips[0]].y > landmarks[self.finger_pips[1]].y and
            landmarks[self.finger_tips[0]].x > landmarks[self.finger_mcps[1]].x and
            all(landmarks[self.finger_tips[4]].y > landmarks[self.finger_tips[i]].y for i in range(1, 4))):
            return "Gesto M"
        elif (all(landmarks[self.finger_tips[i]].y > landmarks[self.finger_pips[i]].y for i in range(1, 5)) and
            landmarks[self.finger_tips[0]].y > landmarks[self.finger_pips[1]].y and
            landmarks[self.finger_tips[0]].x > landmarks[self.finger_mcps[1]].x and
            all(landmarks[self.finger_tips[3]].y > landmarks[self.finger_tips[i]].y for i in range(1, 3)) and
            all(landmarks[self.finger_tips[4]].y > landmarks[self.finger_tips[i]].y for i in range(1, 3))):
            return "Gesto N"
        elif (landmarks[self.finger_tips[0]].x < landmarks[self.finger_pips[0]].x and 
            landmarks[self.finger_tips[0]].y < landmarks[self.finger_mcps[0]].y and 
            all(landmarks[self.finger_tips[i]].x < landmarks[self.finger_pips[i]].x and
                landmarks[self.finger_tips[i]].y > landmarks[self.finger_pips[i]].y
                for i in range(1, 5)) and 
            abs(landmarks[self.finger_tips[0]].x - landmarks[self.finger_tips[4]].x) < 0.01):  
            return "Gesto O"
        elif (landmarks[self.finger_tips[0]].x > landmarks[self.finger_mcps[0]].x and
              landmarks[self.finger_tips[0]].x > landmarks[self.finger_pips[2]].x and
              landmarks[self.finger_tips[1]].x > landmarks[self.finger_mcps[1]].x and
              landmarks[self.finger_tips[1]].y >= landmarks[self.finger_mcps[1]].y and
              landmarks[self.finger_tips[2]].y > landmarks[self.finger_mcps[2]].y):
            return "Gesto P" 
        elif (all(landmarks[self.finger_tips[i]].y > landmarks[self.finger_mcps[i]].y for i in range(2, 5)) and
              landmarks[self.finger_tips[0]].y > landmarks[self.finger_mcps[0]].y and
              landmarks[self.finger_tips[1]].y > landmarks[self.finger_mcps[1]].y and
              all(landmarks[self.finger_tips[1]].y > landmarks[self.finger_tips[i]].y for i in range(2, 5))):
            return "Gesto Q"
        elif (landmarks[self.finger_tips[1]].y < landmarks[self.finger_mcps[1]].y and 
            landmarks[self.finger_tips[2]].y < landmarks[self.finger_mcps[2]].y and  
            abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[2]].x) < 0.1 and 
            landmarks[self.finger_tips[3]].y > landmarks[self.finger_pips[3]].y and  
            landmarks[self.finger_tips[4]].y > landmarks[self.finger_pips[4]].y and  
            landmarks[self.finger_tips[0]].y > landmarks[self.finger_pips[0]].y):    
            return "Gesto R" #No Funciona
        elif (all(landmarks[self.finger_tips[i]].y > landmarks[self.finger_mcps[i]].y for i in range(1, 5)) and
            landmarks[self.finger_tips[0]].x < landmarks[self.finger_mcps[0]].x and
            landmarks[self.finger_tips[0]].x > landmarks[self.finger_pips[1]].x):
            return "Gesto S"
        elif (landmarks[self.finger_tips[0]].x > landmarks[self.finger_pips[1]].x and  
            landmarks[self.finger_tips[1]].y < landmarks[self.finger_mcps[1]].y and 
            all(landmarks[self.finger_tips[i]].y > landmarks[self.finger_pips[i]].y for i in range(2, 5))):  
            return "Gesto T" #No Funciona
        elif (landmarks[self.finger_tips[1]].y < landmarks[self.finger_mcps[1]].y - 0.01 and
            landmarks[self.finger_tips[2]].y < landmarks[self.finger_mcps[2]].y - 0.01 and
            abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[2]].x) < 0.09 and
            landmarks[self.finger_tips[3]].y > landmarks[self.finger_dips[3]].y - 0.02 and
            landmarks[self.finger_tips[4]].y > landmarks[self.finger_dips[4]].y - 0.02 and
            abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[2]].x) < abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[0]].x) and
            abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[2]].x) < abs(landmarks[self.finger_tips[2]].x - landmarks[self.finger_tips[0]].x) and
            abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[2]].x) < 0.5 * abs(landmarks[self.finger_mcps[1]].y - landmarks[self.finger_tips[1]].y)): # Ejemplo de relación
            return "Gesto U"
        # elif (landmarks[self.finger_tips[1]].y < landmarks[self.finger_mcps[1]].y - 0.01 and
        #     landmarks[self.finger_tips[2]].y < landmarks[self.finger_mcps[2]].y - 0.01 and
        #     abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[2]].x) < 0.09 and
        #     landmarks[self.finger_tips[3]].y > landmarks[self.finger_dips[3]].y - 0.02 and
        #     landmarks[self.finger_tips[4]].y > landmarks[self.finger_dips[4]].y - 0.02 and
        #     abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[2]].x) < abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[0]].x) and
        #     abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[2]].x) < abs(landmarks[self.finger_tips[2]].x - landmarks[self.finger_tips[0]].x) and
        #     abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[2]].x) < 0.5 * abs(landmarks[self.finger_mcps[1]].y - landmarks[self.finger_tips[1]].y)): # Ejemplo de relación
        #     return "Gesto U"
        elif (landmarks[self.finger_tips[1]].y < landmarks[self.finger_mcps[1]].y - 0.01 and
            landmarks[self.finger_tips[2]].y < landmarks[self.finger_mcps[2]].y - 0.01 and
            abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[2]].x) > 0.2 and
            landmarks[self.finger_tips[3]].y > landmarks[self.finger_dips[3]].y - 0.02 and
            landmarks[self.finger_tips[4]].y > landmarks[self.finger_dips[4]].y - 0.02 and
            abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[2]].x) < abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[0]].x) and
            abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[2]].x) < abs(landmarks[self.finger_tips[2]].x - landmarks[self.finger_tips[0]].x) and
            abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[2]].x) < 0.5 * abs(landmarks[self.finger_mcps[1]].y - landmarks[self.finger_tips[1]].y)): # Ejemplo de relación
            return "Gesto V"
        # elif (landmarks[self.finger_tips[1]].y < landmarks[self.finger_mcps[1]].y and 
        #     landmarks[self.finger_tips[2]].y < landmarks[self.finger_mcps[2]].y and
        #     abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[2]].x) > 0.1):  
        #     return "Gesto V" #No Funciona
        elif (landmarks[self.finger_tips[1]].y < landmarks[self.finger_mcps[1]].y and  
            landmarks[self.finger_tips[2]].y < landmarks[self.finger_mcps[2]].y and  
            abs(landmarks[self.finger_tips[1]].x - landmarks[self.finger_tips[2]].x) > 0.1):  
            return "Gesto W"
        elif (landmarks[self.finger_tips[0]].x < landmarks[self.finger_pips[0]].x and
            landmarks[self.finger_tips[4]].x > landmarks[self.finger_pips[4]].x and
            all(landmarks[self.finger_tips[i]].y > landmarks[self.finger_pips[i]].y for i in range(1, 4))):
            return "Gesto Y"
        return " "

# Tener acceso a la webcam del dispositivo
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5, max_num_hands = 2) as hands:
    finger_state = [False] * 5
    texto_transcrito = []
    gesto_anterior = None
    tiempo_gesto = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        # Convertir la imagen a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Procesar la imagen y detectar las manos
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            # Bucle para graficar las manos detectadas
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar los puntos de referencia de la mano
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                           mp_drawing_styles.get_default_hand_landmarks_style(),
                                           mp_drawing_styles.get_default_hand_connections_style())
                
                finger_tips = [4, 8, 12, 16, 20]
                finger_mcps = [2, 6, 10, 14, 18]

                # Crear una instancia de la clase HandGestureRecognition
                gesture_recognition = HandGestureRecognition()
                # Obtener el gesto actual
                gesture = gesture_recognition.get_gesture(hand_landmarks.landmark)
                # Si el gesto es diferente de " ", agregarlo al texto transcrito
                if gesture != " ":
                    letra_actual = gesture_recognition.letters_alpha(gesture)
                    if gesture != gesto_anterior:
                        gesto_anterior = gesture
                        tiempo_gesto = time.time()
                    elif time.time() - tiempo_gesto > 0.5:  
                        if not texto_transcrito or letra_actual != texto_transcrito[-1]:
                            texto_transcrito.append(letra_actual)
                            tiempo_gesto = time.time()
                else:
                    gesto_anterior = None
                    tiempo_gesto = time.time()
                # Mostrar el gesto en la imagen
                cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Mostrar el estado de los dedos en la imagen
                for i in range(5):
                    cv2.putText(frame, f"Finger {i+1}: {'Up' if finger_state[i] else 'Down'}", (10, 50 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                # Mostrar el estado de los dedos en la imagen
                cv2.putText(frame, f"Finger 1: {'Up' if finger_state[0] else 'Down'}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, f"Finger 2: {'Up' if finger_state[1] else 'Down'}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, f"Finger 3: {'Up' if finger_state[2] else 'Down'}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, f"Finger 4: {'Up' if finger_state[3] else 'Down'}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, f"Finger 5: {'Up' if finger_state[4] else 'Down'}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                # Mostrar las coordenadas (x, y, z) de los dedos en la imagen
                for i in range(5):
                    cv2.putText(frame, f"Finger {i+1}: ({round(hand_landmarks.landmark[finger_tips[i]].x, 2)}, {round(hand_landmarks.landmark[finger_tips[i]].y, 2)}, {round(hand_landmarks.landmark[finger_tips[i]].z, 2)})", (10, 200 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                # Mostrar el texto transcrito en la imagen
                cv2.putText(frame, ''.join(texto_transcrito), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                # Mostrar el gesto en la imagen
                cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #Nombrar ventana y pasar el frame a la imagen
        cv2.imshow('Hand Tracking', frame)
        # Salir del bucle si se presiona la tecla 'esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break
    print(texto_transcrito)

cap.release()
cv2.destroyAllWindows()