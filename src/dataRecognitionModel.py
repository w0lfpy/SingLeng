# DESCARGAR EL DATASET DE KAGGLE para el reconocimiento de signos
# import kagglehub

# # Descarga la última versión del dataset
# path = kagglehub.dataset_download("datamunge/sign-language-mnist")

# print("Path to dataset files:", path)

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def finger_y_positions(landmarks, finger_tip, finger_dips, finger_pips, finger_mcp):
    if landmarks[finger_tip].y < landmarks[finger_dips].y:
        return 0
    elif landmarks[finger_tip].y > landmarks[finger_dips].y and landmarks[finger_tip].y < landmarks[finger_pips].y:
        return 1
    elif landmarks[finger_tip].y > landmarks[finger_pips].y and landmarks[finger_tip].y < landmarks[finger_mcp].y:
        return 2
    elif landmarks[finger_tip].y > landmarks[finger_mcp].y:
        return 3
    return None

def finger_x_positions(landmarks, finger_tip, finger_mcp):
    if landmarks[finger_tip].x == landmarks[finger_mcp].x:
        return 0
    elif landmarks[finger_tip].x > landmarks[finger_mcp].x:
        return 1
    elif landmarks[finger_tip].x < landmarks[finger_mcp].x:
        return 2
    return None

def finger_z_positions(landmarks, finger_tip, finger_mcp):
    if abs(landmarks[finger_tip].z - landmarks[finger_mcp].z) < 0.02:
        return 0  # en el mismo plano (o tocándose)
    elif landmarks[finger_tip].z < landmarks[finger_mcp].z:
        return 1  # más cerca de la cámara
    elif landmarks[finger_tip].z > landmarks[finger_mcp].z:
        return 2  # más lejos (hacia la palma)
    return None

class HandGestureRecognition:
    def __init__(self):
        self.palm_state = False
        self.finger_state = [[False, False, False]] * 5
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

    def update_palm_state(self, landmarks):
        if landmarks[0].y < landmarks[17].y:
            self.palm_state = False
        elif landmarks[0].y > landmarks[17].y:
            self.palm_state = True
        return self.palm_state
    
    def update_finger_state(self, landmarks):
        for i in range(5):
            coordenada_y = finger_y_positions(landmarks, self.finger_tips[i], self.finger_dips[i], self.finger_pips[i], self.finger_mcps[i])
            coordenada_x = finger_x_positions(landmarks, self.finger_tips[i], self.finger_mcps[i])
            coordenada_z = finger_z_positions(landmarks, self.finger_tips[i], self.finger_mcps[i])
            self.finger_state[i] = [coordenada_y, coordenada_x, coordenada_z]
        return self.finger_state

    def get_gesture(self):
        f = self.finger_state

        if (f[0][0] == 0 and 
            all(f[i][0] == 3 for i in range(1, 5))):
            return "Gesto A"
        
        elif (f[0][0] == 0 and f[0][1] == 1 and 
            all(f[i][0] == 0 for i in range(1, 5))):
            return "Gesto B"
        
        elif (f[0][1] == 2 and 
            all(f[i][0] == 2 and f[i][1] == 2 for i in range(1, 5))):
            return "Gesto C"
        
        elif (f[0][0] == 0 and f[1][0] == 0 and 
            all(f[i][0] == 2 for i in range(2, 5))):
            return "Gesto D"
        
        elif (f[0][0] == 0 and 
            all(f[i][0] == 2 for i in range(1, 5))):
            return "Gesto E"
        
        elif (f[0][0] == 0 and f[1][0] == 3 and 
            all(f[i][0] == 0 for i in range(2, 5))):
            return "Gesto F"
        
        elif ((f[1][0] == 3 or f[1][0] == 2)):
            return "Gesto G" # Mejorar la lógica para Gesto G
        
        elif ((f[1][0] == 3 or f[1][0] == 2) and (f[2][0] == 3 or f[2][0] == 2)):
            return "Gesto H" # Mejorar la lógica para Gesto H
        
        elif f[0][0] == 0 and f[1][0] == 3 and f[2][0] == 3 and f[3][0] == 3 and f[4][0] == 0:
            return "Gesto I"
        
        elif (f[0][0] == 0 and 
            all(f[i][0] == 3 for i in range(1, 4)) and 
            f[4][0] == 0):
            return "Gesto Y"
        
        return " "

# Tener acceso a la webcam del dispositivo
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5, max_num_hands = 2) as hands:
    finger_state = [False] * 5
    texto_transcrito = []

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
                # Actualizar el estado de los dedos
                finger_state = gesture_recognition.update_finger_state(hand_landmarks.landmark)
                # Obtener el gesto actual
                gesture = gesture_recognition.get_gesture()
                # Si el gesto es diferente de " ", agregarlo al texto transcrito
                if gesture != " ":
                    texto_transcrito.append(gesture_recognition.letters_alpha(gesture))
                # Mostrar el gesto en la imagen
                cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Mostrar el estado de los dedos en la imagen
                for i in range(5):
                    cv2.putText(frame, f"Finger {i+1}: {'Up' if finger_state[i] else 'Down'}", (10, 50 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                # Mostrar el estado de los dedos en la imagen
                cv2.putText(frame, f"Finger 1: {'Up' if finger_state[0][0] == 0 else 'Down'}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, f"Finger 2: {'Up' if finger_state[1][0] == 0 else 'Down'}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, f"Finger 3: {'Up' if finger_state[2][0] == 0 else 'Down'}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, f"Finger 4: {'Up' if finger_state[3][0] == 0 else 'Down'}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, f"Finger 5: {'Up' if finger_state[4][0] == 0 else 'Down'}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                # Mostrar las coordenadas (y, x, z) de los dedos en la imagen
                for i in range(5):
                    cv2.putText(frame, f"Finger {i+1}: ({finger_state[i][0]}, {finger_state[i][1]}, {finger_state[i][2]})", (10, 200 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                # Mostrar el gesto en la imagen
                cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #Nombrar ventana y pasar el frame a la imagen
        cv2.imshow('Hand Tracking', frame)
        # Salir del bucle si se presiona la tecla 'esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break
    #print(texto_transcrito)

cap.release()
cv2.destroyAllWindows()