import cv2 # type: ignore
import mediapipe as mp # type: ignore

# Inicializar el módulo de MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Función para detectar y dibujar manos en el fotograma
def detect_hands(frame):
    # Convertir la imagen a escala de grises
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Realizar la detección de manos
    results = hands.process(frame_rgb)
    
    # Verificar si se han detectado manos y dibujarlas
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                # Convertir las coordenadas normalizadas a píxeles
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
    
    return frame

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer el fotograma del video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detectar manos en el fotograma
    frame_with_hands = detect_hands(frame)
    
    # Mostrar el fotograma con las manos detectadas
    cv2.imshow('Hand Detection', frame_with_hands)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
