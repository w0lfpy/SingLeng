# Reconocimiento de Lenguaje de Signos con Visión Artificial

Este proyecto implementa un sistema de reconocimiento de lenguaje de signos utilizando visión por computadora y aprendizaje automático. El sistema detecta gestos de la mano a través de la cámara web, identifica la letra correspondiente del alfabeto en lenguaje de señas y transcribe la secuencia de letras reconocidas.

## Estructura del Proyecto
- **Datasets/**: Contiene los archivos CSV del dataset de lenguaje de signos (Sign Language MNIST).
- **Models/**: Almacena el modelo entrenado en formato `.h5`.
- **src/**: Código fuente principal del sistema.
- **README.md**: Este archivo de documentación.

---

## Requisitos

- Python 3.7+
- [OpenCV](https://pypi.org/project/opencv-python/)
- [MediaPipe](https://pypi.org/project/mediapipe/)
- [NumPy](https://pypi.org/project/numpy/)
- [Pandas](https://pypi.org/project/pandas/)

Instala las dependencias con:

```sh
pip install opencv-python mediapipe numpy pandas
```

# Descripción del Funcionamiento

## 1. Detección de Manos
El sistema utiliza **MediaPipe Hands** para detectar y extraer **21 puntos de referencia (landmarks)** de la mano en tiempo real desde la cámara web.

## 2. Reconocimiento de Gestos
En los archivos `src/dataRecognitionModel.py` y `src/dataRecognitionModel2.py`, la clase principal implementa la lógica para identificar gestos específicos de la mano, correspondientes a letras del alfabeto en lenguaje de señas.

- Se analizan las posiciones relativas de los dedos (`y`, `x`, `z`) para determinar si cada dedo está **extendido** o **doblado**.
- Cada combinación de posiciones de los dedos se asocia a un gesto (por ejemplo, **"Gesto A"**, **"Gesto B"**, etc.).
- El método principal devuelve el **nombre del gesto detectado**.

## 3. Mapeo de Gestos a Letras
El método de mapeo convierte el **nombre del gesto detectado** en la **letra correspondiente del alfabeto**.

## 4. Transcripción
El sistema mantiene una **lista de letras reconocidas** y las muestra en pantalla en tiempo real.

# Ejecución Paso a Paso

1. **Clona el repositorio** y navega a la carpeta del proyecto:
```sh
git clone <URL-del-repositorio>
cd LenguajeSingRecognition
```

2. Asegúrate de tener los datasets y el modelo en las carpetas correspondientes.

3. Instala las dependencias necesarias:
```sh
pip install opencv-python mediapipe numpy pandas
```

4. Ejecuta el script principal:

Puedes elegir entre los dos modelos de reconocimiento:
- Para el modelo basado en lógica de landmarks:
```sh
python src/dataRecognitionModel2.py
```
- Para el modelo basado en lógica alternativa:
```sh
python src/dataRecognitionModel.py
```

5. Permite el acceso a la cámara web cuando se solicite.

6. Realiza los gestos de las letras del alfabeto frente a la cámara.

    - El sistema mostrará en pantalla:
    - El gesto detectado.
    - El estado de cada dedo (arriba/abajo).
    - Las coordenadas de los dedos.
    - El texto transcrito con las letras reconocidas.

7. Para salir, presiona la tecla Esc.

## Notas Técnicas

- El reconocimiento depende de la correcta visibilidad de la mano y la iluminación.
- Algunos gestos pueden requerir ajustes finos en la lógica de detección para mejorar la precisión.
- El modelo puede ser extendido para reconocer palabras completas o frases.

## Créditos
Dataset: Sign Language MNIST
MediaPipe: Google MediaPipe
**Autor:** 
Jose Suárez
**Licencia:**  
Todos los derechos tanto personales como comerciales quedan reservados en exclusiva a los creadores del contenido mostrado.