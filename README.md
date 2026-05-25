# Reconocimiento de senas con CNN entrenada desde cero

Sistema de vision por computadora para reconocer senas estaticas usando camara, segmentacion HSV, bordes y una red neuronal convolucional pequena entrenada desde cero.

El proyecto no usa modelos preentrenados para clasificar las senas.

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Captura de muestras

Las imagenes se guardan por clase:

```text
data/raw/A/
data/raw/B/
data/raw/C/
```

Capturar una clase:

```powershell
python scripts\capture_dataset.py --label A --count 200 --backend dshow
```

Durante la captura veras:

- video con recuadro central;
- mascara HSV;
- entrada final que vera la CNN.

Presiona:

- `c` para capturar;
- `q` para salir.

## Revisar imagenes procesadas

Antes de entrenar puedes generar vistas de diagnostico:

```powershell
python scripts\preview_preprocessing.py --data data\raw --out models\previews --per-class 20
```

Las imagenes se guardan en:

```text
models/previews/
```

Cada preview muestra:

```text
Original | Mascara HSV | ROI detectado | Bordes | Entrada CNN
```

## Entrenar CNN

```powershell
python scripts\train_cnn.py --data data\raw --model-out models\sign_cnn.pt --epochs 25
```

El entrenamiento usa:

- 75% de imagenes para entrenamiento;
- 25% para prueba;
- aumentacion de datos;
- matriz de confusion.

Salidas:

```text
models/sign_cnn.pt
models/cnn_confusion_matrix.png
```

## Interfaz en vivo

```powershell
python scripts\live_gui.py --cnn-model models\sign_cnn.pt --backend dshow
```

La interfaz permite:

- ver la camara en tiempo real;
- ver la mascara HSV;
- calibrar HSV desde menu;
- tomar una muestra de piel del centro;
- predecir continuamente con suavizado por historial.

## Flujo del sistema

```text
Camara
-> recuadro central
-> conversion HSV
-> mascara de piel
-> limpieza morfologica
-> contorno de mano
-> bordes
-> entrada 64x64 de 2 canales
-> CNN entrenada desde cero
-> prediccion
```

## Archivos principales

```text
scripts/capture_dataset.py       Captura muestras con vista HSV.
scripts/preview_preprocessing.py Revisa como se procesan las imagenes.
scripts/train_cnn.py             Entrena la CNN.
scripts/live_gui.py              Interfaz PySide6 en vivo.
sign_recognition/preprocess.py   HSV, mascara, ROI, bordes.
sign_recognition/cnn_model.py    Arquitectura CNN y guardado/carga.
sign_recognition/camera.py       Apertura robusta de camara.
sign_recognition/dataset.py      Recorrido de imagenes por clase.
```
