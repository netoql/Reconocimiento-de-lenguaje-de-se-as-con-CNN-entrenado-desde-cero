from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sign_recognition.cnn_model import (
    RedConvolucionalSenas,
    aumentar_entrada_cnn,
    entrada_cnn_desde_bgr,
    guardar_modelo_cnn,
)
from sign_recognition.dataset import iterar_imagenes
from sign_recognition.preprocess import ConfiguracionPreprocesamiento


class DatasetSenasCNN(Dataset):
    """Dataset de PyTorch con aumentacion opcional."""

    def __init__(self, muestras: list[np.ndarray], objetivos: list[int], aumentar: bool = False) -> None:
        self.muestras = muestras
        self.objetivos = objetivos
        self.aumentar = aumentar

    def __len__(self) -> int:
        return len(self.muestras)

    def __getitem__(self, indice: int):
        muestra = self.muestras[indice]
        if self.aumentar:
            muestra = aumentar_entrada_cnn(muestra)
        return torch.from_numpy(muestra), torch.tensor(self.objetivos[indice], dtype=torch.long)


def cargar_muestras_cnn(
    directorio_datos: Path,
    config: ConfiguracionPreprocesamiento,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Lee imagenes por carpeta y las convierte a entradas de 2 canales."""
    muestras: list[np.ndarray] = []
    etiquetas: list[str] = []
    estadisticas: dict[str, dict[str, int]] = {}

    for etiqueta, ruta_imagen in iterar_imagenes(directorio_datos):
        estadisticas.setdefault(etiqueta, {"total": 0, "usadas": 0, "omitidas": 0})
        estadisticas[etiqueta]["total"] += 1
        imagen = cv2.imread(str(ruta_imagen))
        if imagen is None:
            estadisticas[etiqueta]["omitidas"] += 1
            continue

        muestra = entrada_cnn_desde_bgr(imagen, config)
        if muestra is None:
            estadisticas[etiqueta]["omitidas"] += 1
            continue

        muestras.append(muestra)
        etiquetas.append(etiqueta)
        estadisticas[etiqueta]["usadas"] += 1

    print("Carga por clase:")
    print("Clase | Total | Usadas | Omitidas")
    for etiqueta in sorted(estadisticas):
        item = estadisticas[etiqueta]
        print(f"{etiqueta:>5} | {item['total']:>5} | {item['usadas']:>6} | {item['omitidas']:>8}")

    if not muestras:
        raise RuntimeError(f"No se pudieron cargar muestras validas desde {directorio_datos}")
    return muestras, np.array(etiquetas), sorted(set(etiquetas))


def evaluar(modelo: nn.Module, cargador: DataLoader, dispositivo: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    modelo.eval()
    correctas = 0
    total = 0
    reales: list[int] = []
    predichas: list[int] = []
    with torch.no_grad():
        for x, y in cargador:
            x = x.to(dispositivo)
            y = y.to(dispositivo)
            salidas = modelo(x)
            prediccion = salidas.argmax(dim=1)
            correctas += int((prediccion == y).sum().item())
            total += int(y.numel())
            reales.extend(y.cpu().numpy().tolist())
            predichas.extend(prediccion.cpu().numpy().tolist())
    return correctas / max(total, 1), np.array(reales), np.array(predichas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena una CNN desde cero con mascara HSV y bordes.")
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-out", type=Path, default=Path("models/sign_cnn.pt"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--edge-mode", choices=["canny", "sobel"], default="canny")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo}")

    config = ConfiguracionPreprocesamiento(modo_bordes=args.edge_mode)
    muestras, etiquetas, clases = cargar_muestras_cnn(args.data, config)
    indice_por_clase = {etiqueta: indice for indice, etiqueta in enumerate(clases)}
    objetivos = np.array([indice_por_clase[etiqueta] for etiqueta in etiquetas])

    indices_entrenamiento, indices_prueba = train_test_split(
        np.arange(len(muestras)),
        test_size=0.25,
        random_state=42,
        stratify=objetivos,
    )

    muestras_entrenamiento = [muestras[i] for i in indices_entrenamiento]
    objetivos_entrenamiento = [int(objetivos[i]) for i in indices_entrenamiento]
    muestras_prueba = [muestras[i] for i in indices_prueba]
    objetivos_prueba = [int(objetivos[i]) for i in indices_prueba]

    cargador_entrenamiento = DataLoader(
        DatasetSenasCNN(muestras_entrenamiento, objetivos_entrenamiento, aumentar=True),
        batch_size=args.batch_size,
        shuffle=True,
    )
    cargador_prueba = DataLoader(
        DatasetSenasCNN(muestras_prueba, objetivos_prueba, aumentar=False),
        batch_size=args.batch_size,
        shuffle=False,
    )

    modelo = RedConvolucionalSenas(numero_clases=len(clases)).to(dispositivo)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=0.001)
    funcion_perdida = nn.CrossEntropyLoss()
    mejor_exactitud = 0.0
    mejor_estado = None

    for epoca in range(1, args.epochs + 1):
        modelo.train()
        perdida_acumulada = 0.0
        for x, y in cargador_entrenamiento:
            x = x.to(dispositivo)
            y = y.to(dispositivo)
            optimizador.zero_grad()
            perdida = funcion_perdida(modelo(x), y)
            perdida.backward()
            optimizador.step()
            perdida_acumulada += float(perdida.item()) * int(y.numel())

        exactitud, _, _ = evaluar(modelo, cargador_prueba, dispositivo)
        perdida_promedio = perdida_acumulada / max(len(cargador_entrenamiento.dataset), 1)
        print(f"Epoca {epoca:03d}/{args.epochs} | perdida={perdida_promedio:.4f} | exactitud_val={exactitud:.4f}")

        if exactitud > mejor_exactitud:
            mejor_exactitud = exactitud
            mejor_estado = {clave: valor.detach().cpu().clone() for clave, valor in modelo.state_dict().items()}

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)

    exactitud, reales, predichas = evaluar(modelo, cargador_prueba, dispositivo)
    print(classification_report(reales, predichas, target_names=clases, zero_division=0))
    guardar_modelo_cnn(args.model_out, modelo.cpu(), clases, config, exactitud)
    print(f"Modelo CNN guardado en: {args.model_out}")

    matriz = confusion_matrix(reales, predichas, labels=list(range(len(clases))))
    visualizador = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=clases)
    visualizador.plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    ruta_matriz = args.model_out.parent / "cnn_confusion_matrix.png"
    plt.savefig(ruta_matriz, dpi=160)
    print(f"Matriz de confusion guardada en: {ruta_matriz}")


if __name__ == "__main__":
    main()
