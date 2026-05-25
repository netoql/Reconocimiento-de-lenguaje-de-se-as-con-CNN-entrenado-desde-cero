from __future__ import annotations

from pathlib import Path


EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iterar_imagenes(directorio_datos: Path):
    """Recorre `data/raw` y entrega pares (etiqueta, ruta_imagen)."""
    for carpeta_clase in sorted(p for p in directorio_datos.iterdir() if p.is_dir()):
        for ruta_imagen in sorted(carpeta_clase.rglob("*")):
            if ruta_imagen.suffix.lower() in EXTENSIONES_IMAGEN:
                yield carpeta_clase.name, ruta_imagen


# Alias temporal por compatibilidad con scripts viejos que aun pudieran importarlo.
iter_images = iterar_imagenes
