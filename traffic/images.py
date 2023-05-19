# --------------------------------------------------------------------------
# Módulo: images
# Descripción: Módulo de procesamiento de datos de imágenes.
# --------------------------------------------------------------------------

# Imports (standard, third party, local)
from typing import Tuple, List
from pathlib import Path
from os import listdir

import numpy as np
import cv2 as cv

# --------------------------------------------------------------------------
def reescalate(img) -> np.ndarray:
    """
    Reescala la imagen para que se adapte a la pantalla

    :param img: Imagen a reescalar
    :return: Imagen reescalada
    """
    screen_width, screen_height = 1600, 900

    # Tamano de la pantalla
    try:
        height, width, _ = img.shape
    except ValueError:
        height, width = img.shape

    # Factor de escala
    width_scale = screen_width / width
    height_scale = screen_height / height
    scale = min(width_scale, height_scale)

    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_img = cv.resize(img, (new_width, new_height))

    return resized_img

# --------------------------------------------------------------------------
def show_images(images : Tuple[np.ndarray, str]) -> None:
    """
    Muestra una lista de imágenes en una ventana una a una.

    :param lista_imagenes: Lista de imágenes a mostrar junto con su título
    """
    for img, title in images:
        img = reescalate(img)
        cv.imshow(f"{title}", img)
        cv.waitKey(0)
        cv.destroyAllWindows()


# --------------------------------------------------------------------------
def show_image(img, title) -> None:
    """
    Muestra una imagen en una ventana.

    :param img: Imagen a mostrar
    :param titulo: Título de la ventana
    """
    q = reescalate(img)
    cv.imshow(title, q)
    cv.waitKey(0)
    cv.destroyAllWindows()

# --------------------------------------------------------------------------
def load_images(path : Path) -> Tuple[np.ndarray, str]:
    """
    Carga las imágenes de una ruta y les asigna un título.

    :param path: Ruta de las imágenes
    :return: Lista de imágenes
    """
    images = [(cv.imread(str(path / image)), image)for image in listdir(path)]
    return images