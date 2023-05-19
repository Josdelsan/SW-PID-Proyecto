# --------------------------------------------------------------------------
# Módulo: detection
# Descripción: Módulo de detección de señales de tráfico 
#              haciendo uso de OpenCV.
# --------------------------------------------------------------------------

# Imports (standard, third party, local)
import cv2 as cv
import numpy as np
from typing import Tuple

# --------------------------------------------------------------------------
def apply_red_mask(img) -> np.ndarray:
    """
    Aplica una máscara que resalta los píxeles rojos de la imagen.

    :param img: Imagen BGR a la que se le aplica la máscara.
    :return: Imagen BGR con la máscara aplicada.
    """
    img_copy = img.copy()

    # Se define el rango de colores a detectar
    h_inf_1=170
    h_sup_1=180
    h_inf_2=0
    h_sup_2=10

    lower_red_1 = np.array([h_inf_1,50,50])
    upper_red_1 = np.array([h_sup_1,255,255])
    lower_red_2 = np.array([h_inf_2,50,50])
    upper_red_2 = np.array([h_sup_2,255,255])

    # Se convierte la imagen a HSV
    hsv = cv.cvtColor(img_copy, cv.COLOR_BGR2HSV)
    red_mask_1 = cv.inRange(hsv, lower_red_1, upper_red_1)
    red_mask_2 = cv.inRange(hsv, lower_red_2, upper_red_2)
    red_mask = red_mask_1+red_mask_2

    # Se aplica una mascara para eliminar todo lo que no sea rojo
    img_rojo = cv.bitwise_and(img_copy,img_copy, mask= red_mask)

    return img_rojo


# --------------------------------------------------------------------------
def apply_binary_and_erode(img) -> np.ndarray:
    """
    Aplica una binarización y erosión a la imagen.

    :param img: Imagen BGR a la que se le aplica la binarización y erosión.
    :return: Imagen binarizada GRAY y erosionada.
    """
    img_copy = img.copy()

    # Se convierte la imagen a escala de grises
    gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]

    # Se eliminan el ruido mediante una erosión
    erored = cv.erode(binary, np.ones((3,3),np.uint8), iterations = 1)

    return erored


# --------------------------------------------------------------------------
def detect_contours(img) -> list:
    """
    Detecta los contornos de la imagen.

    :param img: Imagen GRAY a la que se le detectan los contornos.
    :return: Lista de contornos detectados.
    """
    contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
    contours = contours[0]
    biggest_contour = max(contours, key=cv.contourArea)
    contours = [c for c in contours if cv.contourArea(c) > (cv.contourArea(biggest_contour)*0.4)]

    return contours


# --------------------------------------------------------------------------
def highlight_contours(img, contour) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resalta los contornos de la imagen y devuelve las imagen orignal
    con la bounding box dibujada y la imagen recortada.

    :param img: Imagen BGR a la que se le resaltan los contornos.
    :param contours: contorno a resaltar.
    :return: Imagen BGR con la bounding box dibujada y la imagen recortada.
    """

    # Se dibuja la bounding box
    x,y,w,h = cv.boundingRect(contour)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    # Se recorta la imagen
    img_recortada = img[y:y+h,x:x+w]

    return img, img_recortada


    
