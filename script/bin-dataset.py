import cv2 as cv
import numpy as np
from pathlib import Path
from os import listdir

# ----------------------------------------------------
# Constantes
# ----------------------------------------------------
IMAGE_DIRECTORY = Path().parent.absolute() / "images-dataset"
# ----------------------------------------------------
# Funciones
# ----------------------------------------------------
def to_gray_scale(img):

    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_scale = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)[1]

    return gray_scale

images = [cv.imread(str(IMAGE_DIRECTORY / image)) for image in listdir(IMAGE_DIRECTORY)]
senales_gray = []

for image in images:
    gray_scale = to_gray_scale(image)
    senales_gray.append(gray_scale)



def mostrar_imagenes(lista_imagenes):
    for img in lista_imagenes:
        cv.imshow("Imagen", img)
        cv.waitKey(0)

mostrar_imagenes(senales_gray)   
    