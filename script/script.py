import cv2 as cv
import numpy as np
from pathlib import Path
from os import listdir

# ----------------------------------------------------
# Constantes
# ----------------------------------------------------
IMAGE_DIRECTORY = Path().parent.absolute() / "images"

# ----------------------------------------------------
# Funciones
# ----------------------------------------------------
def detectar_contorno_señal(img):
    # ------------------------------------------------
    # Contorno de las señales rojo
    # ------------------------------------------------

    # Se define el rango de colores a detectar
    h_inf_1=170
    h_sup_1=180
    h_inf_2=0
    h_sup_2=5

    lower_red_1 = np.array([h_inf_1,50,50])
    upper_red_1 = np.array([h_sup_1,255,255])
    lower_red_2 = np.array([h_inf_2,50,50])
    upper_red_2 = np.array([h_sup_2,255,255])

    # Se convierte la imagen a HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    red_mask_1 = cv.inRange(hsv, lower_red_1, upper_red_1)
    red_mask_2 = cv.inRange(hsv, lower_red_2, upper_red_2)
    red_mask = red_mask_1+red_mask_2

    # Se aplica una mascara para eliminar todo lo que no sea rojo
    img_rojo = cv.bitwise_and(img,img, mask= red_mask)

    # ------------------------------------------------
    # Eliminación de ruido y binarización
    # ------------------------------------------------

    # Se convierte la imagen a escala de grises
    gray = cv.cvtColor(img_rojo, cv.COLOR_BGR2GRAY)
    binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]

    # Se eliminan el ruido mediante una erosión
    forms = cv.erode(binary, np.ones((3,3),np.uint8), iterations = 1)

    # ------------------------------------------------
    # Contorno de la señal
    # ------------------------------------------------
    contours = cv.findContours(forms, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #biggest_contour = max(contours[0], key=cv.contourArea)

    return contours[0]


# ----------------------------------------------------
# SCRIPT
# ----------------------------------------------------

# Lista de imágenes en la carpeta
images = [cv.imread(str(IMAGE_DIRECTORY / image)) for image in listdir(IMAGE_DIRECTORY)]
lista_circulares = []
lista_triangulares = []

for img in images:
    # Se detecta el contorno de la señal
    contours = detectar_contorno_señal(img)
    biggest_contour = max(contours, key=cv.contourArea)
    contours = [c for c in contours if cv.contourArea(c) > (cv.contourArea(biggest_contour)*0.4)]
    
    for contour in contours:
        img_copy = img.copy()
        # Se dibuja la boundingbox del contorno en la imagen
        x,y,w,h = cv.boundingRect(contour)
        cv.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0),2)

        # Calculo de la compacidad
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour,True)

        try:
            compacidad = 4*np.pi*area/perimeter**2
        except ZeroDivisionError:
            compacidad = 0

        if compacidad >= 0.63 and compacidad <= 0.9:
            lista_circulares.append(img)
            print(f"Es un contorno circular. Compacidad: {compacidad}")
        elif compacidad >= 0.5 and compacidad < 0.63:
            lista_circulares.append(img)
            print(f"Es un contorno triangular. Compacidad: {compacidad}")
        else:
            print(f"No es un contorno circular o triangular. Compacidad: {compacidad}")


        # Se muestra la imagen
        cv.imshow("imagen", img_copy)
        cv.waitKey(0)

