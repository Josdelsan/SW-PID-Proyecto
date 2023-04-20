import cv2 as cv
import numpy as np
from pathlib import Path
from os import listdir

# ----------------------------------------------------
# Constantes
# ----------------------------------------------------
#IMAGE_DIRECTORY = Path().parent.absolute() / "images"
IMAGE_DIRECTORY = Path().parent.absolute() / "images2"

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
    h_sup_2=10

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

def reescalar_imagen(img):
    """
    Reescala la imagen para que se adapte a la pantalla
    """
    screen_width, screen_height = 1600, 900

    # Tamano de la pantalla
    height, width, _ = img.shape

    # Factor de escala
    width_scale = screen_width / width
    height_scale = screen_height / height
    scale = min(width_scale, height_scale)

    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_img = cv.resize(img, (new_width, new_height))

    return resized_img

def calcular_compacidad(contour):
    """
    Calcula la compacidad de un contorno
    """
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour,True)

    try:
        compacidad = 4*np.pi*area/perimeter**2
    except ZeroDivisionError:
        compacidad = 0

    return compacidad

def mostrar_imagenes(lista_imagenes):
    """
    Muestra una lista de imágenes en una ventana una a una
    """
    for tipo, img in lista_imagenes:
        img = reescalar_imagen(img)
        cv.imshow(f"{tipo}", img)
        cv.waitKey(0)


# ----------------------------------------------------
# SCRIPT
# ----------------------------------------------------

# Lista de imágenes en la carpeta
images = [cv.imread(str(IMAGE_DIRECTORY / image)) for image in listdir(IMAGE_DIRECTORY)]
senales_detectadas = []

for img in images:
    # Se detecta el contorno de la señal
    contours = detectar_contorno_señal(img)
    biggest_contour = max(contours, key=cv.contourArea)
    contours = [c for c in contours if cv.contourArea(c) > (cv.contourArea(biggest_contour)*0.4)]
    
    for contour in contours:
        img_copy = img.copy()

        # Calculo de la compacidad
        compacidad = calcular_compacidad(contour)

        if compacidad >= 0.63 and compacidad <= 0.9:
            # Rectangulo verde para contornos circulares
            bounding_color = (0,255,0)
            tipo = "circular"
        elif compacidad >= 0.55 and compacidad < 0.63:
            # Rectangulo azul para contornos triangulares
            bounding_color = (255,0,0)
            tipo = "triangular"
        else:
            print(f"Se ha detectado un contorno que no se puede clasificar en circular o triangular. Compacidad: {compacidad}")
            continue

        # Se dibuja el contorno en la imagen
        # cv.drawContours(img_copy, contour, -1, (0,255,0), 3)

        # Se obtiene la bounding box del contorno
        x,y,w,h = cv.boundingRect(contour)

        # Se dibuja la bounding box en la imagen
        cv.rectangle(img_copy,(x,y),(x+w,y+h),bounding_color,4)
        titulo = f"Es un contorno {tipo}. Compacidad: {compacidad}"
        senales_detectadas.append((titulo, img_copy))


mostrar_imagenes(senales_detectadas)
