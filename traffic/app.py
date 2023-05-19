# --------------------------------------------------------------------------
# Módulo: app
# Descripción: Módulo principal de la aplicación
# --------------------------------------------------------------------------

# Imports (standard, third party, local)
from traffic.classification import Classification
from traffic import images
from traffic import detection

from traffic import IMAGE_DIRECTORY, MODEL_DIRECTORY, LABELS_DIRECTORY

def run():
    """Función principal de la aplicación"""

    # Cargar el modelo y las etiquetas
    Classification.load_model(MODEL_DIRECTORY, LABELS_DIRECTORY)

    # Cargar las imagenes desde la carpeta
    images_list = images.load_images(IMAGE_DIRECTORY)

    # Para cada imagen
    for img, title in images_list:
        # Mostrar la imagen
        images.show_image(img, title)

        # Aplicar mascara de deteccion de rojo
        img_rojo = detection.apply_red_mask(img)
        # Mostrar la imagen
        images.show_image(img_rojo, "Mascara de rojo")

        # Aplicar binarización y erosion
        img_bin = detection.apply_binary_and_erode(img_rojo)
        # Mostrar la imagen
        images.show_image(img_bin, "Binarización y erosión")

        # Aplicar detección de contornos
        contours = detection.detect_contours(img_bin)
        for c in contours:
            # Contorno resaltado y recorte
            img_contour, img_cropped = detection.highlight_contours(img, c)

            # Clasificar la imagen
            pred_class, pred_prob = Classification.predict(img_cropped)
            # Mostrar la imagen
            images.show_image(img_contour, f"Imagen: {title} | Clase predecida: {pred_class} | Probabilidad: {pred_prob}")

    