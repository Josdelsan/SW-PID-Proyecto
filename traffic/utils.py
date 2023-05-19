# --------------------------------------------------------------------------
# Módulo: app
# Descripción: Módulo principal de la aplicación
# --------------------------------------------------------------------------

# Imports (standard, third party, local)
from traffic.classification import Classification
from traffic import images
from traffic import detection

from traffic import IMAGE_DIRECTORY

def predict_from_images_folder(show=True):
    """Función principal de la aplicación"""
    output = []

    # Cargar las imagenes desde la carpeta
    images_list = images.load_images(IMAGE_DIRECTORY)

    # Para cada imagen
    for img, title in images_list:
        # Mostrar la imagen
        if show:
            images.show_image(img, title)

        # Aplicar mascara de deteccion de rojo
        img_rojo = detection.apply_red_mask(img)
        # Mostrar la imagen
        if show:
            images.show_image(img_rojo, "Mascara de rojo")

        # Aplicar binarización y erosion
        img_bin = detection.apply_binary_and_erode(img_rojo)
        # Mostrar la imagen
        if show:
            images.show_image(img_bin, "Binarización y erosión")

        # Aplicar detección de contornos
        contours = detection.detect_contours(img_bin)
        for c in contours:
            img_copy = img.copy()
            # Contorno resaltado y recorte
            img_contour, img_cropped = detection.highlight_contours(img_copy, c)

            # Clasificar la imagen
            pred_class, pred_prob = Classification.predict(img_cropped)
            # Mostrar la imagen
            text = f"Imagen: {title} | Clase predecida: {pred_class} | Probabilidad: {pred_prob}"
            if show:
                images.show_image(img_contour, text)

            cropped = detection.crop_image(img, [c])[0]
            output.append((text, cropped))

    return output

def single_image_classification(path, show=True):
    output = []

    # Cargar imagen
    img= images.load_image(path)

    # Mostrar la imagen
    if show:
        images.show_image(img, "Imagen original")

    # Aplicar mascara de deteccion de rojo
    img_rojo = detection.apply_red_mask(img)
    # Mostrar la imagen
    if show:
        images.show_image(img_rojo, "Mascara de rojo")

    # Aplicar binarización y erosion
    img_bin = detection.apply_binary_and_erode(img_rojo)
    # Mostrar la imagen
    if show:
        images.show_image(img_bin, "Binarización y erosión")

    # Aplicar detección de contornos
    contours = detection.detect_contours(img_bin)
    for c in contours:
        img_copy = img.copy()
        # Contorno resaltado y recorte
        img_contour, img_cropped = detection.highlight_contours(img_copy, c)

        # Clasificar la imagen
        pred_class, pred_prob = Classification.predict(img_cropped)
        # Mostrar la imagen
        text = f"Clase: {pred_class} | Probabilidad: {pred_prob}\n"
        cropped = detection.crop_image(img, [c])[0]
        if show:
            images.show_image(img_contour, text)

        output.append((text, cropped))

    return output
    