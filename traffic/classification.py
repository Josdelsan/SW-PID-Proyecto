# --------------------------------------------------------------------------
# Módulo: classification
# Descripción: Módulo de clasificación de señales de tráfico
#              mediante redes neuronales.
# --------------------------------------------------------------------------

# Imports (standard, third party, local)
from typing import Tuple

import tensorflow.keras.models as kmodel
import joblib
import cv2 as cv
import numpy as np

class Classification():

    model = None
    labels = None

    @staticmethod
    def load_model(model_path: str, labels_path: str) -> None:
        """
        Carga el modelo y las etiquetas de clasificación.

        :param model_path: Ruta al modelo
        :param labels_path: Ruta a las etiquetas
        """
        # Cargar modelo
        Classification.model = kmodel.load_model(model_path)

        # Cargar label encoder
        Classification.labels = joblib.load(labels_path)

    @staticmethod
    def predict(img) -> Tuple[str, str]:
        """
        Realiza la predicción de la imagen.

        :param img: Imagen a predecir
        :return: Etiqueta de la predicción y probabilidad
        """
        img_copy = img.copy()

        img_copy = cv.resize(img_copy, (64,64))
        img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
        img_copy = np.array(img_copy)
        img_copy = img_copy.reshape(1,64,64,1)

        # Predicción
        predictions = Classification.model.predict(img_copy)

        # Obtener la etiqueta de la predicción y la probabilidad
        predicted_class_index = np.argmax(predictions)
        predicted_prob = predictions[0][predicted_class_index]
        predicted_class_label = Classification.labels.inverse_transform([predicted_class_index])

        return predicted_class_label, predicted_prob