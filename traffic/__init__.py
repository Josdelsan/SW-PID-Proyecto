# Archivo de inicialización del paquete principal

from pathlib import Path

# App path
APP_PATH = Path(__file__).parent
IMAGE_DIRECTORY = APP_PATH / "resources/images"
MODEL_DIRECTORY = APP_PATH / "resources/traffic_signs.h5"
LABELS_DIRECTORY = APP_PATH / "resources/label_encoder.pkl"
