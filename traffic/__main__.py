# --------------------------------------------------------------------------
# Módulo: main
# Descripción: Módulo principal de la aplicación
# --------------------------------------------------------------------------

# Imports (standard, third party, local)
import sys

from PyQt5.QtWidgets import QApplication

from traffic.classification import Classification
from traffic import MODEL_DIRECTORY, LABELS_DIRECTORY
from traffic.main_window import MainWindow

if __name__ == "__main__":

    # Cargar el modelo y las etiquetas
    Classification.load_model(MODEL_DIRECTORY, LABELS_DIRECTORY)

    # Ejecutar la aplicación
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

    #app.run()

