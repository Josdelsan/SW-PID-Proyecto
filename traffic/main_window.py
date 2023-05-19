# --------------------------------------------------------------------------
# Módulo: main_window
# Descripción: Módulo de la ventana principal de la aplicación
# --------------------------------------------------------------------------

# Imports (standard, third party, local)
from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np

from traffic.utils import single_image_classification, predict_from_images_folder

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Predicción de señales de tráfico")
        self.setGeometry(300, 200, 400, 200)  # Adjust the window size and position

        layout = QVBoxLayout()

        # Boton para predecir una imagen, muestra solo el resultado
        self.button_predict2 = QPushButton("Predecir imagen", self)
        self.button_predict2.setFont(QFont("Arial", 10))  # Increase the font size
        self.button_predict2.clicked.connect(self.predict_image_result)
        layout.addWidget(self.button_predict2)

        # Boton para predecir una imagen, muestra el proceso
        self.button_predict = QPushButton("Predecir imagen (mostrar proceso)", self)
        self.button_predict.setFont(QFont("Arial", 10))  # Increase the font size
        self.button_predict.clicked.connect(self.predict_image)
        layout.addWidget(self.button_predict)

        # Boton para predecir todas las imagenes de ejemplos
        self.button_predict3 = QPushButton("Predecir imágenes de la carpeta de ejemplo", self)
        self.button_predict3.setFont(QFont("Arial", 10))  # Increase the font size
        self.button_predict3.clicked.connect(self.predict_example_images_result)
        layout.addWidget(self.button_predict3)

        # Boton para predecir todas las imagenes de ejemplos
        self.button_predict4 = QPushButton("Predecir imágenes de la carpeta de ejemplo (mostrar proceso)", self)
        self.button_predict4.setFont(QFont("Arial", 10))  # Increase the font size
        self.button_predict4.clicked.connect(self.predict_example_images)
        layout.addWidget(self.button_predict4)

        label_resultado = QLabel("Resultado:", self)
        label_resultado.setFont(QFont("Arial", 12))
        layout.addWidget(label_resultado)

        self.result_layout = QVBoxLayout()  # Use QVBoxLayout to display multiple elements vertically
        layout.addLayout(self.result_layout)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    # --------------------------------------------------------------------------
    # Acciones de los botones
    # --------------------------------------------------------------------------
    def predict_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")

        if file_path:
            # Call the image processing function from the module and get the result
            result_list = single_image_classification(file_path)

            # Display the result in the main window
            self.display_result_list(result_list)

    def predict_image_result(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")

        if file_path:
            # Call the image processing function from the module and get the result
            result_list = single_image_classification(file_path, show=False)

            # Display the result in the main window
            self.display_result_list(result_list)

    def predict_example_images(self):
        # Call the image processing function from the module and get the result
        result_list = predict_from_images_folder()

        # Display the result in the main window
        self.display_result_list(result_list)

    def predict_example_images_result(self):
        # Call the image processing function from the module and get the result
        result_list = predict_from_images_folder(show=False)

        # Display the result in the main window
        self.display_result_list(result_list)


    # --------------------------------------------------------------------------
    # Funciones auxiliares
    # --------------------------------------------------------------------------
    def display_result_list(self, result_list):
        # Clear the previous results
        self.clear_result_layout()

        for result_text, image in result_list:
            # Create QHBoxLayout for image and text
            result_layout = QHBoxLayout()

            # Create QLabel for image
            label_image = QLabel(self)
            label_image.setAlignment(Qt.AlignCenter)
            label_image.setFixedSize(90, 90)  # Set a fixed size for the image label

            image_bytes = np.ascontiguousarray(image).tobytes()
            qimage = QImage(image_bytes, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qimage)
            label_image.setPixmap(pixmap.scaled(label_image.size(), Qt.AspectRatioMode.KeepAspectRatio))

            result_layout.addWidget(label_image)

            # Create QLabel for text
            label_text = QLabel(result_text, self)
            label_text.setFont(QFont("Arial", 12))
            label_text.setAlignment(Qt.AlignCenter)
            result_layout.addWidget(label_text)

            self.result_layout.addLayout(result_layout)

    def clear_result_layout(self):
        # Remove all widgets from the result layout
        while self.result_layout.count():
            item = self.result_layout.takeAt(0)
            layout = item.layout()
            if layout:
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()