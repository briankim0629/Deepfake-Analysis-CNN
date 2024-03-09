from keras.models import load_model
from efficientnet.tfkeras import EfficientNetB4
import cv2
import numpy as np
import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
from PIL import Image
from PyQt5.QtCore import Qt

class ImageAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.image_path = None

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 400)

        self.open_button = QPushButton("Open Image", self)
        self.open_button.clicked.connect(self.open_image)

        self.analyze_button = QPushButton("Analyze", self)
        self.analyze_button.clicked.connect(self.analyze_image)
        self.analyze_button.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.open_button)
        layout.addWidget(self.analyze_button)

        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.setGeometry(200, 200, 450, 500)
        self.setWindowTitle("test")
        self.show()

    def open_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.image_path = file_name
            self.loaded_image = cv2.imread(file_name)
            self.load_image()
            self.analyze_button.setEnabled(True)

            # Extract the file name from the path
            _, file_extension = os.path.splitext(file_name)
            base_name = os.path.basename(file_name)
            save_path = f"test{file_extension}" 

            cv2.imwrite(save_path, self.loaded_image)
            print(f"Loaded image saved as {save_path}")

    def load_image(self):
        print(self.image_path)
        pixmap = QPixmap(self.image_path)
        pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def analyze_image(self):
        
        analysis_result = self.fake_analyze_image()
        self.show_analysis_result(analysis_result)

    def fake_analyze_image(self):
        return self.analysis(self.image_path)

    def show_analysis_result(self, result):
        self.analyze_button.setEnabled(False)
        self.image_label.clear()
        class_labels = ["Deepfake", "Real"]
        msg = str(class_labels[result[0]])
        self.show_message_box("Analysis Result", msg )
        

    def show_message_box(self, title, message):
        from PyQt5.QtWidgets import QMessageBox
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
    
    def create_datasets(self,fname, img_width, img_height):
        imgs = []
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_width, img_height))
        imgs.append(img)
            
        imgs = np.array(imgs, dtype='float32')
        return imgs

    def analysis(self,fname):
        model = load_model('trained_model.h5')
        IMG_HEIGHT = 256
        IMG_WIDTH = 256

        val_img = self.create_datasets(fname, IMG_WIDTH, IMG_HEIGHT)
        val_img = val_img/255.0

        preds = model.predict(val_img[0:1])
        idx = -1
        mx = -1
        num_elements = preds.shape[1]
        for i in range(num_elements):
            if mx < preds[0][i]:
                idx = i
                mx = preds[0][i]
        result = [idx,preds[0]]
        return result
    
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageAnalyzer()
    sys.exit(app.exec_())
