from PyQt5.QtWidgets import (QWidget,QLCDNumber,QSlider,QMainWindow,
                             QGridLayout,QApplication,QPushButton, QLabel, QLineEdit)

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtCore import Qt
from all import fatigue_detection
from PIL import Image




class Ui_example(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QGridLayout(self)
        self.label_image = QLabel(self)
        self.label_predict_result = QLabel('识别结果', self)
        self.label_predict_result.setStyleSheet("color:red")
        self.label_predict_result_display = QLabel(self)

        self.label_predict_acc = QLabel('识别准确率', self)
        self.label_predict_acc_display = QLabel(self)

        # self.label_predict_result.setStyleSheet("background-color:gold")
        self.button_search_image = QPushButton('选择图片', self)
        self.button_run = QPushButton('运行', self)
        self.setLayout(self.layout)
        self.initUi()

    def initUi(self):

        self.layout.addWidget(self.label_image, 1, 1, 3, 2)
        self.layout.addWidget(self.button_search_image, 1, 3, 1, 2)
        self.layout.addWidget(self.button_run, 3, 3,  1, 2)
        self.layout.addWidget(self.label_predict_result, 4, 3, 1, 1)
        self.layout.addWidget(self.label_predict_result_display, 4, 4, 1, 1)
        self.layout.addWidget(self.label_predict_acc, 5, 3, 1, 1)
        self.layout.addWidget(self.label_predict_acc_display, 5, 4, 1, 1)


        self.button_search_image.clicked.connect(self.openimage)
        self.button_run.clicked.connect(self.run)

        self.setGeometry(300, 300, 300, 300)
        self.setWindowTitle('低光环境下疲劳驾驶检测')
        self.show()

    def openimage(self):
        global fname
        imgName, imgType = QFileDialog.getOpenFileName(self, "选择图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QPixmap(imgName).scaled(self.label_image.width(), self.label_image.height())
        self.label_image.setPixmap(jpg)
        fname = imgName



    def run(self):
        global fname
        file_name = str(fname)
        img = Image.open(file_name)

        a, b = fatigue_detection(img)
        self.label_predict_result_display.setText(a)
        self.label_predict_acc_display.setText(str(b))




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Ui_example()
    sys.exit(app.exec_())