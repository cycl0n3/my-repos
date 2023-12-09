import numpy as np

import PySide6
from PySide6 import QtCore, QtWidgets, QtGui

import sys
import random

# Prints PySide6 version
print(PySide6.__version__)

# Prints the Qt version used to compile PySide6
print(PySide6.QtCore.__version__)

# seed 42
np.random.seed(42)

class Hexagon:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.size = radius

    def points(self):
        return [(self.x - self.size, self.y),
                (self.x - self.size / 2, self.y + self.size * np.sqrt(3) / 2),
                (self.x + self.size / 2, self.y + self.size * np.sqrt(3) / 2),
                (self.x + self.size, self.y),
                (self.x + self.size / 2, self.y - self.size * np.sqrt(3) / 2),
                (self.x - self.size / 2, self.y - self.size * np.sqrt(3) / 2)]
    
    def __repr__(self):
        return f"Hexagon(x={self.x}, y={self.y}, size={self.size})"
    
    def __str__(self):
        return f"Hexagon(x={self.x}, y={self.y}, size={self.size})"
    

hx = Hexagon(0, 0, 1)
print(hx)
print(hx.points())


class Polygon:
    def __init__(self, x, y, radius, sides):
        self.x = x
        self.y = y
        self.size = radius
        self.sides = sides

    def points(self):
        return [(self.x + self.size * np.cos(2 * np.pi * i / self.sides), self.y + self.size * np.sin(2 * np.pi * i / self.sides)) for i in range(self.sides)]
    
    def __repr__(self):
        return f"Polygon(x={self.x}, y={self.y}, size={self.size}, sides={self.sides})"
    
    def __str__(self):
        return f"Polygon(x={self.x}, y={self.y}, size={self.size}, sides={self.sides})"


pg = Polygon(0, 0, 1, 7)
print(pg)
print(pg.points())

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]

        self.button = QtWidgets.QPushButton("Click me!")
        self.text = QtWidgets.QLabel("Hello World",
                                     alignment=QtCore.Qt.AlignCenter)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.button)

        self.button.clicked.connect(self.magic)

    @QtCore.Slot()
    def magic(self):
        self.text.setText(random.choice(self.hello))
    

if __name__ == "__main__":
    QtWidgets.QApplication.setStyle("fusion")
    
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())

