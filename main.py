import logging
from PyQt5.QtWidgets import QApplication
import sys

from App import ManagerApp


logging.basicConfig(level=logging.INFO)

app = QApplication(sys.argv)
ex = ManagerApp()
sys.exit(app.exec_())

