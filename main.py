from PyQt5.QtWidgets import QApplication
from  src.mainwin import mainwindow
import  sys



if __name__ == "__main__":
    App = QApplication(sys.argv)
    mainWin = mainwindow()
    sys.exit(App.exec_())
