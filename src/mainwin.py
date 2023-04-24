import subprocess
import sys

from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QPushButton, QLineEdit, QRadioButton, QMenu

QMenu.triggered
import os


class SignalStore(QObject):
    output = pyqtSignal(str)
    
    subprocess_over = pyqtSignal(int)


# 动态载入
class mainwindow(QMainWindow):
    # 自定义的类中包含一个 信号 成员
    signalStore = SignalStore()
    
    def __init__(self):
        super().__init__()
        # PyQt5
        self.ui = uic.loadUi("./UI/UI")
        # 这里与静态载入不同，使用 self.ui.show()
        # 如果使用 self.show(),会产生一个空白的 MainWindow
        
        self.custom_init()
        self.ui.show()
    
    def custom_init(self):
        
        self.ui.input_pushButton.clicked.connect(self.on_input_pushButton_clicked)
        self.ui.whisper_pushButton.clicked.connect(self.on_whisper_pushButton_clicked)
        self.ui.output_pushButton.clicked.connect(self.on_output_pushButton_clicked)
        self.ui.process_pushButton.clicked.connect(self.on_process_pushButton_clicked)
        
        self.signalStore.output.connect(self.printToTB)
        self.signalStore.subprocess_over.connect(self.process_over)
        
        self.ui.textBrowser.textChanged.connect(self.moveTextCurser)
        
        self.ui.actionAbout.triggered.connect(self.about_clicked)
    
    def about_clicked(self):
        QMessageBox.warning(self, "Code", "Based on https://github.com/ggerganov/whisper.cpp/blob/master/models/convert-pt-to-ggml.py \nGUI with PyQt5 Community Editon", QMessageBox.Yes, QMessageBox.Yes)
    
    def moveTextCurser(self):
        
        self.ui.textBrowser.moveCursor(QTextCursor.End)
    
    def on_input_pushButton_clicked(self):
        fileInput, filter = QFileDialog.getOpenFileName(self, "打开pt文件", "D:\WhisperModels", filter="All files(*.*);;OpenAI models(*.pt)")
        if fileInput == "":
            return
        
        self.ui.input_lineEdit.setText(fileInput)
        
        inputDir = os.path.dirname(fileInput)
        self.ui.output_lineEdit.setText(inputDir)
    
    def on_whisper_pushButton_clicked(self):
        fileInput, filter = QFileDialog.getOpenFileName(self, "选择python.exe文件", "D:\python39", filter="python.exe(*.exe)")
        if fileInput == "":
            return
        
        whisper_path = os.path.dirname(fileInput)
        whisper_path = whisper_path + "/Lib/site-packages"
        self.ui.whisper_lineEdit.setText(whisper_path)
    
    def on_output_pushButton_clicked(self):
        dirOutput = QFileDialog.getExistingDirectory(self, "选择python.exe文件", "D:\python39")
        if dirOutput == "":
            return
        
        self.ui.whisper_lineEdit.setText(dirOutput)
    
    def printToTB(self, text: str):
        self.ui.textBrowser.insertPlainText(text)
    
    def process_over(self, poll: int):
        if poll == 0:
            yes_No = QMessageBox.warning(self, "处理完毕", "处理结束！ 是否打开输出文件夹？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if yes_No == QMessageBox.No:
                return
            else:
                out_dir = self.ui.output_lineEdit.text()
                out_dir = "\\".join(out_dir.split("/"))
                
                commandLine = ["cmd", "/c", "explorer.exe", out_dir]
                
                print(" ".join(commandLine))
                res = subprocess.Popen(commandLine, creationflags=subprocess.CREATE_NO_WINDOW, text=True, stdout=subprocess.PIPE)
                # for line in res.stdout:
                #     print(line)
                #
                res.wait()
        
        if poll != 0:
            QMessageBox.warning(self, "错误", "处理出错，请检查输入文件及输出文件夹")
        
        self.changeChildrenEnabled()
    
    def on_process_pushButton_clicked(self):
        
        self.ui.textBrowser.setText("")
        
        print(sys.executable)
        
        commandLine = ["cmd", "/c", sys.executable, "./src/convert-pt-to-ggml.py"]
        
        fname_inp = self.ui.input_lineEdit.text()
        commandLine.append(fname_inp)
        
        dir_whisper = self.ui.whisper_lineEdit.text()
        commandLine.append(dir_whisper)
        
        dir_out = self.ui.output_lineEdit.text()
        commandLine.append(dir_out)
        
        if self.ui.radioButton_f16.isChecked():
            use_f16 = True
        
        # 使用f32位输出 添加最后一个参数
        elif self.ui.radioButton_f32.isChecked():
            use_f16 = False
            commandLine.append("1")
        
        print(" ".join(commandLine))
        self.changeChildrenEnabled()
        
        def call_process():
            subprocess_convert = subprocess.Popen(commandLine, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW, encoding="ANSI",
                                                  text=True)
            
            for line in subprocess_convert.stdout:
                self.signalStore.output.emit(line)
                print(line)
            
            subprocess_convert.wait()
            
            print(subprocess_convert.poll())
            self.signalStore.subprocess_over.emit(subprocess_convert.poll())
        
        from threading import Thread
        threa_1 = Thread(target=call_process, daemon=True)
        threa_1.start()
        
        # convert(fname_inp=fname_inp, dir_whisper=dir_whisper, dir_out=dir_out, use_f16=use_f16)
    
    def changeChildrenEnabled(self):
        
        buttons = self.ui.findChildren(QPushButton)
        
        for button in buttons:
            button.setEnabled((not (button.isEnabled())))
        
        LineEdits = self.ui.findChildren(QLineEdit)
        
        for LineEdit in LineEdits:
            LineEdit.setEnabled((not (LineEdit.isEnabled())))
        
        radioButtons = self.ui.findChildren(QRadioButton)
        
        for radioButton in radioButtons:
            radioButton.setEnabled((not (radioButton.isEnabled())))
