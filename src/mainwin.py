import base64
import io
import os
import struct
import subprocess

import numpy as np
import torch
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QPushButton, QLineEdit, QRadioButton


class SignalStore(QObject):
    output = pyqtSignal(str)
    
    subprocess_over = pyqtSignal(int)


# 动态载入
class mainwindow(QMainWindow):
    # 自定义的类中包含一个 信号 成员
    signalStore = SignalStore()
    python_exe: str = ""
    
    def __init__(self):
        super().__init__()
        # PyQt5
        self.ui = uic.loadUi("../UI/UI")
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
        QMessageBox.warning(self, "Code", "Based on https://github.com/ggerganov/whisper.cpp/blob/master/models/convert-pt-to-ggml.py \nGUI with PyQt5 Community Editon",
                            QMessageBox.Yes, QMessageBox.Yes)
    
    def moveTextCurser(self):
        
        self.ui.textBrowser.moveCursor(QTextCursor.End)
    
    def on_input_pushButton_clicked(self):
        fileInput, filter = QFileDialog.getOpenFileName(self, "打开pt文件", "D:\\", filter="All files(*.*);;OpenAI models(*.pt)")
        if fileInput == "":
            return
        
        self.ui.input_lineEdit.setText(fileInput)
        
        inputDir = os.path.dirname(fileInput)
        self.ui.output_lineEdit.setText(inputDir)
    
    def on_whisper_pushButton_clicked(self):
        fileInput, filter = QFileDialog.getOpenFileName(self, "选择whisper目录下的__init__.py文件", "./", filter="__init__.py(*.py)")
        if fileInput == "":
            return
        
        
        whisper_path = os.path.dirname(fileInput)
        whisper_path = os.path.dirname(whisper_path)
        # whisper_path = whisper_path + "/Lib/site-packages"
        self.ui.whisper_lineEdit.setText(whisper_path)
    
    def on_output_pushButton_clicked(self):
        dirOutput = QFileDialog.getExistingDirectory(self, "选择python.exe文件", "D:\\")
        if dirOutput == "":
            return
        
        self.ui.output_lineEdit.setText(dirOutput)
    
    def printToTB(self, text: str):
        # self.ui.textBrowser.insertPlainText(text)
        self.ui.textBrowser.append(text)
    
    def process_over(self, poll: int):
        if poll == 0:
            yes_No = QMessageBox.warning(self, "处理完毕", "处理结束！ 是否打开输出文件夹？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if yes_No == QMessageBox.Yes:

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
        fname_inp = self.ui.input_lineEdit.text()
        dir_whisper = self.ui.whisper_lineEdit.text()
        dir_out = self.ui.output_lineEdit.text()
        
        if self.ui.radioButton_f16.isChecked():
            use_f16 = True
        
        # 使用f32位输出 添加最后一个参数
        elif self.ui.radioButton_f32.isChecked():
            use_f16 = False
            
            
        #
        # print(sys.executable)
        
        # commandLine = ["cmd", "/c", self.python_exe, "./convert-pt-to-ggml.py"]
        #
        # commandLine.append(fname_inp)
        #
        
        # commandLine.append(dir_whisper)
        #
        
        # commandLine.append(dir_out)
        #

        #    commandLine.append("1")
        #
        # print(" ".join(commandLine))
        
        #
        # def call_process():
        #     subprocess_convert = subprocess.Popen(commandLine, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW, encoding="ANSI",
        #                                           text=True)
        #
        #     for line in subprocess_convert.stdout:
        #         self.signalStore.output.emit(line)
        #         print(line)
        #
        #     subprocess_convert.wait()
        #
        #     print(subprocess_convert.poll())
        #     self.signalStore.subprocess_over.emit(subprocess_convert.poll())
        

        def call_process_2():
            res = self.convert(fname_inp=fname_inp, dir_whisper=dir_whisper, dir_out=dir_out, use_f16=use_f16)
            self.signalStore.subprocess_over.emit(res)
        
        self.changeChildrenEnabled()
        from threading import Thread
        threa_1 = Thread(target=call_process_2, daemon=True)
        threa_1.start()
        
        
    
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
    
    # from transformers import GPTJForCausalLM
    # from transformers import GPT2TokenizerFast
    
    # ref: https://github.com/openai/whisper/blob/8cf36f3508c9acd341a45eb2364239a3d81458b9/whisper/tokenizer.py#L10-L110
    # LANGUAGES = {
    #    "en": "english",
    #    "zh": "chinese",
    #    "de": "german",
    #    "es": "spanish",
    #    "ru": "russian",
    #    "ko": "korean",
    #    "fr": "french",
    #    "ja": "japanese",
    #    "pt": "portuguese",
    #    "tr": "turkish",
    #    "pl": "polish",
    #    "ca": "catalan",
    #    "nl": "dutch",
    #    "ar": "arabic",
    #    "sv": "swedish",
    #    "it": "italian",
    #    "id": "indonesian",
    #    "hi": "hindi",
    #    "fi": "finnish",
    #    "vi": "vietnamese",
    #    "iw": "hebrew",
    #    "uk": "ukrainian",
    #    "el": "greek",
    #    "ms": "malay",
    #    "cs": "czech",
    #    "ro": "romanian",
    #    "da": "danish",
    #    "hu": "hungarian",
    #    "ta": "tamil",
    #    "no": "norwegian",
    #    "th": "thai",
    #    "ur": "urdu",
    #    "hr": "croatian",
    #    "bg": "bulgarian",
    #    "lt": "lithuanian",
    #    "la": "latin",
    #    "mi": "maori",
    #    "ml": "malayalam",
    #    "cy": "welsh",
    #    "sk": "slovak",
    #    "te": "telugu",
    #    "fa": "persian",
    #    "lv": "latvian",
    #    "bn": "bengali",
    #    "sr": "serbian",
    #    "az": "azerbaijani",
    #    "sl": "slovenian",
    #    "kn": "kannada",
    #    "et": "estonian",
    #    "mk": "macedonian",
    #    "br": "breton",
    #    "eu": "basque",
    #    "is": "icelandic",
    #    "hy": "armenian",
    #    "ne": "nepali",
    #    "mn": "mongolian",
    #    "bs": "bosnian",
    #    "kk": "kazakh",
    #    "sq": "albanian",
    #    "sw": "swahili",
    #    "gl": "galician",
    #    "mr": "marathi",
    #    "pa": "punjabi",
    #    "si": "sinhala",
    #    "km": "khmer",
    #    "sn": "shona",
    #    "yo": "yoruba",
    #    "so": "somali",
    #    "af": "afrikaans",
    #    "oc": "occitan",
    #    "ka": "georgian",
    #    "be": "belarusian",
    #    "tg": "tajik",
    #    "sd": "sindhi",
    #    "gu": "gujarati",
    #    "am": "amharic",
    #    "yi": "yiddish",
    #    "lo": "lao",
    #    "uz": "uzbek",
    #    "fo": "faroese",
    #    "ht": "haitian creole",
    #    "ps": "pashto",
    #    "tk": "turkmen",
    #    "nn": "nynorsk",
    #    "mt": "maltese",
    #    "sa": "sanskrit",
    #    "lb": "luxembourgish",
    #    "my": "myanmar",
    #    "bo": "tibetan",
    #    "tl": "tagalog",
    #    "mg": "malagasy",
    #    "as": "assamese",
    #    "tt": "tatar",
    #    "haw": "hawaiian",
    #    "ln": "lingala",
    #    "ha": "hausa",
    #    "ba": "bashkir",
    #    "jw": "javanese",
    #    "su": "sundanese",
    # }
    
    ## ref: https://github.com/openai/whisper/blob/8cf36f3508c9acd341a45eb2364239a3d81458b9/whisper/tokenizer.py#L273-L292
    # def build_tokenizer(path_to_whisper_repo: str, name: str = "gpt2"):
    #    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #    path = os.path.join(path_to_whisper_repo, "whisper/assets", name)
    #    tokenizer = GPT2TokenizerFast.from_pretrained(path)
    #
    #    specials = [
    #        "<|startoftranscript|>",
    #        *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
    #        "<|translate|>",
    #        "<|transcribe|>",
    #        "<|startoflm|>",
    #        "<|startofprev|>",
    #        "<|nocaptions|>",
    #        "<|notimestamps|>",
    #    ]
    #
    #    tokenizer.add_special_tokens(dict(additional_special_tokens=specials))
    #    return tokenizer
    
    # ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
    
    # if len(sys.argv) < 4:
    #     print("Usage: convert-pt-to-ggml.py model.pt path-to-whisper-repo dir-output [use-f32]\n")
    #     sys.exit(1)
    
    #
    # fname_inp   = sys.argv[1]
    # dir_whisper = sys.argv[2]
    # dir_out     = sys.argv[3]
    
    def convert(self, fname_inp: str, dir_whisper: str, dir_out: str, use_f16=True):
        def bytes_to_unicode():
            """
            Returns list of utf-8 byte and a corresponding list of unicode strings.
            The reversible bpe codes work on unicode strings.
            This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
            When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
            This is a signficant percentage of your normal, say, 32K bpe vocab.
            To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
            And avoids mapping to whitespace/control characters the bpe code barfs on.
            """
            bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
            cs = bs[:]
            n = 0
            for b in range(2 ** 8):
                if b not in bs:
                    bs.append(b)
                    cs.append(2 ** 8 + n)
                    n += 1
            cs = [chr(n) for n in cs]
            return dict(zip(bs, cs))
        
        # try to load PyTorch binary data
        try:
            model_bytes = open(fname_inp, "rb").read()
            with io.BytesIO(model_bytes) as fp:
                checkpoint = torch.load(fp, map_location="cpu")
        except:
            self.signalStore.output.emit("Error: failed to load PyTorch model file: %s" % fname_inp)
            return 2
        
        hparams = checkpoint["dims"]
        self.signalStore.output.emit("hparams:" + str(hparams))
        
        list_vars = checkpoint["model_state_dict"]
        
        # print(list_vars['encoder.positional_embedding'])
        # print(list_vars['encoder.conv1.weight'])
        # print(list_vars['encoder.conv1.weight'].shape)
        
        # load mel filters
        n_mels = hparams["n_mels"]
        with np.load(os.path.join(dir_whisper, "whisper/assets", "mel_filters.npz")) as f:
            filters = torch.from_numpy(f[f"mel_{n_mels}"])
            # print (filters)
        
        # code.interact(local=locals())
        
        multilingual = hparams["n_vocab"] == 51865
        tokenizer = os.path.join(dir_whisper, "whisper/assets", multilingual and "multilingual.tiktoken" or "gpt2.tiktoken")
        
        # output in the same directory as the model
        
        f_l = fname_inp.split("\\")
        
        if len(f_l) == 1:
            f_l = fname_inp.split("/")
        
        f_l = f_l[-1]
        
        f_wothOutExt = f_l.split(".")[:-1]
        
        f_out = "-".join(f_wothOutExt)
        
        with open(tokenizer, "rb") as f:
            contents = f.read()
            tokens = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in contents.splitlines() if line)}
        
        # use 16-bit or 32-bit floats
        if use_f16 == False:
            fname_out = dir_out + "/ggml-model-Whisper-" + f_out + "-f32.bin"
        else:
            fname_out = dir_out + "/ggml-model-Whisper-" + f_out + ".bin"
        
        fout = open(fname_out, "wb")
        
        fout.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex
        fout.write(struct.pack("i", hparams["n_vocab"]))
        fout.write(struct.pack("i", hparams["n_audio_ctx"]))
        fout.write(struct.pack("i", hparams["n_audio_state"]))
        fout.write(struct.pack("i", hparams["n_audio_head"]))
        fout.write(struct.pack("i", hparams["n_audio_layer"]))
        fout.write(struct.pack("i", hparams["n_text_ctx"]))
        fout.write(struct.pack("i", hparams["n_text_state"]))
        fout.write(struct.pack("i", hparams["n_text_head"]))
        fout.write(struct.pack("i", hparams["n_text_layer"]))
        fout.write(struct.pack("i", hparams["n_mels"]))
        fout.write(struct.pack("i", use_f16))
        
        # write mel filters
        fout.write(struct.pack("i", filters.shape[0]))
        fout.write(struct.pack("i", filters.shape[1]))
        for i in range(filters.shape[0]):
            for j in range(filters.shape[1]):
                fout.write(struct.pack("f", filters[i][j]))
        
        byte_encoder = bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}
        
        fout.write(struct.pack("i", len(tokens)))
        
        for key in tokens:
            fout.write(struct.pack("i", len(key)))
            fout.write(key)
        
        for name in list_vars.keys():
            data = list_vars[name].squeeze().numpy()
            self.signalStore.output.emit("Processing variable: " + name + " with shape: " + str(data.shape))
            
            # reshape conv bias from [n] to [n, 1]
            if name == "encoder.conv1.bias" or \
                    name == "encoder.conv2.bias":
                data = data.reshape(data.shape[0], 1)
                self.signalStore.output.emit("  Reshaped variable: " + name + " to shape: " + str(data.shape))
            
            n_dims = len(data.shape);
            
            # looks like the whisper models are in f16 by default
            # so we need to convert the small tensors to f32 until we fully support f16 in ggml
            # ftype == 0 -> float32, ftype == 1 -> float16
            ftype = 1;
            if use_f16:
                if n_dims < 2 or \
                        name == "encoder.conv1.bias" or \
                        name == "encoder.conv2.bias" or \
                        name == "encoder.positional_embedding" or \
                        name == "decoder.positional_embedding":
                    self.signalStore.output.emit("  Converting to float32")
                    data = data.astype(np.float32)
                    ftype = 0
            else:
                data = data.astype(np.float32)
                ftype = 0
            
            # if name.startswith("encoder"):
            #    if name.endswith("mlp.0.weight") or \
            #       name.endswith("mlp.2.weight"):
            #        print("  Transposing")
            #        data = data.transpose()
            
            # header
            str_name = name.encode('utf-8')
            fout.write(struct.pack("iii", n_dims, len(str_name), ftype))
            for i in range(n_dims):
                fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
            fout.write(str_name);
            
            # data
            data.tofile(fout)
        
        fout.close()
        
        self.signalStore.output.emit("Done. Output file: " + fname_out)
        self.signalStore.output.emit("")
        return 0
    
    """
    https://github.com/ggerganov/whisper.cpp/blob/master/models/convert-pt-to-ggml.py
    """
