
import tkinter as tk
from zFISHer.processing.write_excel import WriteExcel


class FinishGUI(tk.Frame):
    def __init__(self, master, switch_to_gui_two, logger): 
        self.switch = switch_to_gui_two
        self.logger = logger




        print("fINISHED~~~~!!!!!!")

        writeexcel = WriteExcel(logger)

    def complete_everything(self):
        self.switch()