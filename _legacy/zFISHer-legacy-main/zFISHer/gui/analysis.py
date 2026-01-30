import tkinter as tk
import os
from PIL import Image, ImageTk
#import zFISHer.config.config_manager as cfgmgr
import zFISHer.data.parameters as aparams
import cv2
import numpy as np
import math
import platform
from tkinter import colorchooser
import random

'''This GUI gives stats from the previous steps and lets the user define the analysis parameters prior to data processing'''


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
class AnalysisGUI(tk.Frame):
    def __init__(self, master, switch_to_gui_two, logger):  # Ensure it takes an argument
        self.logger = logger
        self.switch = switch_to_gui_two

        # Pulling all data to display analysis stats in the GUI
        self.kpchan_kpnuc_xbundle = aparams.kpchan_kpnuc_xbundle 
        self.kpchan_ROIradius_xbundle = aparams.kpchan_ROIradius_xbundle
        self.kpchan_coloc_xbundle = aparams.kpchan_coloc_xbundle
        self.kpchan_analysistoggle_xbundle = aparams.kpchan_analysistoggle_xbundle
        self.arrows_xbundle = aparams.arrows_xbundle

        # Build GUI window
        self.master = master
        self.master.title("zFISHer -- Analyis")

        # File 1 nametag
        self.f1nametaglabel = tk.Label(master, text="File 1 Nametag:")
        self.f1nametaglabel.grid(row=3, column=0, sticky="e", padx=10, pady=5)

        self.f1nametagentry = tk.Entry(master, width=50)
        self.f1nametagentry.grid(row=3, column=1, sticky="ew", padx=10, pady=5)
        self.f1nametagentry.insert(0, "F1")

        # File 2 nametag
        self.f2nametaglabel = tk.Label(master, text="File 2 Nametag:")
        self.f2nametaglabel.grid(row=4, column=0, sticky="e", padx=10, pady=5)

        self.f2nametagentry = tk.Entry(master, width=50)
        self.f2nametagentry.grid(row=4, column=1, sticky="ew", padx=10, pady=5)
        self.f2nametagentry.insert(0, "F2")

        self.make_finish_button()

    def make_finish_button(self):
        self.finish_button = tk.Button(self.master, text="!!!--Finalize ROIs--!!!", command=self.finish_analysis)
        self.finish_button.grid(row=500, column=0, columnspan=3)   


    def finish_analysis(self):
        self.switch()