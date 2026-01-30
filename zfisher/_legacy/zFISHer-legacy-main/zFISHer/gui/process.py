import os
import numpy as np
import PIL
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import cv2
#from nd2reader import ND2Reader

#import zFISHer.config.config_manager as cfgmgr #TODO REMOVE
import zFISHer.utils.makedir as mkdir
import zFISHer.utils.config as cfg
import zFISHer.processing.z_slicer as zslicer
import zFISHer.processing.mip_maker as mipmaker

import queue
import threading

"""
Displays window of file processing of both input files.
"""


class ProcessingGUI():
    """
    Window that allows user to input the the two XYZ stacks and displays file information.
    """


    def __init__(self, master, switch_to_new_gui,logger):
        """
        Initialize the window for user.
        """

        self.logger = logger
        self.master = master
        self.switch = switch_to_new_gui

        self.setup_window(master)
        #self.create_processing_directories()
        #Populate z-slice directories
        self.status_l.config(text="Slicing Z stack of all channels...")
        self.master.update()
        self.create_z_slices()
        #Populate MIP directories
        self.status_l.config(text="Making MIPs of all channels...")
        self.master.update()
        self.create_mips()
        #Finish processing
        self.status_l.config(text="Finished processing...")
        self.master.update()
        self.finish_processing()
        

    def setup_window(self,master) -> None:
        """
        Builds the GUI window and populates with widgets.

        Args:
        master (tkinter.Toplevel) : frame passed by GUImanager to build GUI.
        """

        # Window title
        self.master.title("zFISHer --- Input Processing")
        #self.master.update_idletasks()

        # Header label
        self.status_l = tk.Label(master, text="Nd2 Files are Being Processed...")
        self.status_l.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # Modifiable Directory Names
        self.inputs_directory_folder = "Inputs"
        self.inputs_directory_F1 = "FILE_1"
        self.inputs_directory_F2 = "FILE_2"
        self.processing_directory_folder = "Processing"
        self.p_MIP_directory = "MIP"
        self.p_MIP_raw_directory = "RAW_MIP"
        self.p_MIP_masked_directory = "Masked"
        self.p_zslices_directory = "zslices"
        self.outputs_directory_folder = "Outputs"

        # File Metadata
        self.f1_c_num = "?"
        self.f2_c_num = "?"
        self.f1_ntag = "name"
        self.f2_ntag = "name"
        #self.f1_path = cfgmgr.get_config_value("FILE_1_PATH")  # replace with actual path
        #self.f2_path = cfgmgr.get_config_value("FILE_2_PATH")  # replace with actual path
        self.f1_path = cfg.F1_PATH  # replace with actual path
        self.f2_path = cfg.F2_PATH  # replace with actual path
        self.f1_z_num = None  # replace with actual number
        self.f2_z_num = None  # replace with actual number
        self.f1channels = ["ch1", "ch2"]  # replace with actual channels
        self.f2channels = ["ch1", "ch2"]  # replace with actual channels

        #self.master.update_idletasks()


    def create_processing_directories(self) -> None:
        """
        Create the directories based on nametags inside the processing directory.
        Store in config for reference.
        """

        # Update status label
        self.status_l.config(text="Creating processing directories...")
        #self.master.update_idletasks()

        # Pull config info to build directories
        f1_ntag = cfg.F1_NTAG
        f2_ntag = cfg.F2_NTAG
        processing_dir = cfg.PROCESSING_DIR
        f1_c_list = cfg.F1_C_LIST
        f2_c_list = cfg.F2_C_LIST

        # Define file processing directories based on nametags inside the processing directory
        f1_processing_dir = os.path.join(processing_dir, f1_ntag)
        f2_processing_dir = os.path.join(processing_dir, f2_ntag)
        cfg.F1_PROCESSING_DIR = f1_processing_dir
        cfg.F2_PROCESSING_DIR = f2_processing_dir

        # Define file MIP and Z-SLICE processing directories
        for c in f1_c_list:
            c_dir = os.path.join(f1_processing_dir, c)
            c_zslice_dir = os.path.join(c_dir,"Z_SLICES")
            c_mip_dir = os.path.join(c_dir,"MIP")
            cfg.F1_C_DIR_LIST.append(c_dir)
            cfg.F1_C_ZSLICE_DIR_DICT[c] = c_zslice_dir
            cfg.F1_C_MIP_DIR_DICT[c] = c_mip_dir
        for c in f2_c_list:
            c_dir = os.path.join(f2_processing_dir, c)
            c_zslice_dir = os.path.join(c_dir,"Z_SLICES")
            c_mip_dir = os.path.join(c_dir,"MIP")
            cfg.F2_C_DIR_LIST.append(c_dir)
            cfg.F2_C_ZSLICE_DIR_DICT[c] = c_zslice_dir
            cfg.F2_C_MIP_DIR_DICT[c] = c_mip_dir

        # Define segmentation processing folder
            cfg.SEG_PROCESSING_DIR = os.path.join(cfg.PROCESSING_DIR,"SEGMENTATION")
        # Define segmentation algorithm directory
        #    cfg.SEG_ALGO_DIR = os.path.join(cfg.BASE_DIR,"/zFISHer/processing/segmentation")
            
        mkdir.make_processing_directories()

    
    def create_z_slices(self) -> None:
        """
        Extract image capture metadata info from each file for analysis.
        """

        # Update status label
        self.status_l.config(text="Slicing Z stack of file 1")
        self.master.update()
        #
        zslicer.slice_stack(cfg.F1_PATH,1)

        self.status_l.config(text="Slicing Z stack of file 2")
        self.master.update()
        zslicer.slice_stack(cfg.F2_PATH,2)


    def create_mips(self):
        """
        Iterates through each channel of both input files to make MIPs.
        Generates a MIP of input file z-stack.

        Args: 
        file () :
        c_index :
        cname
        numslices
        find
        ntag
        """

        # Update status label
        self.status_l.config(text="Generating MIPs of all channels")
        self.master.update()
  
        #
        mipmaker.mip_stack(cfg.F1_PATH,1)
        mipmaker.mip_stack(cfg.F2_PATH,2)


            
    def continue_button(self):
        self.finishprocessing()


    def finish_processing(self):
        """
        Triggers when all all slices and MIPs generated and saved.
        """

        self.master.destroy()
        self.switch()
        
