import os
import logging
import shutil
from datetime import datetime


import zFISHer.utils.config as cfg
#import zFISHer.config.config_manager as cfgmgr  #TODO REMOVE

"""
Make directories for pipeline processing and sets their paths for reference in the config file.
"""

def create_processing_directories() -> None:
    """
    Create the directories based on nametags inside the processing directory.
    Store in config for reference.
    """

    # Update status label
    #self.status_l.config(text="Creating processing directories...")
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
    # Define capture image (during puncta picking) directory
        cfg.CAPTURE_IMG_DIR = os.path.join(cfg.PROCESSING_DIR,"IMAGE_CAPTURE")
        
    make_processing_directories()


def set_core_dir_values() -> None:
    """
    Set the primary directories for reference by app modules.
    """
    
    # These are the directories to generate analysis files
    BASE_DIR = os.getenv("BASE_DIR",os.path.abspath(os.getcwd()))
    OUTPUT_DIR = os.getenv("OUTPUT_DIR",os.path.join(BASE_DIR, "OUTPUT"))
    PROCESSING_DIR = os.getenv("PROCESSING_DIR",os.path.join(BASE_DIR, OUTPUT_DIR, "processing"))
    LOGS_DIR = os.getenv("LOGS_DIR",os.path.join(BASE_DIR, OUTPUT_DIR, "logs"))



    # This sets the path to the segmentation directory
    # Get absolute path to the directory where the current file lives
    CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up to the zFISHer module root
    MODULE_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, ".."))
    # Then build path to processing/segmentation
    SEG_ALGO_DIR = os.path.join(MODULE_ROOT, "processing", "segmentation")

    # Store path values in config
    cfg.BASE_DIR = BASE_DIR
    cfg.OUTPUT_DIR = OUTPUT_DIR
    cfg.PROCESSING_DIR = PROCESSING_DIR
    cfg.LOGS_DIR = LOGS_DIR
    cfg.SEG_ALGO_DIR = SEG_ALGO_DIR
    #cfgmgr.set_config_value("BASE_DIR",BASE_DIR)    #TODO REMOVE
    #cfgmgr.set_config_value("OUTPUT_DIR",OUTPUT_DIR)#TODO REMOVE
    #cfgmgr.set_config_value("PROCESSING_DIR",PROCESSING_DIR)#TODO REMOVE
    #cfgmgr.set_config_value("LOGS_DIR",LOGS_DIR)#TODO REMOVE


def make_output_directories(self):
    """
    Creates core directories that the analysis pipeline will populate.
    """

    '''
    os.makedirs(cfgmgr.get_config_value("OUTPUT_DIR"), exist_ok=True)
    os.makedirs(cfgmgr.get_config_value("PROCESSING_DIR"), exist_ok=True)
    os.makedirs(cfgmgr.get_config_value("LOGS_DIR"), exist_ok=True)
    '''
    pass


def make_processing_directories():
    """
    Creates each file's channel directories in the processing folder.
    """

    # Empties Output directory if for some reason it already exists.
    clear_output_dir()

    # Create directories for file 1 and file 2
    os.makedirs(cfg.F1_PROCESSING_DIR, exist_ok=True)
    os.makedirs(cfg.F2_PROCESSING_DIR, exist_ok=True)       
    
    # Create directories for file 1 and file 2 channels
    for dir in cfg.F1_C_DIR_LIST:
        os.makedirs(dir, exist_ok=True)
    for dir in cfg.F2_C_DIR_LIST:
        os.makedirs(dir, exist_ok=True)

    # Create z-slice directories per channel
    for dir in cfg.F1_C_ZSLICE_DIR_DICT.values():
         os.makedirs(dir, exist_ok=True)
    for dir in cfg.F2_C_ZSLICE_DIR_DICT.values():
         os.makedirs(dir, exist_ok=True)

    # Create MIP directories per channel
    for dir in cfg.F1_C_MIP_DIR_DICT.values():
         os.makedirs(dir, exist_ok=True)
    for dir in cfg.F2_C_MIP_DIR_DICT.values():
         os.makedirs(dir, exist_ok=True)

    # Create processing segmentation directory
    os.makedirs(cfg.SEG_PROCESSING_DIR, exist_ok=True)

    # Create image capture directory
    os.makedirs(cfg.CAPTURE_IMG_DIR, exist_ok=True)


def clear_output_dir():
    """
    If cfg.OUTPUT_DIR exists, delete all contents inside it (files and folders),
    but do NOT delete the OUTPUT_DIR folder itself.
    """
    output_dir = cfg.OUTPUT_DIR
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        # List all items in the directory
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # remove file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # remove directory and contents
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")