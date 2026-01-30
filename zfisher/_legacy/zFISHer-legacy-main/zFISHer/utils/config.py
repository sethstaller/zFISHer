import os
from zFISHer import version

"""
Holds all data to be referenced and updated across modules.
WARNING: EDIT THIS FILE AT YOUR OWN PERIL.
"""

# Version of the zFISHer software
VERSION = version

# Created directory paths for reference
BASE_DIR = None
OUTPUT_DIR = None
PROCESSING_DIR = None
SEG_PROCESSING_DIR = None # segmentation folder to store analysis images
LOGS_DIR = None

F1_PROCESSING_DIR = None
F2_PROCESSING_DIR = None

F1_C_DIR_LIST = [] # folder inside each file's processing directory named by channel
F2_C_DIR_LIST = []

F1_C_MIP_DIR_DICT = {}  # key channel name (string) : value is path (string)
F2_C_MIP_DIR_DICT = {}

F1_C_ZSLICE_DIR_DICT = {} # key channel name (string) : value is path (string)
F2_C_ZSLICE_DIR_DICT = {}

# Input XYZ file paths 
F1_PATH = None
F2_PATH = None

# Displayed and saved nickname of input file
F1_NTAG = None
F2_NTAG = None

# List of channel names (string)
F1_C_LIST = None
F2_C_LIST = None

# Number of channels
F1_C_NUM = None
F2_C_NUM = None

# Number of z-slices
F1_Z_NUM = None
F2_Z_NUM = None

# Channel to use for XYZ registration
F1_REG_C = None
F2_REG_C = None

# Channel to use for nuclei segmentation
F1_SEG_C = None
F2_SEG_C = None

# Offset of File 2 relative to File 1 XYZ
OFFSET_X = None # in pixels
OFFSET_Y = None # in pixels
OFFSET_Z = None # in z-slices

# Nuclei segmentation algorithm script path
#NUC_SEG_ALGO_PATH = "zFISHer/processing/segmentation/zFISHer_basic_nucseg.py"
SEG_ALGO_DIR = None # Directory housing the nuclei segmentation scripts
NUC_SEG_ALGO_PATH = None
NUC_SEG_DEFAULT_SCRIPT = "zFISHer_basic_nucseg.py" # change if you put another script in segmentation folder that you want to use as autosegmentation when workflow module initializes

# Nuclei segmentation finalization
SEG_NUC_COUNT = None    # Number of segmented nuclei
SEG_NUC_POLYGONS = []   # array contain [index, i, x, y] index is polygon id, i is coord index within a polygon id

# Random colocalization value
RAND_COLOC_PERCENT = None   # During calculations how many ROI pairs between two files randomly colocalize when File 2 ROI XYZ is rotated 90deg XY

# Channel Metadata
CH_METADATA = None

# Capture Images path
CAPTURE_IMG_DIR = None # path to folder for images captured by user during punctapick