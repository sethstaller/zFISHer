import os
import numpy as np

import cv2

import zFISHer.utils.config as cfg
import nd2
#from nd2reader import ND2Reader

"""
Generates MIPs of every channel of a given 4D stack
"""

# Convert nd2 to tiffstack
def convert_nd2_inputs() -> None:
    pass
def mip_stack(path,f_num) -> None:
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

    if f_num == 1 : 
        c_list = cfg.F1_C_LIST
        c_num = cfg.F1_C_NUM
        z_num = cfg.F1_Z_NUM
        ntag = cfg.F1_NTAG
        export_dir_dict = cfg.F1_C_MIP_DIR_DICT
    else:
        c_list = cfg.F2_C_LIST
        c_num = cfg.F2_C_NUM
        z_num = cfg.F2_Z_NUM
        ntag = cfg.F2_NTAG
        export_dir_dict = cfg.F2_C_MIP_DIR_DICT

    array = nd2.imread(path)
    print("Array shape (Z,C,Y,X):", array.shape)

    z_dim, c_dim, y_dim, x_dim = array.shape

        
    for c in range(c_dim):
        c_name = list(export_dir_dict.keys())[c]
        export_dir = list(export_dir_dict.values())[c]
        # Build a list of 2D frames (Y, X) for the given channel
        frames = [array[z_idx, c, :, :] for z_idx in range(array.shape[0])]
        frame_stack = np.stack(frames, axis=0)
        MIP_z_output = np.max(frame_stack, axis=0)
        MIP_z_raw_output = MIP_z_output
        output_img = MIP_z_raw_output
        output_name = os.path.join(f"{ntag}_{c_name}_MIP_.tif")          
        output_path = os.path.join(export_dir,output_name)
        cv2.imwrite(output_path,output_img)
    



def mip_stack_old(filepath,f_num) -> None:
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

    if f_num == 1 : 
        c_list = cfg.F1_C_LIST
        c_num = cfg.F1_C_NUM
        z_num = cfg.F1_Z_NUM
        ntag = cfg.F1_NTAG
        export_dir_dict = cfg.F1_C_MIP_DIR_DICT
    else:
        c_list = cfg.F2_C_LIST
        c_num = cfg.F2_C_NUM
        z_num = cfg.F2_Z_NUM
        ntag = cfg.F2_NTAG
        export_dir_dict = cfg.F2_C_MIP_DIR_DICT


    with ND2Reader(filepath) as nd2_file:
        nd2_file.iter_axes = 'z'
        for c in c_list:
            nd2_file.default_coords['c'] = c_list.index(c)
            export_dir = export_dir_dict[c]
            frames = [frame for frame in nd2_file]
            frame_stack = np.stack(frames, axis=0)
            MIP_z_output = np.max(frame_stack, axis=0)

            MIP_z_raw_output = MIP_z_output
            output_img = MIP_z_raw_output
            output_name = os.path.join(f"{ntag}_{c}_MIP_.tif")          
            output_path = os.path.join(export_dir,output_name)
            cv2.imwrite(output_path,output_img)


