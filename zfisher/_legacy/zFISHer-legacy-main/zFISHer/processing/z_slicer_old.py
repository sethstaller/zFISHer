import os

import cv2

import zFISHer.utils.config as cfg
#from nd2reader import ND2Reader

"""
oass
"""

# Convert nd2 to tiffstack
def convert_nd2_inputs() -> None:
    pass

def slice_stack(filepath,f_num) -> None:
    """
    From input file write each z-slice as a .tif into processing directory.

    Args:
    fpath (string): The filepath to open image.
    c_ind (int): The index position of the channel in the input file.
    c_name (string): The name of the channel in the input file.
    slice_count (int): Number of z-slices in the input file.
    f_ind (int): Indicates whether file is file 1 or file 2.
    ntag (string): User-defined name of the file.
    """
    if f_num == 1 : 
        c_list = cfg.F1_C_LIST
        c_num = cfg.F1_C_NUM
        z_num = cfg.F1_Z_NUM
        ntag = cfg.F1_NTAG
        export_dir_dict = cfg.F1_C_ZSLICE_DIR_DICT
    else:
        c_list = cfg.F2_C_LIST
        c_num = cfg.F2_C_NUM
        z_num = cfg.F2_Z_NUM
        ntag = cfg.F2_NTAG
        export_dir_dict = cfg.F2_C_ZSLICE_DIR_DICT

    with ND2Reader(filepath) as nd2_file:
        nd2_file.iter_axes = 'z'
        for c in c_list:
            nd2_file.default_coords['c'] = c_list.index(c)
            export_dir = export_dir_dict[c]
            for frame_index in range(z_num):
                z_level_data = nd2_file[frame_index]
                z_slice_img_output = z_level_data
                output_name = f"{ntag}_{c}_z{frame_index+1}.tif"
                output_path = os.path.join(export_dir, output_name)
                cv2.imwrite(output_path, z_slice_img_output)

