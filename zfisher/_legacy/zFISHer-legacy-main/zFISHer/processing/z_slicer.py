import os
import cv2
import numpy as np
import nd2
import zFISHer.utils.config as cfg

def convert_nd2_inputs() -> None:
    pass


def slice_stack(path, f_num):
    if f_num == 1: 
        c_list = cfg.F1_C_LIST
        z_num = cfg.F1_Z_NUM
        ntag = cfg.F1_NTAG
        export_dir_dict = cfg.F1_C_ZSLICE_DIR_DICT
    else:
        c_list = cfg.F2_C_LIST
        z_num = cfg.F2_Z_NUM
        ntag = cfg.F2_NTAG
        export_dir_dict = cfg.F2_C_ZSLICE_DIR_DICT

    array = nd2.imread(path)
    print("Array shape (Z,C,Y,X):", array.shape)

    z_dim, c_dim, y_dim, x_dim = array.shape

        
    for c in range(c_dim):
        c_name = list(export_dir_dict.keys())[c]
        for z in range(z_dim):
            export_dir = list(export_dir_dict.values())[c]
            slice_2d = array[z, c, :, :]
            filename = f"{ntag}_{c_name}_z{z+1}.tif"
            filepath = os.path.join(export_dir, filename)


            cv2.imwrite(filepath, slice_2d.astype(np.uint16))  # assuming 16-bit depth
            print(slice_2d.dtype)
            print(f"Saved {filepath}")


def slice_stack_old(filepath, f_num) -> None:
    """
    From input file write each z-slice as a .tif into processing directory.
    Uses `nd2` library instead of `nd2reader`.

    Args:
    filepath (str): Path to the ND2 file.
    f_num (int): 1 for base file, 2 for moving file.
    """
    if f_num == 1: 
        c_list = cfg.F1_C_LIST
        z_num = cfg.F1_Z_NUM
        ntag = cfg.F1_NTAG
        export_dir_dict = cfg.F1_C_ZSLICE_DIR_DICT
    else:
        c_list = cfg.F2_C_LIST
        z_num = cfg.F2_Z_NUM
        ntag = cfg.F2_NTAG
        export_dir_dict = cfg.F2_C_ZSLICE_DIR_DICT

    with nd2.ND2File(filepath) as nd2_file:
        metadata = nd2_file.metadata
        channel_names = metadata.channels

        # Make sure we're working with a Z-stack
        if "z" not in nd2_file.sizes:
            raise ValueError("ND2 file is not a Z-stack.")

        for c in c_list:
            if c not in channel_names:
                raise ValueError(f"Channel '{c}' not found in file. Available channels: {channel_names}")

            c_index = channel_names.index(c)
            export_dir = export_dir_dict[c]

            for z in range(z_num):
                # Access by (T, C, Z, Y, X) or similar depending on file structure
                frame = nd2_file.get_frame_2D(c=c_index, z=z)
                output_name = f"{ntag}_{c}_z{z + 1}.tif"
                output_path = os.path.join(export_dir, output_name)
                cv2.imwrite(output_path, frame.astype(np.uint16))  # assuming 16-bit depth
