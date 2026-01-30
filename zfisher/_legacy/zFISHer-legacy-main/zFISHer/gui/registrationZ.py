import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
#from nd2reader import ND2Reader
#import zFISHer.config.config_manager as cfgmgr


'''This creates the gui that let's the user view and modify the slice offset of the two Z-stacks. '''

class RegistrationZGUI():
    def __init__(self, master, switch_to_gui_two, logger):
        print("Z REGISTRATION GUI INITIALIZED")
        self.logger = logger
        self.switch = switch_to_gui_two

        # Get the File 2 XY offset from previous step
        self.x_offset = cfgmgr.get_config_value("X_OFFSET")
        self.y_offset = cfgmgr.get_config_value("Y_OFFSET")    
        print(f"FILE 2 OFFSET = {self.x_offset},{self.y_offset}") 

        # Get directories information to access files
        self.output_dir = cfgmgr.get_config_value("OUTPUT_DIR")
        self.f1_cs = cfgmgr.get_config_value("FILE_1_CHANNELS")
        self.f2_cs = cfgmgr.get_config_value("FILE_2_CHANNELS")
        self.processing_dir = cfgmgr.get_config_value("PROCESSING_DIR")
        self.f1_ntag = cfgmgr.get_config_value("FILE_1_NAMETAG")
        self.f2_ntag = cfgmgr.get_config_value("FILE_2_NAMETAG")

        # Use DAPI channel as registration channel or other if specified
        self.f1_reg_c = self.get_reg_channel(self.f1_cs)
        self.logger.log_message(f"Channel 1 registration channel: INDEX:{self.f1_reg_c} - NAME:{self.f1_cs[self.f1_reg_c]}")
        self.f2_reg_c= self.get_reg_channel(self.f2_cs)
        self.logger.log_message(f"Channel 2 registration channel: INDEX:{self.f2_reg_c} - NAME:{self.f2_cs[self.f2_reg_c]}")

        # Get the stacks of Z Slices to align
        self.f1_zslices_sorted = []
        self.f2_zslices_sorted = []
        self.load_z_stacks()

        # Preprocess stacks with normalization, grayscale, and cropping
        self.f1_stack_n = self.normalize_stack(self.f1_zslices_sorted)
        self.f2_stack_n = self.normalize_stack(self.f2_zslices_sorted)
        self.f1_s_ng = self.grayscale_stack(self.f1_stack_n)
        self.f2_s_ng = self.grayscale_stack(self.f2_stack_n)
        self.f1_s_ngc = []
        self.f2_s_ngc = []
        self.f1_s_ngc, self.f2_s_ngc = self.crop_stacks(self.f1_s_ng,self.f2_s_ng,self.x_offset,self.y_offset)
        # Find middle slice of each stack to set as Z-offset = 0
        self.f1_midslice, self.f2_midslice = self.find_middles(self.f1_s_ngc,self.f2_s_ngc)

        # Generate GUI wndow
        self.generate_gui_window(master)
    
    def generate_gui_window(self, master):
        # Define Master Window (Overlay)
        self.master = master
        self.master.title("ZFISHER --- Registration")

        # Access height and width from the first image in the stack
        height, width = self.f1_s_ngc[0].shape

        # Now you can use height and width for further calculations
        self.hwmultiplier = height  # Use height from the first image
        self.screen_width = master.winfo_screenwidth() * 0.8
        self.screen_height = master.winfo_screenheight() * 0.8
        self.hwmult = width / height  # width to height ratio

        # Create canvas
        self.canvas = tk.Canvas(master, width=self.screen_height * self.hwmult, height=self.screen_height)
        self.canvas.grid(row=0, column=0, columnspan=5, pady=10)  # Use grid for canvas

        # Finish Button
        self.finish_button = tk.Button(self.master, text=f"Finalize {self.f2_ntag} (F2) Offset", pady=12, command=self.finalize_offset)
        self.finish_button.grid(row=1, column=0, columnspan=5, pady=5)  # Place button using grid

        self.master.update_idletasks()

    def finalize_offset(self):
      #  cfgmgr.set_config_value("Z_OFFSET",self.f2_offset_x_scaled)

         # Close the current window
        self.master.destroy()

        # Switch to the next GUI
        self.switch()

    def find_middles(self,f1_stack,f2_stack):

        f1_middle = int(len(f1_stack)/2)
        f2_middle = int(len(f2_stack)/2)

        print(f"f1_middle {f1_middle} - f2_middle {f2_middle}")

        return f1_middle, f2_middle

    def crop_stacks(self,f1stack_in,f2stack_in,x_offset,y_offset):
        tx = -x_offset
        ty = -y_offset

        # Get the dimensions of the first image in f1stack_in
        height, width = f1stack_in[0].shape

        # Create empty NumPy arrays with the same shape as the first image in f1stack_in
        ref = np.zeros((height, width), dtype=np.float32)  # or np.empty() depending on your needs
        mov = np.zeros((height, width), dtype=np.float32)

        # Get the dimensions of the images
        ref_height, ref_width = ref.shape[:2]
        mov_height, mov_width = mov.shape[:2]

        # Calculate the bounding box for the overlapping region
        start_x = max(0, int(tx))
        start_y = max(0, int(ty))
        end_x = min(ref_width, int(tx) + mov_width)
        end_y = min(ref_height, int(ty) + mov_height)

        # Ensure start and end coordinates are within bounds
        start_x_mov = max(0, -int(tx))
        start_y_mov = max(0, -int(ty))
        end_x_mov = end_x - int(tx)
        end_y_mov = end_y - int(ty)


        f1stack_out = []
        f2stack_out = []

        for slice in f1stack_in:
            slice = np.array(slice)
            cropped_slice = slice[start_y:end_y, start_x:end_x]
            f1stack_out.append(cropped_slice)

        for slice in f2stack_in:
            slice = np.array(slice)
            cropped_slice = slice[start_y_mov:end_y_mov, start_x_mov:end_x_mov]
            f2stack_out.append(cropped_slice) 

        print(f"OFFSET: {x_offset},{y_offset}")
        print(f"INPUT f1stack length {len(f1stack_in)} OUTPUT: {len(f1stack_out)}")
        print(f"INPUT f2stack length {len(f2stack_in)} OUTPUT: {len(f2stack_out)}")

        print(f"IN SIZE f1 {f1stack_in[0].size} (width,height) OUT {f1stack_out[0].shape} (height,width)")
        print(f"IN SIZE f2 {f2stack_in[0].size} (width,height) OUT {f2stack_out[0].shape} (height,width)")

        return f1stack_out,f2stack_out
    
    def normalize_stack(self,stack):
        """
        Normalize a z-stack to have values between 0 and 1.
        """

        print(stack)
        
        if isinstance(stack, list):
            stack = np.array(stack)
        stack = stack.astype(np.float32)
        stack -= np.min(stack)
        stack /= np.max(stack) if np.max(stack) > 0 else 1
        return stack
    
        if isinstance(stack, list):
            stack = np.array(stack)
        stack = stack.astype(np.float32)
        newstack = []
        for slice in stack:
            slice -= np.min(slice)
            slice /= np.max(slice) if np.max(slice) > 0 else 1
            #slice[slice < 0.2] = 0
            #slice[slice >= 0.8] = 1
            newstack.append(slice)
        return newstack

    def grayscale_img(self,image):
    # Convert PIL Image to numpy array
        image_array = np.array(image)
        
        # Check if the image has 3 dimensions (indicating an RGB image)
        if image_array.ndim == 3:
            # Convert RGB to grayscale by averaging the channels
            image_array = np.mean(image_array, axis=2)
        
        return image_array  
    
    def grayscale_stack(self,stack):
        gs_stack = []
        for slice in stack:
            slice = self.grayscale_img(slice)
            gs_stack.append(slice)
        return gs_stack
    
    def get_reg_channel(self, channels):
        for index,channel in enumerate(channels):
            if channel == "DAPI":
                return index

    def slice_sort_key(self, filename):
        # Extract the number between "z" and "_.tif" in the filename
        start_index = filename.find('z') + 1
        end_index = filename.find('_.tif')
        number_str = filename[start_index:end_index]
        # Convert the extracted number string to an integer
        return int(number_str)
    
    def load_z_stacks(self):
        print("sart")
        f1_zslices_dir = os.path.join(self.processing_dir,self.f1_ntag,self.f1_cs[self.f1_reg_c],"z_slices")
        f2_zslices_dir = os.path.join(self.processing_dir,self.f2_ntag,self.f2_cs[self.f2_reg_c],"z_slices")
        print(f1_zslices_dir)
        print(f2_zslices_dir)
        print("fin")
        # Load the f1_slices into an array
        slices = os.listdir(f1_zslices_dir)
        sorted_slices = sorted(slices, key=self.slice_sort_key)
        print(sorted_slices)
        for s in sorted_slices:
            path = os.path.join(f1_zslices_dir,s)
            sliceimage = Image.open(path)
            print(s)
            self.f1_zslices_sorted.append(sliceimage)
            print(self.f1_zslices_sorted)

        # Load the f2_slices into an array
        slices = os.listdir(f2_zslices_dir)
        sorted_slices = sorted(slices, key=self.slice_sort_key)
        print(sorted_slices)
        for s in sorted_slices:
            path = os.path.join(f2_zslices_dir,s)
            sliceimage = Image.open(path)
            self.f2_zslices_sorted.append(sliceimage)

        
        print(f"FINAL + {self.f1_zslices_sorted[0]}")
        print(f"FINAL + {self.f1_zslices_sorted[-1]}")
        print(f"FINAL + {self.f2_zslices_sorted[0]}")
        print(f"FINAL + {self.f2_zslices_sorted[-1]}")