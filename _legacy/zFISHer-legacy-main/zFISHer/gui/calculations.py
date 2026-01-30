
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from datetime import datetime
import queue
import os

#import zFISHer.config.config_manager as cfgmgr
import zFISHer.data.parameters as aparams
import zFISHer.utils.config as cfg

import math
import numpy as np

''' 
This monitors the calculation of the max intensity xyz pixel of each channel ROI.
Then it generates pairs of every ROI.
Then it calculates the max of
'''
class CalculationsGUI(tk.Frame):
    def __init__(self, master, switch_to_gui_two, logger): 
        # Set up input arguments for the GUI.
        self.master = master
        self.switch = switch_to_gui_two
        self.logger = logger

    
        self.initialize_window(self.master)
        self.get_data()
        self.load_z_slices()
        self.create_full_ROI_list()
        self.max_z_finder()
        self.generate_ROI_pairs()
        self.coloc_parser()
        self.random_coloc_calculation()
    
    def initialize_window(self, master):
        master.title("ZFISHER --- Calculations")
        master.geometry("500x500")
        master.config(bg="lightblue")
        master.grid_rowconfigure(0, weight=1)
        master.grid_rowconfigure(1, weight=1)
        master.grid_rowconfigure(2, weight=1)
        master.grid_rowconfigure(3, weight=1)
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, weight=1)
        master.grid_columnconfigure(3, weight=1)

        self.titlelabel = tk.Label(master, text="Performing Calculations... ", background="black", foreground="white", font=("Helvetica", 20))
        self.titlelabel.grid(row=0, column=1, sticky="nsew", padx=0, pady=20, columnspan=2)


        self.starttime = str(datetime.now())
        self.endtime = "???"


        self.starttime_label = tk.Label(master, text=f"Start Time: {self.starttime}")
        self.starttime_label.grid(row=1, column=1, sticky="nsew", padx=0, pady=2, columnspan=2)
        self.endtime_label = tk.Label(master, text=f"End Time: {self.endtime}")
        self.endtime_label.grid(row=2, column=1, sticky="nsew", padx=0, pady=2, columnspan=2)        

        #PROGRESS BAR
        self.progress1 = tk.IntVar()
        self.progress_label1 = tk.Label(master, text="PROGRESS: 0%")
        self.progress_label1.grid(row=3, column=1, sticky="nsew")
        self.progressbar1 = ttk.Progressbar(master, orient="horizontal", length=200, mode="determinate", variable=self.progress1)
        self.progressbar1.grid(row=3, column=2, sticky="nsew")

        # Create a queue for progress updates
        self.progress_queue = queue.Queue()
    def get_data(self):
        self.x_offset = cfg.OFFSET_X
        self.y_offset = cfg.OFFSET_Y
        self.z_offset = cfg.OFFSET_Z
        self.f2_offset = [cfg.OFFSET_X,cfg.OFFSET_Y]
        # Pulling all data to display analysis stats in the GUI
        self.kpchan_kpnuc_xbundle = aparams.kpchan_kpnuc_xbundle #for bundle of each channel kp, an array = [kpID,kpOvalID,kp_x,kp_y,nucInd,polyID,polyTextID]
        self.kpchan_ROIradius_xbundle = aparams.kpchan_ROIradius_xbundle  # need to get in picking
        self.kpchan_coloc_xbundle = aparams.kpchan_coloc_xbundle # need to get in picking

        self.f1_full_exp_arr = aparams.f1_full_exp_arr #file_id,channel,chanrad,chan_e,kpID,kpOvalID,f_kp_x,f_kp_y,nucInd,polyID,polyTextID
        self.f2_full_exp_arr = aparams.f2_full_exp_arr

        #self.kpchan_analysistoggle_xbundle = aparams.kpchan_analysistoggle_xbundle
        self.arrows_xbundle = aparams.arrows_xbundle


        # Get directories information to access files
        self.output_dir = cfg.OUTPUT_DIR
        self.f1_cs = cfg.F1_C_LIST
        self.f2_cs = cfg.F2_C_LIST
        self.processing_dir = cfg.PROCESSING_DIR
        self.f1_ntag = cfg.F1_NTAG
        self.f2_ntag = cfg.F2_NTAG


        self.mpp = 0.108333333333333     # This many microns per pixel
        self.ppm = 1/self.mpp                 # This is pixels per 1 um
        self.z_spacing = 0.1
    def slice_sort_key(self, filename):
        # Extract the number between "z" and "_.tif" in the filename
        start_index = filename.find('z') + 1
        end_index = filename.find('.tif')
        number_str = filename[start_index:end_index]
        # Convert the extracted number string to an integer
        return int(number_str)
    def load_z_slices(self):
        # For each channel load all slices, sort, and put into file-channel:slice dictionary.
        # Channel: Sorted Z-slice array
        self.f1_z_slice_dict = {}
        self.f2_z_slice_dict = {}

        # Populate the File 1 dictionary
        for i,c in enumerate(self.f1_cs):
            c_zslice_arr = []
            f1_zslices_dir = os.path.join(self.processing_dir,self.f1_ntag,c,"Z_SLICES")
            slices = os.listdir(f1_zslices_dir)
            sorted_slices = sorted(slices, key=self.slice_sort_key)
            for s in sorted_slices:
                path = os.path.join(f1_zslices_dir,s)
                sliceimage = Image.open(path)
                c_zslice_arr.append(sliceimage)
            self.f1_z_slice_dict[c] = c_zslice_arr

        # Populate the File 2 dictionary
        for i,c in enumerate(self.f2_cs):
            c_zslice_arr = []
            f2_zslices_dir = os.path.join(self.processing_dir,self.f2_ntag,c,"Z_SLICES")
            slices = os.listdir(f2_zslices_dir)
            sorted_slices = sorted(slices, key=self.slice_sort_key)
            for s in sorted_slices:
                path = os.path.join(f2_zslices_dir,s)
                sliceimage = Image.open(path)
                c_zslice_arr.append(sliceimage)
            self.f2_z_slice_dict[c] = c_zslice_arr
    def create_full_ROI_list(self):
        print("GENERATE FULL ROI LIST")
        print(self.f1_full_exp_arr)
        print(self.f2_full_exp_arr)
        self.full_ROI_arr = []
        for i,chanarr in enumerate(self.f1_full_exp_arr):
            for row in chanarr:
                self.full_ROI_arr.append(row)
        for i,chanarr in enumerate(self.f2_full_exp_arr):
            for row in chanarr:
                self.full_ROI_arr.append(row)     
    
        print("FULL ROIS")
    def generate_ROI_pairs(self):
        '''
        Prepares an array from all ROIs that will be checked for colocalization
        '''
        print("ROI pairs")
        self.pairs_array = []
        
        # Iterate through all pairs of rows
        for i in range(len(self.kpchan_kpnucxyz_xbundle )):
            for j in range(i + 1, len(self.kpchan_kpnucxyz_xbundle )):  # j starts from i+1 to avoid pairing the same row with itself
                self.pairs_array.append((self.kpchan_kpnucxyz_xbundle [i], self.kpchan_kpnucxyz_xbundle [j]))  # Append the pair as a tuple

        print(len(self.pairs_array))
    def max_z_finder(self):
        #self.full_ROI_arr = []
        #[file_id,channel,chanrad,chan_e,kpID,kpOvalID,f_kp_x,f_kp_y,nucInd,polyID,polyTextID]
        kpchannucbundle = self.full_ROI_arr

        self.titlelabel.configure(text="Finding max intensity of each ROI.")
        self.master.config(bg="red")
        self.master.update_idletasks()
        # Calculate total number of keypoints for the progress bar
        total_keypoints = sum(len(chan_i) for chan_i in kpchannucbundle)
        processed_keypoints = 0

        print("CLACULAINg MAX Zs")
        self.kpchan_kpnucxyz_xbundle = [] #[file_id,channel,chanrad,chan_e,kpID,kpOvalID,f_kp_x,f_kp_y,nucInd,polyID,polyTextID,max_intensity_pixel]
        print(kpchannucbundle)
        for index, chan_i in enumerate(kpchannucbundle):
            #chanbundle = []
            print("HERE")
            row = chan_i
            print(f"ROW {row}")
            file_id,channel,chanrad,chan_e,kpID,kpOvalID,kp_x,kp_y,nucInd,polyID,polyTextID = row#[kpID,kpOvalID,kp_x,kp_y,nucInd,polyID,polyTextID]
            chanroi_radius = chanrad
            pixelradius = math.ceil(chanroi_radius * self.ppm)
            radius = pixelradius
            center = int(kp_x), int(kp_y)

            if file_id == 1:
                chan_slices = self.f1_z_slice_dict[channel]
            elif file_id == 2:
                chan_slices = self.f2_z_slice_dict[channel]


            max_intensity =-1
            max_slice_index = ''
            for sliceindex, slice in enumerate(chan_slices, start=1):
                print(f"slice ind {sliceindex}")
                gray_image_slice = np.array(chan_slices[sliceindex-1])
                y, x = np.ogrid[:gray_image_slice.shape[0], :gray_image_slice.shape[1]]
                mask_circle = (x - center[0]) ** 2 + (y - center[1]) ** 2 > radius ** 2
                gray_image_slice[mask_circle] = 0  # Set the area outside the circle to 0
                roi = gray_image_slice
                
                max_value = np.max(roi)

                print(f" SLICE {sliceindex} - {max_value}")

                if max_value > max_intensity:
                    max_slice_index = sliceindex
                    max_intensity = max_value
                print(f"max intensity slice {max_slice_index}")
                print(f"max intensity slice counter {max_intensity}")     
    
            temprow = file_id,channel,chanrad,chan_e,kpID,kpOvalID,kp_x,kp_y,nucInd,polyID,polyTextID, max_slice_index, max_intensity
            self.kpchan_kpnucxyz_xbundle.append(temprow) 
            # Update progress
            processed_keypoints += 1
            progress_percent = (processed_keypoints / total_keypoints) * 100
            self.progress1.set(progress_percent)
            self.progress_label1.config(text=f"MAX INTENSITY ROI PROGRESS: {int(progress_percent)}%")
            self.master.update_idletasks()

        print(len(self.kpchan_kpnucxyz_xbundle ))
        aparams.kpchan_kpnucxyz_xbundle = self.kpchan_kpnucxyz_xbundle 
    def same_file_check(self,r1_name,r2_name):
        r1_f = -1
        r2_f = -2
        same_file = None

        # Get file of roi_1
        if r1_name == 1:
            r1_f = 1
        elif r1_name == 2:
            r1_f = 2

        # Get file of roi_2
        if r2_name == 1:
            r2_f = 1
        elif r2_name == 2:
            r2_f = 2

        # Check if same file
        if r1_f == r2_f:
            same_file = True
        elif r1_f != r2_f:
            same_file = False
        
        return r1_f,r2_f,same_file
    def determine_r2_xy(self,r2_x_in, r2_y_in, r1_c, r2_c, same_file, f2_offset_x, f2_offset_y):
        if same_file:
                x2 = r2_x_in
                y2 = r2_y_in
        elif not same_file:
                if r1_c == 1 and r2_c == 2:
                    x2 = r2_x_in + f2_offset_x
                    y2 = r2_y_in + f2_offset_y
                if r1_c == 2 and r2_c == 1:
                    x2 = r2_x_in - f2_offset_x
                    y2 = r2_y_in - f2_offset_y
        return x2, y2
    def calculate_z_offset(self,f1_mid, f2_mid):
        if f1_mid == f2_mid:
            z_offset = 0
        elif f1_mid > f2_mid:
            z_offset = f1_mid - f2_mid
        elif f1_mid < f2_mid:
            z_offset = -(f2_mid - f1_mid)
        return z_offset
    def coloc_parser(self):
            ### def reprocess_ROIs(all_ROIs_arr, f2_offset_x, f2_offset_y, f1_middle_Zslice, f2_middle_Zslice, c_coloc_dict):
            # Pair all the ROIs into a new array [ROI_1_info, ROI_2_info]
            #delete # [ROI #, Nuclei #, Channel, X_pos, Y_pos, Z_slice, Max Intensity Pixel Value (0-65536)]
            # delete#[file_id,channel,chanrad,chan_e,kpID,kpOvalID,f_kp_x,f_kp_y,nucInd,polyID,polyTextID]
            # delete#[file_id,channel,chanrad,chan_e,kpID,kpOvalID,kp_x,kp_y,nucInd,polyID,polyTextID, max_slice_index, max_intensity
            ROI_pairs = self.pairs_array
            f2_offset_x = self.x_offset
            f2_offset_y = self.y_offset
            f1_middle_Zslice = int(cfg.F1_Z_NUM/2)
            f2_middle_Zslice = int(cfg.F2_Z_NUM/2)

            yes_arr = []
            no_arr = []

            for row in ROI_pairs:
                # Get each ROI info of the pair
                r1 = row[0]
                r2 = row[1]
                
                # Determine if it is the same file or different files
                r1_file, r2_file, same_file = self.same_file_check(r1[0],r2[0])

                # Get the X Y positions of each ROI
                #file_id,channel,chanrad,chan_e,kpID,kpOvalID,kp_x,kp_y,nucInd,polyID,polyTextID, max_slice_index, max_intensity
                x1 = r1[6]
                y1 = r1[7]


                x2,y2 = self.determine_r2_xy(r2[6], r2[7], r1_file, r2_file, same_file,f2_offset_x, f2_offset_y)

                # Determine the x and y distance delta in microns
                x_delta = (x2 - x1) * self.mpp
                y_delta = (y2 - y1) * self.mpp


                # Determine the Z slice delta between the two stacks
                z1 = abs(r1[11])
                z2 = abs(r2[11])
                zslice_offset = self.calculate_z_offset(f1_middle_Zslice, f2_middle_Zslice)
                z_delta = abs(r1[11] - (r2[11]+zslice_offset)) * self.z_spacing

                # Calculate distance
                distance = math.sqrt((x_delta)**2 + (y_delta)**2 + (z_delta)**2)

                print(f"x1 {x1}")
                print(f"x2 {x2}")
                print(f"xdelta {x_delta}")
                print(f"y1 {y1}")
                print(f"y2 {y2}")
                print(f"ydelta {y_delta}")
                print(f"z1 {z1}")
                print(f"z2 {z2}")
                print(f"zdelta {z_delta}")
                print(distance)

                # Determine if it is a colocalization
                f1_coloc_cutoff = r1[3]
                f2_coloc_cutoff = r2[3]

                coloc_cutoff = min(f1_coloc_cutoff,f2_coloc_cutoff)

                if distance <= coloc_cutoff:
                    is_coloc = True
                    print(f"COLOC FOUND - {distance}")
                elif distance > coloc_cutoff:
                    is_coloc = False

                # Gather info to write to output array
                # This is the input array structure:
                # [ROI #, Nuclei #, Channel, X_pos, Y_pos, Z_slice, Max Intensity Pixel Value (0-65536)]
                
                out_nuc = r1[1]
                out_distance = distance
                
                out_c_A = r1[2]
                out_c_A_roi_id = r1[0]
                out_r1_x = r1[3]
                out_r1_y = r1[4]
                out_r1_zslice = r1[5]
                out_r1_max = r1[6]

                out_c_B = r2[2]
                out_c_B_roi_id = r2[0]
                out_r2_x = r2[3]
                out_r2_y = r2[4]
                out_r2_zslice = r2[5]
                out_r2_max = r2[6]

                # Construct row to put in the output array 
                # This is the structure of from the input sheet: 
                # #Nuc_#    Distance(um)	Channel_A	A_ROI_#	A_X	A_Y	A_Zslice	A_max_intensity_pixel_value(0-65536)	Channel_B	B_ROI_#	B_X	B_Y	B_Zslice	B_max_intensity_pixel_value(0-65536)    
                
                out_row = [
                    out_nuc,
                    out_distance,
                    out_c_A,
                    out_c_A_roi_id,
                    out_r1_x,
                    out_r1_y,
                    out_r1_zslice,
                    out_r1_max,
                    out_c_B,
                    out_c_B_roi_id,
                    out_r2_x,
                    out_r2_y,
                    out_r2_zslice,
                    out_r2_max
                ]

                # Put in the correct colocalization bucket
                if is_coloc == True:
                    yes_arr.append(out_row)
                elif is_coloc == False:
                    no_arr.append(out_row)

            aparams.yes_arr = yes_arr
            aparams.no_arr = no_arr

            
##############################
##############################
###-----RANDOM_COLOC_CALCULATIONS-----###
    def rotate_xy(self,x,y):
        width = 2048        # need to store in cfg metadata
        height = 2044       # need to store in cfg metadata

        # Center point of image for rotation
        cx = width / 2  # = 2048 / 2 = 1024.0
        cy = height / 2 # = 2044 / 2 = 1022.0

        # Translate point to origin
        x_shifted = x - cx
        y_shifted = y - cy

        # Rotate 90° clockwise
        x_rotated = y_shifted
        y_rotated = -x_shifted

        # Translate back
        x_new = x_rotated + cx
        y_new = y_rotated + cy

        return x_new,y_new
    
    def rot_maxZ_finder(self, r2, x_rot, y_rot):
        file_id,channel,chanrad,chan_e,kpID,kpOvalID,kp_x,kp_y,nucInd,polyID,polyTextID, max_slice_index, max_intensity = r2
        chanroi_radius = chanrad
        pixelradius = math.ceil(chanroi_radius * self.ppm)
        radius = pixelradius
        center = int(x_rot), int(y_rot)

        if file_id == 1:
            chan_slices = self.f1_z_slice_dict[channel]
        elif file_id == 2:
            chan_slices = self.f2_z_slice_dict[channel]

        max_intensity =-1
        max_slice_index = ''
        for sliceindex, slice in enumerate(chan_slices, start=1):
            print(f"slice ind {sliceindex}")
            gray_image_slice = np.array(chan_slices[sliceindex-1])
            y, x = np.ogrid[:gray_image_slice.shape[0], :gray_image_slice.shape[1]]
            mask_circle = (x - center[0]) ** 2 + (y - center[1]) ** 2 > radius ** 2
            gray_image_slice[mask_circle] = 0  # Set the area outside the circle to 0
            roi = gray_image_slice
            
            max_value = np.max(roi)
            print(f" SLICE {sliceindex} - {max_value}")

            if max_value > max_intensity:
                max_slice_index = sliceindex
                max_intensity = max_value
            print(f"max intensity slice {max_slice_index}")
            print(f"max intensity slice counter {max_intensity}")  

        return max_slice_index
        
        
    def random_coloc_calculation(self):
        ROI_pairs = self.pairs_array
        f2_offset_x = self.x_offset
        f2_offset_y = self.y_offset
        f1_middle_Zslice = int(cfg.F1_Z_NUM/2)
        f2_middle_Zslice = int(cfg.F2_Z_NUM/2)

        yes_arr = []
        no_arr = []

        
        for row in ROI_pairs:
            # file_id,channel,chanrad,chan_e,kpID,
            # kpOvalID,kp_x,kp_y,nucInd,polyID, 
            # polyTextID,max_slice_index, max_intensity
            r1 = row[0]
            r2 = row[1]

            # Determine if it is the same file or different files
            r1_file, r2_file, same_file = self.same_file_check(r1[0],r2[0])

            #if r1[0] == r2[0]: continue  # ignore if same file
            if r1[0] == r2[0] and r1[1] == r2[1]: continue # ignore if same file channel
            if r1[8] != r2[8]: continue # ignore if not same nucleus

            # Get x,y coordinates from file 1
            x1 = r1[6]
            y1 = r1[7]

            # Rotate stored r2 x,y positions in original image
            x2_rot,y2_rot = self.rotate_xy(r2[6],r2[7])

            # Calculate r2 x,y positions in offset space
            x2,y2 = self.determine_r2_xy(x2_rot, y2_rot, r1_file, r2_file, same_file,f2_offset_x, f2_offset_y)

            # Find z-slice for r2 rotated positions
            #r2_rot_z_sliceIND = self.rot_maxZ_finder(r2,x2_rot,y2_rot)

            ###CALCULATE DELTAS###
            x2,y2 = self.determine_r2_xy(r2[6], r2[7], r1_file, r2_file, same_file,f2_offset_x, f2_offset_y)
            #X
            x_delta = (x2 - x1) * self.mpp
            #Y
            y_delta = (y2 - y1) * self.mpp
            #Z
            zslice_offset = self.calculate_z_offset(f1_middle_Zslice, f2_middle_Zslice)
            z_delta = abs(r1[11] - (r2[11]) * self.z_spacing)

            # Calculate distance
            distance = math.sqrt((x_delta)**2 + (y_delta)**2 + (z_delta)**2)

            # Determine if it is a colocalization
            f1_coloc_cutoff = 1#r1[3]
            f2_coloc_cutoff = 1#r2[3]

            coloc_cutoff = min(f1_coloc_cutoff,f2_coloc_cutoff)
            
            newrow = [r1,r2]
            if distance <= coloc_cutoff:
                is_coloc = True
                yes_arr.append(newrow)
                
            elif distance > coloc_cutoff:
                is_coloc = False
                no_arr.append(newrow)
            
            print(f"{distance} - {coloc_cutoff} - {is_coloc} ")

        #percent_coloc = len(yes_arr) / (len(yes_arr) + len(no_arr))
        total = len(yes_arr) + len(no_arr)
        if total == 0:
            print("No ROI pairs processed. Skipping percent_coloc calculation.")
            percent_coloc = 0
        else:
            percent_coloc = len(yes_arr) / total


        #file_id,channel,chanrad,chan_e,kpID,kpOvalID,kp_x,kp_y,nucInd,polyID,polyTextID, max_slice_index, max_intensity
        print("$$$$$$$$$$$$$RANDOM COLOC CALCULATIONS$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(ROI_pairs)
        print("-------------")
        print(len(yes_arr))
        print(len(no_arr))
        print(percent_coloc)

        cfg.RAND_COLOC_PERCENT = percent_coloc

        self.to_finish_analysis()

##############################
##############################
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

    def to_finish_analysis(self):
        self.switch()

    def make_finish_button(self):
        self.finish_button = tk.Button(self.master, text="!!!--Finalize ROIs--!!!", command=self.to_finish_analysis)
        self.finish_button.grid(row=500, column=0, columnspan=3)   
