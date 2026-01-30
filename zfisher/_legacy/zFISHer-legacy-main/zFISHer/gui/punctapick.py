import tkinter as tk
import os
from PIL import Image, ImageTk, ImageGrab, ImageFont, ImageDraw
#import zFISHer.config.config_manager as cfgmgr
import zFISHer.utils.config as cfg
import zFISHer.data.parameters as aparams
import cv2
import numpy as np
import math
import platform
from tkinter import colorchooser
import random
import io



class PunctaPickGUI(tk.Frame):
    def __init__(self, master, switch_to_gui_two, logger):  # Ensure it takes an argument
        
        #TODO MOVE THIS TO CFGMGR
        self.ppm = 0.108333333333333 #micron_per_pixel 
        #TODO set up arrows
        self.arrows_arr = []
        self.arrow_sp = None
        self.arrow_ep = None
        #Store capture_img_index
        self.capimg_count = 0
           
        print("INTIALIZE ROI PICKING")
        self.set_root_window(master)    
        self.set_gui_switch(switch_to_gui_two)
        self.set_logger(logger)
        self.get_input_filepaths()
        self.get_analysis_params()
        ######################################################
        ######################################################
        self.generate_MIP_dicts()
        self.initialize_ROI_canvas_window(master)
        self.initialize_control_window()
        ######################################################
        ######################################################
        self.set_user_input_controls()
        self.initialize_img_constructor_info()       
        self.update_composite_image()
        self.initialize_input_polygons()
        ######################################################
        ######################################################  
        self.initialize_ROI_dicts()     


    def set_root_window(self,master):
        """
        """
        if hasattr(master, 'newWindow'):
            self.root = master.newWindow
        else:
            self.root = master  # or handle the case differently


    def set_gui_switch(self,switch_to_gui_two):
        """
        """
        self.switch = switch_to_gui_two  # Save the function


    def set_logger(self, logger):
        """
        """
        self.logger = logger


    def get_input_filepaths(self):
        """
        """
        self.f1_ntag = cfg.F1_NTAG
        self.f2_ntag = cfg.F2_NTAG

        self.f1_cs = cfg.F1_C_LIST
        self.f2_cs = cfg.F2_C_LIST

        self.processing_dir = cfg.PROCESSING_DIR


        self.f1_reg_c = self.get_reg_channel(self.f1_cs)
        self.logger.log_message(f"Channel 1 registration channel: INDEX:{self.f1_reg_c} - NAME:{self.f1_cs[self.f1_reg_c]}")
        self.f2_reg_c= self.get_reg_channel(self.f2_cs)
        self.logger.log_message(f"Channel 2 registration channel: INDEX:{self.f2_reg_c} - NAME:{self.f2_cs[self.f2_reg_c]}")   

        self.F1_base_path = os.listdir(os.path.join(os.path.join(self.processing_dir,self.f1_ntag,self.f1_cs[self.f1_reg_c],"MIP"))[0])
        self.F2_base_path = os.listdir(os.path.join(os.path.join(self.processing_dir,self.f2_ntag,self.f2_cs[self.f2_reg_c],"MIP"))[0])

        self.f1_zslices_dir = os.path.join(self.processing_dir,self.f1_ntag,self.f1_cs[self.f1_reg_c],"z_slices")
        self.f2_zslices_dir = os.path.join(self.processing_dir,self.f2_ntag,self.f2_cs[self.f2_reg_c],"z_slices")


    def get_reg_channel(self, channels):
        for index,channel in enumerate(channels):
            if channel == "DAPI":
                return index   
            

    def get_analysis_params(self):
         ###------------Pull info to initialize-------------###
        #Get Nucleus and Puncta Arrays DynData
        self.kp_input = aparams.keypoints_array
        #Get F2 Offset
        self.f2_offset = [cfg.OFFSET_X, cfg.OFFSET_Y]
        #Get Polygon Coords
        self.polygons = aparams.nuc_polygons #[nucIndex,polyCoords] from Nuc Picking
        self.polygons = cfg.SEG_NUC_POLYGONS
        #----------------------------------------------------#        
    def generate_MIP_dicts(self):
        # Load the file1 mips into its dictionary
        self.f1_MIP_dict = {}
        self.f2_MIP_dict = {}
        for c in self.f1_cs:
            mip_dir = os.path.join(os.path.join(self.processing_dir,self.f1_ntag,c,"MIP"))
            mip_file = os.listdir(mip_dir)[0]
            mip_fullpath = os.path.join(mip_dir,mip_file)
            mip_file = cv2.imread(mip_fullpath, cv2.IMREAD_UNCHANGED)
            self.f1_MIP_dict[c] = mip_file

        for c in self.f2_cs:
            mip_dir = os.path.join(os.path.join(self.processing_dir,self.f2_ntag,c,"MIP"))
            mip_file = os.listdir(mip_dir)[0]
            mip_fullpath = os.path.join(mip_dir,mip_file)
            mip_file = cv2.imread(mip_fullpath, cv2.IMREAD_UNCHANGED)
            self.f2_MIP_dict[c] = mip_file


    def initialize_ROI_canvas_window(self,master):
        print("BEGIN INITIALIZE ROI CANVAS FRAME")
        tk.Frame.__init__(self, master)
        self.master.title('ZFISHER --- ROI Picking')
        master.geometry("1024x1022")

        self.highlighted_keypoints = set()      ###### NOT SURE IF NECESSARY#####

        # Vertical and horizontal scrollbars for canvas
        vbar = AutoScrollbar(self.master, orient='vertical')
        hbar = AutoScrollbar(self.master, orient='horizontal')
        vbar.grid(row=0, column=1, rowspan=2, sticky='ns')
        hbar.grid(row=2, column=0, sticky='we')

        # Create canvas and put image on it
        self.canvas = tk.Canvas(self.master, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        vbar.configure(command=self.scroll_y)  # bind scrollbars to the canvas
        hbar.configure(command=self.scroll_x)

        #Set F2 OPAC
        self.f2opac = 0.7
        # Make the canvas expandable
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.scalebar_visible = True

        mip_dir = os.path.join(os.path.join(self.processing_dir,self.f1_ntag,self.f1_cs[0],"MIP"))
        mip_file = os.listdir(mip_dir)[0]
        self.input_img_path= os.path.join(mip_dir,mip_file)
        self.image = Image.open(self.input_img_path)  # open image
        self.modimage = self.image.copy()
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.width, self.height = self.image.size
        #print(f"self.width height {self.width,self.height}")
        self.imscale = 1.0  # scale for the canvaas image
        self.scalefactor = 1.0
        self.delta = 1.3  # zoom magnitude
        

        
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)


        self.padx = int(self.f2_offset[0])
        self.pady = int(self.f2_offset[1])

       # print(f"selfcon {self.canvas.coords(self.container)}")
       # print(f"padx {self.padx} pady {self.pady}")
        self.containerpadded = self.canvas.create_rectangle(0, 0, self.width+abs(math.ceil((self.padx*2)/2)*2), self.height+abs(math.ceil((self.pady*2)/2)*2), width=0)
       # print(f"selfconpadded {self.canvas.coords(self.containerpadded)}")
        bbox_padded = self.canvas.bbox(self.containerpadded)
        width_padcont= int(bbox_padded[2] - bbox_padded[0])
        height_padcont =  int(bbox_padded[3] - bbox_padded[1])
        self.padimgwidth = width_padcont
        self.padimgheight = height_padcont
       # print(f"{self.padimgwidth},{self.padimgheight}")

        print("FINISHED INITALIZED ROI CANVAS FRAME")     
    def initialize_control_window(self):
        self.control_window = tk.Toplevel(self.master)
        self.control_window.title("Control Panel - ROI Picking")      

        #---------------------------------------------------------------------------
        # Make the channel toggle and color picker for each frame
        self.make_channel_toggle_frame()
        
        self.make_ROI_picking_frame()

        self.make_misc_frame()

        self.make_finish_button()









        #self.toggle_image_channels()
        #self.make_toggle_mip_dict()
        print("INITiALIZED CONTROL WIDOW")
    def make_channel_toggle_frame(self):
        self.chanframe = tk.Frame(self.control_window)
        self.chanframe.grid(row=3, column=0)
        # Make channel view toggle variables for composite image
        self.set_channel_view_toggle_vars()
        # Make and set channel colors for each channel in a dictionary
        self.set_channel_color_picker_vars()
        # Make the file channel control gui frame for each channel
        self.make_bc_frames()
    def set_channel_view_toggle_vars(self):
        self.F1_toggles = {}
        self.F2_toggles = {}   
        for i, channel in enumerate(self.f1_cs):
            toggle_name = f"F1_{channel}"
            self.F1_toggles[toggle_name] = tk.BooleanVar(value=True)
        for i, channel in enumerate(self.f2_cs):
            toggle_name = f"F2_{channel}"
            self.F2_toggles[toggle_name] = tk.BooleanVar(value=True)
    def set_channel_color_picker_vars(self):
        self.f1_color_dict = {}
        self.f2_color_dict = {}

        for c in self.f1_cs:
            self.f1_color_dict[c] = f'#{random.randint(0, 0xFFFFFF):06x}'
        for c in self.f2_cs:
            self.f2_color_dict[c] = f'#{random.randint(0, 0xFFFFFF):06x}'

        for index, chan in enumerate(self.f1_cs):
            if chan == "DAPI":
                self.f1_color_dict[chan] = "#0000FF"
        for index, chan in enumerate(self.f2_cs):
            if chan == "DAPI":
                self.f2_color_dict[chan] = "#0000FF"
    def set_channel_roi_color_picker_vars(self):
        self.f1_roi_color_dict = {}
        self.f2_roi_color_dict = {}

        for c in self.f1_cs:
            self.f1_roi_color_dict[c] = f'#{random.randint(0, 0xFFFFFF):06x}'
        for c in self.f2_cs:
            self.f2_roi_color_dict[c] = f'#{random.randint(0, 0xFFFFFF):06x}'

    def make_bc_frames(self):
        # Define main frame to hold both of the file frames and a spacer frame
        self.bc_mframe = tk.Frame(self.control_window)
        self.bc_mframe.grid(row=5,column=0)
        # Define the spacer frame
        self.bcspacerframe = tk.Frame(self.bc_mframe)
        self.bcspacerframe.grid(row=5,column=7,padx=20)
        # Define frame and grid for file 1 and file 2 frames
        self.bc1frame = tk.Frame(self.bc_mframe)
        self.bc1frame.grid(row=5,column=5)
        self.bc2frame = tk.Frame(self.bc_mframe)
        self.bc2frame.grid(row=5,column=10)
        # Define the control widgets that comprise the frame
        self.F1_checkbox = {}
        self.F2_checkbox = {}
        self.F1_cpick_dict = {}
        self.F2_cpick_dict = {}
        self.F1_cpick_but_dict = {}
        self.F2_cpick_but_dict = {}
        self.F1_brightness_labels_dict = {}
        self.F2_brightness_labels_dict = {}
        self.F1_brightness_sliders_dict = {}
        self.F2_brightness_sliders_dict = {}
        self.F1_contrast_labels_dict = {}
        self.F2_contrast_labels_dict = {}
        self.F1_contrast_sliders_dict = {}
        self.F2_contrast_sliders_dict = {}
        # Generate widgets in each of the file's control frame guis
        brightness_min = -200
        brightness_max = 400
        contrast_min = -200
        contrast_max = 400
        self.make_file_channel_control_gui_frame_widgets(brightness_max,brightness_min,contrast_max,contrast_min)
        # Set the values for the brightness and contrast sliders
        self.set_bc_values()
    def make_file_channel_control_gui_frame_widgets(self,brightness_max,brightness_min,contrast_max,contrast_min):
        base_row = 50
        base_column = 50
        for i, channel in enumerate(self.f1_cs):
            toggle_name = f"F1_{channel}"
            # Define the channel checkbox toggles    
            self.F1_checkbox[channel] = tk.Checkbutton(self.bc1frame, text=f"{self.f1_ntag}_C0_{self.f1_cs[i]}", variable=self.F1_toggles[toggle_name], command=self.update_composite_image)
            self.F1_checkbox[channel].grid(row=base_row+i*50, column=base_column, sticky='w')
            # Define the channel color picker squares 
            self.F1_cpick_dict[toggle_name] = tk.Label(self.bc1frame, width=2, height=1, bg=self.f1_color_dict[channel], relief="solid")
            self.F1_cpick_dict[toggle_name].grid(row=base_row+i*50, column=base_column + 10, sticky='w')
            # Define the color pick button
            self.F1_cpick_but_dict[channel] = tk.Button(self.bc1frame, text=f"{self.f1_ntag}_C0_{self.f1_cs[0]}_COLOR", command=lambda ch=channel, lbl=self.F1_cpick_dict[f"F1_{channel}"]: self.pick_channel_color(1, ch, lbl))

            self.F1_cpick_but_dict[channel].grid(row=base_row+i*50, column=base_column + 5, sticky='w', pady=10)
            # Define brightness sliders
            self.F1_brightness_labels_dict[toggle_name] = tk.Label(self.bc1frame, text=f"Brightness")
            self.F1_brightness_labels_dict[toggle_name].grid(row=base_row+10+i*50, column=base_column,sticky='s')
            self.F1_brightness_sliders_dict[channel] = tk.Scale(
                self.bc1frame,
                from_=brightness_min, 
                to=brightness_max,length=300,  
                orient="horizontal", 
                command=lambda value, channel=channel: self.update_bright_values(value, channel, 1)
                )
            self.F1_brightness_sliders_dict[channel].set(0)
            self.F1_brightness_sliders_dict[channel].grid(row=base_row+10+i*50, column=base_column+5)
            # Define contrast sliders
            self.F1_contrast_labels_dict[toggle_name] = tk.Label(self.bc1frame, text=f"Contrast")
            self.F1_contrast_labels_dict[toggle_name].grid(row=base_row+20+i*50, column=base_column,sticky='s')
            self.F1_contrast_sliders_dict[channel] = tk.Scale(
                self.bc1frame, 
                from_=contrast_min, 
                to=contrast_max,length=300,  
                orient="horizontal", 
                command=lambda value, channel=channel: self.update_contrast_values(value, channel, 1)
                )
            self.F1_contrast_sliders_dict[channel].set(0)
            self.F1_contrast_sliders_dict[channel].grid(row=base_row+20+i*50, column=base_column+5)
        for i, channel in enumerate(self.f2_cs):
            toggle_name = f"F2_{channel}"
            # Define channel checkbox toggles
            self.F2_checkbox[channel] = tk.Checkbutton(self.bc2frame, text=f"{self.f2_ntag}_C0_{self.f2_cs[i]}", variable=self.F2_toggles[toggle_name], command=self.update_composite_image)
            self.F2_checkbox[channel].grid(row=base_row+i*50, column=base_column, sticky='w')
            # Define the channel color picker squares 
            self.F2_cpick_dict[toggle_name] = tk.Label(self.bc2frame, width=2, height=1, bg=self.f2_color_dict[channel], relief="solid")
            self.F2_cpick_dict[toggle_name].grid(row=base_row+i*50, column=base_column + 10, sticky='w')
            # Define the color pick button
            self.F2_cpick_but_dict[channel] = tk.Button(self.bc2frame, text=f"{self.f2_ntag}_C0_{self.f2_cs[0]}_COLOR", command=lambda ch=channel, lbl=self.F2_cpick_dict[f"F2_{channel}"]: self.pick_channel_color(2, ch, lbl))

            self.F2_cpick_but_dict[channel].grid(row=base_row+i*50, column=base_column + 5, sticky='w', pady=10)
            # Define brightness sliders
            self.F2_brightness_labels_dict[toggle_name] = tk.Label(self.bc2frame, text=f"Brightness")
            self.F2_brightness_labels_dict[toggle_name].grid(row=base_row+10+i*50, column=base_column,sticky='s')
            self.F2_brightness_sliders_dict[channel] = tk.Scale(
                self.bc2frame,
                from_=brightness_min, 
                to=brightness_max,length=300,  
                orient="horizontal", 
                command=lambda value, channel=channel: self.update_bright_values(value, channel, 2)
                )
            self.F2_brightness_sliders_dict[channel].set(0)
            self.F2_brightness_sliders_dict[channel].grid(row=base_row+10+i*50, column=base_column+5)
            # Define contrast sliders
            self.F2_contrast_labels_dict[toggle_name] = tk.Label(self.bc2frame, text=f"Contrast")
            self.F2_contrast_labels_dict[toggle_name].grid(row=base_row+20+i*50, column=base_column,sticky='s')
            self.F2_contrast_sliders_dict[channel] = tk.Scale(
                self.bc2frame, 
                from_=contrast_min, 
                to=contrast_max,length=300,  
                orient="horizontal", 
                command=lambda value, channel=channel: self.update_contrast_values(value, channel, 2)
                )
            self.F2_contrast_sliders_dict[channel].set(0)
            self.F2_contrast_sliders_dict[channel].grid(row=base_row+20+i*50, column=base_column+5)
    def set_bc_values(self):
        self.f1_bright_values_dict = {}
        self.f1_contrast_values_dict = {}
        self.f2_bright_values_dict = {}
        self.f2_contrast_values_dict = {}
        for i,channel in enumerate(self.f1_cs):
            self.f1_bright_values_dict[channel] = float(self.F1_brightness_sliders_dict[channel].get())
            self.f1_contrast_values_dict[channel] = float(self.F1_contrast_sliders_dict[channel].get())
        for i,channel in enumerate(self.f2_cs):
            self.f2_bright_values_dict[channel] = float(self.F2_brightness_sliders_dict[channel].get())
            self.f2_contrast_values_dict[channel] = float(self.F2_contrast_sliders_dict[channel].get())          
    ###
    def make_ROI_picking_frame(self):
        # Define frame position in parent control window GUI
        kpchan_frame = tk.Frame(self.control_window)
        kpchan_frame.grid(row=100,column=0,columnspan=3)
        self.kpchantitle_label = tk.Label(kpchan_frame, text="------------------------------------------------------------Channel ROI Autopicking------------------------------------------------------------")
        self.kpchantitle_label.grid(row=0, column=0, columnspan=8, padx=5, pady=5)
        # Define variables for each channel to be referenced during ROI picking
        self.set_ROI_picking_toggle_vars()
        # Make and set channel colors for each channel in a dictionary
        self.set_channel_roi_color_picker_vars()
        # Declare dictionaries of widgets for ROI control panel
        self.F1_roi_cb_dict = {}
        self.F2_roi_cb_dict = {}
        self.F1_roi_radius_entry_label_dict = {}
        self.F2_roi_radius_entry_label_dict = {}
        self.F1_roi_radius_entry_dict = {}
        self.F2_roi_radius_entry_dict = {}
        self.F1_coloc_entry_label_dict = {}
        self.F2_coloc_entry_label_dict = {}
        self.F1_coloc_entry_dict = {}
        self.F2_coloc_entry_dict = {}
        self.F1F2_rb_dict = {}
        self.F1F2_rb_color_dict = {}
        self.F1F2_rb_color_buttons_dict = {}
        # Generate widgets to control ROI picking on canvas
        self.make_ROI_picking_control_gui_frame_widgets(kpchan_frame)
    def make_ROI_picking_control_gui_frame_widgets(self,frame):
        # Define grid values
        base_row = 10
        base_column = 2
        #HEADERS:
        self.rb_color_header = tk.Label(frame, text="ROI Color").grid(row=1,column= base_column-1)
        self.kprbheader = tk.Label(frame, text="Active ROI").grid(row=1,column=base_column)
        self.kptoggleheader = tk.Label(frame,text="Visible").grid(row=1,column=base_column+1)
        self.kproisizeheader = tk.Label(frame,text="ROI Radius").grid(row=1,column=base_column+2,columnspan=2)
        self.kpheadspacer1 = tk.Label(frame,text="").grid(row=1,column=base_column+3)
        self.kpcolocheader = tk.Label(frame,text="Coloc Z Thresh").grid(row=1,column=base_column+4,columnspan=base_column+4)
        self.kpheadspacer2 = tk.Label(frame,text="").grid(row=1,column=base_column+5)
        channel_counter = 0 # channel counter for iterating through both file channels
        # Define file 1 widget options
        for i, channel in enumerate(self.f1_cs):
            self.f1_toggle_name = f"F1_{channel}"
            toggle_name = self.f1_toggle_name
            # Define channel ROI check button 
            self.F1_roi_cb_dict[toggle_name] = tk.Checkbutton(frame, text=f"{self.f1_ntag}_{self.f1_cs[0]}_ROI", variable=self.F1_kp_toggles[toggle_name], command=self.toggle_kp_visible)
            self.F1_roi_cb_dict[toggle_name].grid(row=base_row+i, column=base_column+1, sticky='w')
            # Define ROI entry labels
            self.F1_roi_radius_entry_label_dict[toggle_name] = tk.Label(frame, text="um") 
            self.F1_roi_radius_entry_label_dict[toggle_name].grid(row=base_row+i,column=base_column+3, sticky="w") 
            # Define ROI entry 
            self.F1_roi_radius_entry_dict[toggle_name] = tk.Entry(frame, width=10)
            self.F1_roi_radius_entry_dict[toggle_name].insert(0,"0.5")
            self.F1_roi_radius_entry_dict[toggle_name].grid(row=base_row+i,column=base_column+2, sticky='e')
            # Define ROI coloc entry label
            self.F1_coloc_entry_label_dict[toggle_name] = tk.Label(frame, text="um") 
            self.F1_coloc_entry_label_dict[toggle_name].grid(row=base_row+i,column=base_column+5, sticky='w') 
            # Define ROI coloc entry label
            self.F1_coloc_entry_dict[toggle_name] = tk.Entry(frame, width=10)
            self.F1_coloc_entry_dict[toggle_name].insert(0, "1.0")
            self.F1_coloc_entry_dict[toggle_name].grid(row=base_row+i,column=base_column+4,sticky='e')
            # Define radiobutton for active channel picking
            self.F1F2_rb_dict[toggle_name] = tk.Radiobutton(frame, text=f"{self.f1_ntag}_{channel}_ROI",variable=self.KP_rb_selection, value=toggle_name, command=None)
            self.F1F2_rb_dict[toggle_name].grid(row=base_row+i, column=base_column, sticky=tk.W)
            # Define colorbutton for changing ROI circle color
            self.F1F2_rb_color_dict[toggle_name] = tk.Label(frame, width=2, height=1, bg=self.f1_roi_color_dict[channel], relief="solid")
            self.F1F2_rb_color_dict[toggle_name].grid(row=base_row+i, column=base_column - 1)
            # Define a button underneath the label
            self.F1F2_rb_color_buttons_dict[toggle_name] = tk.Button(
                frame,
                text="Change Color",
                command=lambda f=1, ch=channel, sq=self.F1F2_rb_color_dict[toggle_name]: self.pick_ROI_color(f, ch, sq)
            )
            self.F1F2_rb_color_buttons_dict[toggle_name].grid(row=base_row + i, column=base_column-2)
            # Iterate to next channel
            channel_counter += 1
        # Define File 2 widget options
        for i, channel in enumerate(self.f2_cs):
            toggle_name = f"F2_{channel}"
            # Define channel ROI check button 
            self.F2_roi_cb_dict[toggle_name] = tk.Checkbutton(frame, text=f"{self.f2_ntag}_{self.f2_cs[0]}_ROI", variable=self.F2_kp_toggles[toggle_name], command=self.toggle_kp_visible)
            self.F2_roi_cb_dict[toggle_name].grid(row=base_row+i+channel_counter, column=base_column+1, sticky='w')
            # Define ROI entry labels
            self.F2_roi_radius_entry_label_dict[toggle_name] = tk.Label(frame, text="um") 
            self.F2_roi_radius_entry_label_dict[toggle_name].grid(row=base_row+i+channel_counter,column=base_column+3, sticky="w") 
            # Define ROI entry 
            self.F2_roi_radius_entry_dict[toggle_name] = tk.Entry(frame, width=10)
            self.F2_roi_radius_entry_dict[toggle_name].insert(0,"0.5")
            self.F2_roi_radius_entry_dict[toggle_name].grid(row=base_row+i+channel_counter,column=base_column+2, sticky='e')
            # Define ROI coloc entry label
            self.F2_coloc_entry_label_dict[toggle_name] = tk.Label(frame, text="um") 
            self.F2_coloc_entry_label_dict[toggle_name].grid(row=base_row+i+channel_counter,column=base_column+5, sticky='w') 
            # Define ROI coloc entry label
            self.F2_coloc_entry_dict[toggle_name] = tk.Entry(frame, width=10)
            self.F2_coloc_entry_dict[toggle_name].insert(0, "1.0")
            self.F2_coloc_entry_dict[toggle_name].grid(row=base_row+i+channel_counter,column=base_column+4,sticky='e')
            # Define radiobutton for active channel picking
            self.F1F2_rb_dict[toggle_name] = tk.Radiobutton(frame, text=f"{self.f2_ntag}_{channel}_ROI",variable=self.KP_rb_selection, value=toggle_name, command=None)
            self.F1F2_rb_dict[toggle_name].grid(row=base_row+i+channel_counter, column=base_column+0, sticky=tk.W)
            # Define colorbutton for changing ROI circle color
            self.F1F2_rb_color_dict[toggle_name] = tk.Label(frame, width=2, height=1, bg=self.f2_roi_color_dict[channel], relief="solid")
            self.F1F2_rb_color_dict[toggle_name].grid(row=base_row+i+channel_counter, column=base_column - 1)
            # Define a button underneath the label
            self.F1F2_rb_color_buttons_dict[toggle_name] = tk.Button(
                frame,
                text="Change Color",
                command=lambda f=2, ch=channel, sq=self.F1F2_rb_color_dict[toggle_name]: self.pick_ROI_color(f, ch, sq)
            )
            self.F1F2_rb_color_buttons_dict[toggle_name].grid(row=base_row + i+channel_counter, column=base_column-2)
            # Iterate to next channel
            channel_counter += 1
    def set_ROI_picking_toggle_vars(self):
        #Toggles if we should analyze
        self.F1_out_toggles = {}
        self.F2_out_toggles = {}   
        for i, channel in enumerate(self.f1_cs):
            toggle_name = f"F1_C{channel}_toggle"
            self.F1_out_toggles[toggle_name] = tk.BooleanVar(value=True)
        for i, channel in enumerate(self.f2_cs):
            toggle_name = f"F2_C{channel}_toggle"
            self.F2_out_toggles[toggle_name] = tk.BooleanVar(value=True)

        #CHANNEL KP RB Selectors
        self.KP_rb_selection = tk.StringVar()
        #CHANNEL KP PICK TOGGLES
        self.F1_kp_toggles = {}
        self.F2_kp_toggles = {}   
        for i, channel in enumerate(self.f1_cs):
            toggle_name = f"F1_{channel}"
            self.F1_kp_toggles[toggle_name] = tk.BooleanVar(value=True)
        for i, channel in enumerate(self.f2_cs):
            toggle_name = f"F2_{channel}"
            self.F2_kp_toggles[toggle_name] = tk.BooleanVar(value=True)
    def set_user_input_controls(self):
        #BINDINGS
        self.canvas.bind('<Configure>', self.display_composite_image)  # canvas is resized 
        self.canvas.bind("<Motion>", self.motion)
       # self.canvas.bind("<Button-1>", self.kp_add)
        self.canvas.bind("<Button-1>", self.left_mouseclick_wrapper)

        #KP_REMOVE
        #self.canvas.bind("<Button-2>", self.kp_remove)
        if platform.system() == "Darwin":  # macOS
            self.canvas.bind('<Button-2>', self.right_mouseclick_wrapper) # right-click for macOS
        elif platform.system() == "Linux":  # Linux
            self.canvas.bind('<Button-3>', self.right_mouseclick_wrapper) # right-click for Linux

        #DRAG
        #self.canvas.bind('<ButtonPress-3>', self.move_from)
        #self.canvas.bind('<B3-Motion>',     self.move_to)
        if platform.system() == "Darwin":  # macOS
             self.canvas.bind('<ButtonPress-3>', self.move_from)
             self.canvas.bind('<B3-Motion>',     self.move_to)
        elif platform.system() == "Linux":  # Linux
             self.canvas.bind('<ButtonPress-2>', self.move_from)
             self.canvas.bind('<B2-Motion>',     self.move_to)

        #ZOOM    
        #self.canvas.bind("<MouseWheel>", self.zoom)
        if platform.system() == "Darwin":  # macOS
            self.canvas.bind("<MouseWheel>", self.zoom)
        elif platform.system() == "Linux":  # Linux
            self.canvas.bind("<Button-4>", self.zoom)
            self.canvas.bind("<Button-5>", self.zoom)      
    def make_finish_button(self):
        self.finish_button = tk.Button(self.control_window, text="!!!--Finalize ROIs--!!!", command=self.finalize_ROI_picking)
        self.finish_button.grid(row=500, column=0, columnspan=3)    
    ###
    def make_misc_frame(self):
        self.miscframe = tk.Frame(self.control_window)
        self.miscframe.grid(row=5,column=1)

        self.scalebar_toggle_var = tk.BooleanVar(value=True)
        self.scalebar_toggle_checkbox = tk.Checkbutton(self.miscframe,text="Show Scalebar", variable=self.scalebar_toggle_var, command=self.toggle_scalebar).grid(row=2, column=1, columnspan=1)

        self.polygon_toggle_var = tk.BooleanVar(value=True)
        self.polygon_toggle_checkbox = tk.Checkbutton(self.miscframe,text="Show Nuclei", variable=self.polygon_toggle_var, command=self.polygon_toggle).grid(row=2, column=2, columnspan=1)

        self.arrow_toggle_var = tk.BooleanVar(value=False)
        self.arrow_toggle_checkbox = tk.Checkbutton(self.miscframe, text="Draw Arrows", variable=self.arrow_toggle_var, command=None).grid(row=5, column=1, columnspan=2)

        #CAPTURE IMAGE
        self.imgcap_button = tk.Button(self.miscframe, text="IMG CAPTURE WINDOW", command=lambda: self.capture_image())
        self.imgcap_button.grid(row=7,column=1, columnspan=2)

        #Mouse position label 
        self.mousepos_label = tk.Label(self.miscframe, text="Current Mouse Position: (0000.00,0000.00)", fg="white", font=("Courier"))
        self.mousepos_label.grid(row=3, column=0, columnspan=3)  # Use grid instead of pack
        x_lab = 0
        y_lab = 0
        self.mousepos_label.config(text=f"Mouse Position: ({float(x_lab):07.2f}, {float(y_lab):07.2f})")

        #Zoom label
        self.zoom_label = tk.Label(self.miscframe, text="Magnification: (1.00,1.00)", fg="white", font=("Courier"))
        self.zoom_label.grid(row=4, column=0, columnspan=3)  # Use grid instead of pack
        x_lab = 1.00
        y_lab = 1.00
        self.zoom_label.config(text=f"Magnification: {self.imscale:.2f}x")    
        #print(f"init2 F1C1G {self.F1_C1_MIP.size}")
    ###
    def make_toggle_mip_dict(self):
        self.f1f2_rb_mip_dict = {} #DICTIONARY TO MAKE
        #self.F1F2_rb_dict = {}     # dictionary of rb dictionary
        #self.f1_MIP_dict # dict of f1 mips
        #self.f2_MIP_dict # dict of f2 mips
        
        for i, channel in enumerate(self.f1_cs):
            key = f"F1_{channel}"
            rb_widget = self.F1F2_rb_dict[key]
            mip_image = self.f1_MIP_dict[channel]
            self.f1f2_rb_mip_dict[rb_widget] = mip_image

        for i, channel in enumerate(self.f2_cs):
            key = f"F2_{channel}"
            rb_widget = self.F1F2_rb_dict[key]
            mip_image = self.f2_MIP_dict[channel]
            self.f1f2_rb_mip_dict[rb_widget] = mip_image    

        print(self.f1f2_rb_mip_dict)
    def initialize_img_constructor_info(self):
        self.f1_padding, self.f2_padding = self.calc_image_offset_padding()
        self.padded_baseimg = self.generate_padded_base_img()       
    ###
    def initialize_input_polygons(self):
        self.polygons = aparams.nuc_polygons
        self.polygons_id = []    #[nucInd,polyID,polyTextID]
        self.kp_polygons_arr = [] #[kpID,kp_x,kp_y,nucInd,polyID,polyTextID]

        #Dynamic Info Array
        self.kp_poly_dyn_arr = [] #[kpID,kpOvalID,nucInd,polyID,polyTextID]

        self.polygon_toggle_var = tk.BooleanVar(value=True)
        self.polygon_toggle_checkbox = tk.Checkbutton(self.miscframe,text="Show Nuclei", variable=self.polygon_toggle_var, command=self.polygon_toggle).grid(row=2, column=2, columnspan=1)
        self.draw_input_polygons()
    def draw_input_polygons(self):
        for nucindex, polygoncoords in self.polygons:
            polygon_id= self.canvas.create_polygon(polygoncoords, outline="red", fill="", tags='polygon')
            self.canvas.move(polygon_id,abs(self.padx),abs(self.pady))
            polygon_textid = self.display_index_on_polygon(polygon_id, int(nucindex)) 
            temparray = [nucindex, polygon_id, polygon_textid]
            self.polygons_id.append(temparray)
    def display_index_on_polygon(self, polygon_id, index):
        coords = self.canvas.coords(polygon_id)
        x_coords = [int(coords[i]) for i in range(0, len(coords), 2)]
        y_coords = [int(coords[i]) for i in range(1, len(coords), 2)]
        
        # Check if either x_coords or y_coords is empty before performing division
        if len(x_coords) == 0 or len(y_coords) == 0:
            # Handle the case where one of the coordinate lists is empty
            print("Error: Empty coordinate list.")
            return
        
        x_center = sum(x_coords) // len(x_coords)  # Using integer division
        y_center = sum(y_coords) // len(y_coords)  # Using integer division
        text_item = self.canvas.create_text(x_center, y_center, text=str(index), font=('Arial', 12), fill='red', tags='polygontext')
        return text_item
    def polygon_toggle(self):
        polygon_items = self.canvas.find_withtag('polygon')
        polygon_text_items = self.canvas.find_withtag('polygontext')
        if self.polygon_toggle_var.get():
            for p in polygon_items:
                self.canvas.itemconfigure(p, state="normal")
            for pt in polygon_text_items:
                self.canvas.itemconfigure(pt, state="normal")
        else: 
            for p in polygon_items:
                self.canvas.itemconfigure(p, state="hidden")
            for pt in polygon_text_items:
                self.canvas.itemconfigure(pt, state="hidden")
    
    #######################################################
    ##################_DISPLAY_IMAGE_######################
    def display_composite_image(self, event=None):
        addimg = self.current_composite_img.copy()
        # Takes the current stored composite image and draws viewable portion on canvas
        ''' Show image on the Canvas '''
        bbox1 = self.canvas.bbox(self.container)  # get image area
        # Remove 1 pixel shift at the sides of the bbox1
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (self.canvas.canvasx(0),  # get visible area of the canvas
                self.canvas.canvasy(0),
                self.canvas.canvasx(self.canvas.winfo_width()),
                self.canvas.canvasy(self.canvas.winfo_height()))
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),  # get scroll region box
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # whole image in the visible area
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:  # whole image in the visible area
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        self.canvas.configure(scrollregion=bbox)  # set scroll region

        print(f"bbox1 ({bbox1[0]},{bbox1[1]},{bbox1[2]},{bbox1[3]})")
        print(f"bbox2 ({bbox2[0]},{bbox2[1]},{bbox2[2]},{bbox2[3]})")
        print(f"bbox ({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})")
        # Store bbox1 for later use
        self.bbox1 = bbox1
        self.bbox2 = bbox2

        x1 = max(bbox2[0] - bbox1[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]

        self.si_x1 = x1
        self.si_y1 = y1
        self.si_x2 = x2
        self.si_y2 = y2

        self.grey_image_offset_x = bbox1[0] - bbox2[0]
        self.grey_image_offset_y = bbox1[1] - bbox2[1]       

        print(f"x1,y1,x2,y2 {x1},{y1},{x2},{y2}")
        print(f"grey offset {self.grey_image_offset_x}, {self.grey_image_offset_y}")

        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            x = min(int(x2 / self.imscale), self.width)   # sometimes it is larger on 1 pixel...
            y = min(int(y2 / self.imscale), self.height)  # ...and sometimes not
            #image = self.image.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))

            image = addimg.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
           
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            self.exportimg = imagetk
            
            self.imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]), anchor='nw', image=imagetk)
            
            #self.bgimage= self.canvas.lower(self.imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

            self.canvas.lower(self.imageid)


    def update_composite_image(self):
        print("GENERATING NEW COMPOSITE IMAGE")
        # Get starting black image and image padding for offset
        padded_baseimg = self.padded_baseimg.copy()
        padding_f1 = self.f1_padding
        padding_f2 = self.f2_padding
        # The image is black if no channel is selected
        if self.nochannelsselected(): 
            blank_img = padded_baseimg.transpose(1, 0, 2)
            blank_img = np.clip(blank_img, 0, 255).astype(np.uint8)
            self.current_composite_img = Image.fromarray(blank_img)
            self.display_composite_image()
            return

        # Create composite if image if >=1 channel is selected
        added_image = None
        processed_images = [padded_baseimg]
        # Process file 1 channels
        for i,channel in enumerate(self.f1_cs):
            if not self.F1_toggles[f"F1_{channel}"].get():
                continue
            tempcopy = self.f1_MIP_dict[channel].copy()
            brightness = self.f1_bright_values_dict[channel]
            contrast = self.f1_contrast_values_dict[channel]
            bc_adj_img = self.apply_img_bc(tempcopy, brightness, contrast)
            ai_array = np.array(bc_adj_img)
            ai_array = self.apply_img_color(ai_array, channel, 1)
            padded_image = np.pad(ai_array, padding_f1, mode='constant')
            processed_images.append(padded_image)
        # Process file 2 chanels
        for i,channel in enumerate(self.f2_cs):
            if not self.F2_toggles[f"F2_{channel}"].get():
                continue
            tempcopy = self.f2_MIP_dict[channel].copy()
            brightness = self.f2_bright_values_dict[channel]
            contrast = self.f2_contrast_values_dict[channel]
            bc_adj_img = self.apply_img_bc(tempcopy, brightness, contrast)
            ai_array = np.array(bc_adj_img)
            ai_array = self.apply_img_color(ai_array, channel, 2)
            padded_image = np.pad(ai_array, padding_f2, mode='constant')
            processed_images.append(padded_image)
        # Sum final image array
        added_image = np.sum(processed_images, axis=0)
        added_image = np.clip(added_image, 0, 255).astype(np.uint8)
        self.addedimage = added_image
        if self.addedimage is not None:
            self.addedimage = cv2.cvtColor(self.addedimage, cv2.COLOR_BGR2RGB)
            addpil_image = Image.fromarray(self.addedimage)
            self.addimgmask = Image.new("L", addpil_image.size, int(255 * 1))
            self.addimgwopac = addpil_image.copy()
            self.addimgwopac.putalpha(self.addimgmask)      
            addimg_processed= np.array(self.addimgwopac)
            print(f"addimg shape {addimg_processed.shape}")
            addimg_processed = Image.fromarray(addimg_processed)
        # Store final image and draw on canvase
        self.current_composite_img = addimg_processed
        self.display_composite_image()
    def update_bright_values(self,value,channel,file): 
        if file == 1:
            self.f1_bright_values_dict[channel] = value
        elif file == 2:
            self.f2_bright_values_dict[channel] = value
        self.update_composite_image()
    def update_contrast_values(self,value,channel,file): 
        if file == 1:
            self.f1_contrast_values_dict[channel] = value
        elif file == 2:
            self.f2_contrast_values_dict[channel] = value 
        self.update_composite_image()
    def pick_channel_color(self,fnum,channel,square):
        print(f"{fnum} -- {channel} -- {square}")
        #self.f1_color_dict = {}
        #self.f2_color_dict = {}
        color = colorchooser.askcolor()[1]  # Ask the user to choose a color and get the hex code
        if color == None: return
        if color:
            #color_label.config(foreground=color)  # Set the foreground color of the Label
            print("color chosen")
        #print(f"HEX {color}")
        square.config(bg=color)
        if fnum == 1:
            self.f1_color_dict[channel] = color
        if fnum == 2:
            self.f2_color_dict[channel] = color
        # Remove the hash symbol if present
        hex_color = color.lstrip('#')
        # Convert the hex string to RGB tuple
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        print(f"RGB: {rgb}")
    def calc_image_offset_padding(self):
        #self.containeradded is the image size with padding in all 4 directions = f2offset
        #put f1 in the center (xy+abs(f2offset.xy))
        #put f2 in center + offset (so it is +/- sensitive)
        width_padcont= self.padimgwidth
        height_padcont =  self.padimgheight

        f2padx = int((width_padcont - self.width) /2)
        print(f"{width_padcont} - {self.width} /2 = {int((width_padcont - self.width) /2)}")

        f2pady = int((height_padcont - self.height) /2)


        f1padx = int((abs(f2padx)))
        f1pady = int((abs(f2pady)))

        #F2 OFFSET
        if self.f2_offset[0] < 0:
            f2padx_l = 0
            f2padx_r = 2*f2padx
        if self.f2_offset[0] > 0:
            f2padx_l = 2*f2padx
            f2padx_r = 0
        if self.f2_offset[0] == 0:
            f2padx_l = 0
            f2padx_r = 0

        if self.f2_offset[1] < 0:
            f2pady_u = 0
            f2pady_l = 2*f2pady
        if self.f2_offset[1] > 0:
            f2pady_u = 2*f2pady
            f2pady_l = 0
        if self.f2_offset[1] == 0:
            f2pady_u = 0
            f2pady_l = 0
        print(f"padxy {f1padx},{f1pady} ----")# {padded_baseimg.shape[0]},{padded_baseimg.shape[1]}")

        padding_f1 = ((f1pady, f1pady), (f1padx, f1padx), (0, 0))
        padding_f2 = ((f2pady_u, f2pady_l), (f2padx_l, f2padx_r), (0, 0))

        return padding_f1,padding_f2
    

    def generate_padded_base_img(self):
        width_padcont = self.padimgwidth
        height_padcont =  self.padimgheight    
        padded_baseimg = np.zeros((width_padcont, height_padcont, 3), dtype=np.uint8)
        padded_baseimg = padded_baseimg.transpose(1,0,2)    #numpy goes by height/width rather than width, height
        return padded_baseimg 
    

    def nochannelsselected(self):
        offcounter = 0
        for chantog in self.F1_toggles.values():
            offcounter -= 1
            if chantog.get() == False: offcounter += 1
        for chantog in self.F2_toggles.values():
            offcounter -= 1
            if chantog.get() == False: offcounter += 1
        if offcounter >= 0: return True
        else: return False


    def apply_img_bc(self, img, brightness, contrast):
        brightness = float(brightness)
        contrast = float(contrast)
        # Normalize image to 16-bit range
        normalized_image = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
        # Convert image to 8-bit uint8 for display
        normalized_image_uint8 = (normalized_image / 256).astype(np.uint8)
        # Convert 16-bit grayscale to RGB
        if len(normalized_image_uint8.shape) == 2:
            normalized_image_uint8 = cv2.cvtColor(normalized_image_uint8, cv2.COLOR_GRAY2RGB)


        image = normalized_image_uint8.astype(np.float32)

        # Adjust brightness
        print(f"BRIGHTNESS {brightness} -{type(brightness)}")
        image += brightness

        # Adjust contrast
        if contrast != 0:
            factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
            image = 128 + factor * (image - 128)

        # Clip values to [0, 255] and convert back to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)

        # Convert to PIL Image
        outimage = Image.fromarray(image)

        return outimage 
    def apply_img_color(self,img,chan,file):
        #self.f1_color_dict = {}
        #self.f2_color_dict = {}
        if file == 1:
            hex_color = self.f1_color_dict[chan]
        if file == 2:
            hex_color = self.f2_color_dict[chan]
        hex_code = hex_color.lstrip('#')
        rgb_color = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        grayscale_image = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        colored_image = np.zeros((grayscale_image.shape[0], grayscale_image.shape[1], 3), dtype=np.uint8)
        for i in range(3): colored_image[:, :, i] = grayscale_image * (rgb_color[2-i] / 255.0)
        return colored_image
    #######################################################
    ##################_MOVEMENT_CONTROLS_##################
    def scroll_y(self, *args, **kwargs):
        ''' Scroll canvas vertically and redraw the image '''
        self.canvas.yview(*args, **kwargs)  # scroll vertically
        #self.toggle_image_channels()  # redraw the image
        self.display_composite_image()
    def scroll_x(self, *args, **kwargs):
        ''' Scroll canvas horizontally and redraw the image '''
        self.canvas.xview(*args, **kwargs)  # scroll horizontally
        #self.toggle_image_channels() # redraw the image
        self.display_composite_image()
    def motion(self, event):
        # Calculate the mouse position in the original image
        mouse_x = self.canvas.canvasx(event.x)
        mouse_y = self.canvas.canvasy(event.y)

        mouse_x = self.canvas.canvasx(event.x)/self.imscale
        mouse_y = self.canvas.canvasy(event.y)/self.imscale

        x_lab = float(mouse_x)
        y_lab = float(mouse_y)

        #NEED TO DEFINE LABEL IN CONTROL PANEL
        #self.mousepos_label.config(text=f"Mouse Position: ({float(x_lab):07.2f}, {float(y_lab):07.2f})")
    def zoom(self, event):
        print("scrolling")
        ''' Zoom with mouse wheel '''
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]: pass  # Ok! Inside the image
        else: return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta > 0:  # scroll down
            print("down)")
            i = min(self.width, self.height)
            if int(i * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale /= self.delta
            scale        /= self.delta
        if event.num == 4 or event.delta < 0:  # scroll up
            print("up")
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.delta
            scale        *= self.delta

        print(self.imscale)
        self.zoom_label.config(text=f"Magnification: {self.imscale:.2f}x")    

        self.canvas.scale('all', x, y, scale, scale)  # rescale all canvas objects
        #self.canvas.scale('all', x, y, scale, scale)  # rescale all canvas objects
        self.display_composite_image()
       # self.display_image(self.modimage)
    def left_mouseclick_wrapper(self,event):
        if not self.arrow_toggle_var.get(): self.add_ROI(event)
        if self.arrow_toggle_var.get(): self.make_arrow(event)
    def right_mouseclick_wrapper(self,event):
        if not self.arrow_toggle_var.get(): self.remove_ROI(event)
        if self.arrow_toggle_var.get(): self.remove_arrow(event)
    def move_from(self, event):
        ''' Remember previous coordinates for scrolling with the mouse '''
        self.canvas.scan_mark(event.x, event.y)
    def move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        #
        self.display_composite_image()  # redraw the image
    ########################################################
    ##################_ROI_FUNCTIIONS#######################   
    def initialize_ROI_dicts(self):
        # Each file channel ROI is stored as a row: [kpID,kpOvalID,nucInd,polyID,polyTextID]
        self.F1_ROI_dict = {}
        self.F2_ROI_dict = {}

        for i, channel in enumerate(self.f1_cs):
            self.F1_ROI_dict[channel] = []
        
        for i, channel in enumerate(self.f2_cs):
            self.F2_ROI_dict[channel] = []
    def pick_ROI_color(self,file,channel,square):
        color = colorchooser.askcolor()[1]  # Ask the user to choose a color and get the hex code
        if color == None: return
        if color:
            print(f"Hex color chosen: {color}")
        square.config(bg=color)
        if file == 1:
            self.f1_roi_color_dict[channel] = color #### NEED TO MAKE A NEW ROI COLOR DICT,t his is the DICT FOR THE MIP COLOR
        if file == 2:
            self.f2_roi_color_dict[channel] = color
        # Remove the hash symbol if present
        hex_color = color.lstrip('#')
        # Convert the hex string to RGB tuple
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        print(f"RGB color chosen: {rgb}")

        self.recolor_channel_ROIs(file,channel,color)

    def recolor_channel_ROIs(self,file,channel,color):
        if file == 1:
            roi_list = self.F1_ROI_dict[channel]
        elif file == 2:
            roi_list = self.F2_ROI_dict[channel]
        else:
            print(f"Invalid file number: {file}")
            return

        for roi in roi_list:
            oval_id = roi[1]
            self.canvas.itemconfig(oval_id, outline=color)

    def add_ROI(self,event):
        #NEED TO ADD IN LOGIC OF kp_add
        selectedchannel = self.KP_rb_selection.get()
        channel = selectedchannel[3:]
        # Determine ROI array to add to 
        if selectedchannel.startswith("F1_"):
            file = 1
            active_channel_arr = self.F1_ROI_dict[channel]
            ROI_radius_entry = self.F1_roi_radius_entry_dict[selectedchannel].get()
            color = self.f1_roi_color_dict[channel]
        elif selectedchannel.startswith("F2_"):
            file = 2
            active_channel_arr = self.F2_ROI_dict[channel]
            ROI_radius_entry = self.F2_roi_radius_entry_dict[selectedchannel].get()
            color = self.f2_roi_color_dict[channel]
        
        print(selectedchannel)
        print(channel)
        oval_radius = float(ROI_radius_entry) / self.ppm * self.imscale

        kpx = self.canvas.canvasx(event.x)
        kpy = self.canvas.canvasx(event.y)
        oval_id = self.canvas.create_oval(kpx - oval_radius, 
                                          kpy - oval_radius, 
                                          kpx + oval_radius, 
                                          kpy + oval_radius, 
                                          outline=color, 
                                          fill="", 
                                          state="normal", 
                                          tags='kp')

        kparr = active_channel_arr
        kparr = sorted(kparr, key=lambda x: x[0]) #sort them in ascending order of kpID
        if len(kparr) == 0: 
            kpID = 1  #if there are no keypoints make it first KPID = 1
        else:
            missing_numbers = []
            expected_number = 1
            for inforow in kparr:   #[kpID,OvalID,kp_x,kp_y,NucID,PolygonID,PolygonTextID]
                #kpID,OvalID,kp_x,kp_y,NucID,PolyID,PolyTextID = inforow

                while inforow[0] > expected_number:
                    missing_numbers.append(expected_number)
                    expected_number += 1
                expected_number = inforow[0] + 1
            if len(missing_numbers) > 0: kpID = missing_numbers[0]
            else: kpID = len(kparr) + 1
        

        kp_x = (self.canvas.coords(oval_id)[0] + self.canvas.coords(oval_id)[2])/2
        kp_y = (self.canvas.coords(oval_id)[1] + self.canvas.coords(oval_id)[3])/2

        print(self.canvas.coords(oval_id))
        NucID = 1
        PolyID = 1
        PolyTextID = 1

        NucID,PolyID,PolyTextID = self.findpolygon(kp_x,kp_y)
        print(f" {NucID} - {PolyID} - {PolyTextID}")

        temparray = [kpID,oval_id,kp_x,kp_y,NucID,PolyID,PolyTextID]

        active_channel_arr.append(temparray)

        print(active_channel_arr)
        print(self.F1_ROI_dict)
        print(self.F2_ROI_dict)
    def findpolygon(self, x, y):
        foundNucID = 0
        foundPolyID = 0
        foundPolyTextID = 0
        for row in self.polygons_id:
            NucID, PolyID, PolyTextID = row
            poly_coords = self.canvas.coords(PolyID)
            num_coords = len(poly_coords)
            inside = False
            j = num_coords - 2
            for i in range(0, num_coords, 2):
                x_i, y_i = poly_coords[i], poly_coords[i + 1]
                x_j, y_j = poly_coords[j], poly_coords[j + 1]
                if ((y_i > y) != (y_j > y)) and (x < (x_j - x_i) * (y - y_i) / (y_j - y_i) + x_i):
                    inside = not inside
                j = i
            if inside:
                foundNucID = NucID
                foundPolyID = PolyID
                foundPolyTextID = PolyTextID
                break  # Stop searching after finding the first matching polygon
        return foundNucID, foundPolyID, foundPolyTextID
    def remove_ROI(self,event):
        #NEED TO ADD IN LOGIC OF kp_remove
        selectedchannel = self.KP_rb_selection.get()
        channel = selectedchannel[3:]
        # Determine ROI array to add to 
        if selectedchannel.startswith("F1_"):
            file = 1
            active_channel_arr = self.F1_ROI_dict[channel]
            ROI_radius_entry = self.F1_roi_radius_entry_dict[selectedchannel].get()
        elif selectedchannel.startswith("F2_"):
            file = 2
            active_channel_arr = self.F2_ROI_dict[channel]
            ROI_radius_entry = self.F2_roi_radius_entry_dict[selectedchannel].get()

        # The temp array will be constructed of all existing ROIs except those the one being removed
        temparray = active_channel_arr

        if len(temparray) == 0: return
        for row in active_channel_arr:
            print(row)
            x1, y1, x2, y2 = self.canvas.coords(row[1])
            bboxid = self.canvas.create_rectangle(x1, y1, x2, y2, outline="white", fill="white",tags='bounding_box')

            mousex = self.canvas.canvasx(event.x)
            mousey = self.canvas.canvasy(event.y) 

            overlapping_items = self.canvas.find_overlapping(mousex, mousey, mousex, mousey)
            if overlapping_items:
                filteredarray = temparray
                for item in overlapping_items:
                    tag = self.canvas.gettags(item)
                    print(tag)
                    if 'bounding_box' in tag:
                        print("Clicked on oval:", row[1])
                        self.canvas.delete(row[1])
                        filteredarray = []
                        for trow in temparray:
                            if trow[1] != row[1]:
                                filteredarray.append(trow)
                temparray=filteredarray
            self.canvas.delete(bboxid)    

        # Reset the array without the removed ROI
        if selectedchannel.startswith("F1_"):
            self.F1_ROI_dict[channel] = temparray
        elif selectedchannel.startswith("F2_"):
            self.F2_ROI_dict[channel] = temparray 

        print(self.F1_ROI_dict)
    def toggle_kp_visible(self):
        pass
    ########################################################
    ##################_ARROW_FUNCTIONS_#####################
    def make_arrow(self,event):
        print("ARROW!")

        eventx = self.canvas.canvasx(event.x)
        eventy = self.canvas.canvasy(event.y)   
        if self.arrow_sp is None:
            # Store the starting point (first click)
            self.arrow_sp = (eventx, eventy)
            print(f"ARROW START {self.arrow_sp}")
        else:
            # Get the ending point (second click) and draw the arrow
            self.arrow_ep = (eventx, eventy)
            
            print(f"ARROW END {self.arrow_ep}")
            # Draw an arrow from the start point to the end point
            arrowid = self.canvas.create_line(self.arrow_sp[0], self.arrow_sp[1], self.arrow_ep[0], self.arrow_ep[1], arrow=tk.LAST, width=2,tags="arrow", fill="white")
            

            #Get arrowIndex
            arrowInd = None         #the arrows global arrow index
            arrowarr = self.arrows_arr      #self.arrows_arr = [] #[arrowIndex,arrowID,x1,y1,x2,y2,arrowTextID,text_x,text_y]
            arrowarr = sorted(arrowarr, key=lambda x: x[0]) #sort them in ascending order of arrowIndex
            if len(arrowarr) == 0: arrowInd = 1     #if the arrow array is empty assign it as 1
            else:
                missing_numbers = []
                expected_number = 1
                for row in arrowarr:
                    while row[0] > expected_number:
                        missing_numbers.append(expected_number)
                        expected_number += 1
                    expected_number = row[0] + 1
                if len(missing_numbers) > 0: arrowInd = missing_numbers[0]
                else: arrowInd = len(arrowarr) + 1
            
            
            arrowtextid = self.generate_arrow_text(arrowInd)

            aX1 = self.arrow_sp[0]
            aY1 = self.arrow_sp[1]
            aX2 = self.arrow_ep[0]
            aY2 = self.arrow_ep[1]

            print(self.canvas.coords(arrowtextid)[0])
            print(self.canvas.coords(arrowtextid)[1])
            atextX = self.canvas.coords(arrowtextid)[0]
            atextY = self.canvas.coords(arrowtextid)[1]



            newarrowinforow = [arrowInd,arrowid,aX1,aY1,aX2,aY2,arrowtextid,atextX,atextY]
            self.arrows_arr.append(newarrowinforow)

            print(self.arrows_arr)
            #self.arrows_arr = [] #[arrowIndex,arrowID,x1,y1,x2,y2,arrowTextID,text_x,text_y]
            # Reset the start point for the next arrow
            self.arrow_sp = None   
    def generate_arrow_text(self,arrowindex):
        x1,y1 = self.arrow_sp
        x2,y2 = self.arrow_ep
        x = int((x1+x2)/2)
        y = int((y1+y2)/2)

        arrow_ID = arrowindex
        arrowtextid =self.canvas.create_text(x, y, text=str(arrow_ID), font=('Arial', 20), fill='magenta', tags='arrowtext')
        return arrowtextid
    def remove_arrow(self,event):
        print("remove arrow")
        mousex = self.canvas.canvasx(event.x)
        mousey = self.canvas.canvasy(event.y) 
        overlapping_items = self.canvas.find_overlapping(mousex, mousey, mousex, mousey)
        print(overlapping_items)
        if overlapping_items:
            for item in overlapping_items:
                tag = self.canvas.gettags(item)
                print(tag)
                if 'arrow' in tag:
                    print("Clicked on arrow:")
                    filterarr = []
                    for trow in self.arrows_arr: #self.arrows_arr = [] #[arrowIndex,arrowID,x1,y1,x2,y2,arrowTextID,text_x,text_y]
                        if trow[1] != item:
                            filterarr.append(trow)
                        if trow[1] == item:
                            self.canvas.delete(trow[6])
                    self.arrows_arr = filterarr
                    print(self.arrows_arr)
                    self.canvas.delete(item)
                if 'arrowtext' in tag:
                    print("Clicked on arrowtext:")
                    filterarr = []
                    for trow in self.arrows_arr: #self.arrows_arr = [] #[arrowIndex,arrowID,x1,y1,x2,y2,arrowTextID,text_x,text_y]
                        if trow[6] != item:
                            filterarr.append(trow)
                        if trow[6] == item:
                            self.canvas.delete(trow[1])
                    self.arrows_arr = filterarr
                    print(self.arrows_arr)
                    self.canvas.delete(item)
    ########################################################
    ########################################################
    ##################_CAPTURE_IMG_#########################  
    def safe_color(self,color):
        if not color or color.strip() == '':
            return None
        color = color.lower()
        if color.startswith('system'):  # Filter out system colors
            return 'black'  # fallback color
        return color
    def _draw_arrowhead(self, draw, p1, p2, fill):
        from math import atan2, cos, sin, pi

        # Length and angle of arrowhead
        arrow_length = 10  # pixels
        arrow_angle = pi / 6  # 30 degrees

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = atan2(dy, dx)

        left_angle = angle + pi - arrow_angle
        right_angle = angle + pi + arrow_angle

        left_x = p2[0] + arrow_length * cos(left_angle)
        left_y = p2[1] + arrow_length * sin(left_angle)
        right_x = p2[0] + arrow_length * cos(right_angle)
        right_y = p2[1] + arrow_length * sin(right_angle)

        draw.polygon([p2, (left_x, left_y), (right_x, right_y)], fill=fill) 
    def capture_image(self):
        self.capimg_count += 1 
        export_dir = cfg.CAPTURE_IMG_DIR
        #THIS CAPTURES WITH DRAWN WIDGETS -------
        # Get the coordinates of the canvas
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        #self.master.attributes("-topmost", True)
        self.control_window.lower(self.master)
        self.control_window.update() 
        # Capture the canvas content using ImageGrab
        img = ImageGrab.grab((x, y, x1, y1))
        
        # STEP 6: Save final image with widgets
        output_name = f"{self.capimg_count}_img_widget_{self.imscale:.2f}x.tiff"
        output_path = os.path.join(export_dir, output_name)
        img.save(output_path, 'TIFF')

        
        # STEP 7: Save image without widgets (if exportimg exists)
        filename = f"{self.capimg_count}_img_NOwidget_{self.imscale:.2f}x.tiff"
        nowidget_path = os.path.join(export_dir, filename)
        self.exportimg._PhotoImage__photo.write(nowidget_path)               
    ########################################################
    ##################_SCALEBAR_############################   
    def toggle_scalebar(self):
        if self.scalebar_visible:
            # Hide the scale bar
            self.canvas.delete("scalebar")
        else:
            # Show the scale bar
            self.draw_scalebar()  # Adjust the length of the scale bar as needed
        self.scalebar_visible = not self.scalebar_visible
    def draw_scalebar(self):
        print("DRAWING SCALEBAR")
        #1px =  0.108333333333333 microns
        # Determine the position of the scale bar relative to the image
        # For example, you may want to place it at the bottom-right corner
        self.canvas.delete('scalebar')

        if self.scalebar_toggle_var.get():
            px_per_micron = 0.108333333333333
            scalebar_microns = 10
            scale_bar_width = scalebar_microns / px_per_micron * self.imscale
            scale_bar_height = 5  # Height of the scale bar rectangle


            # Calculate the coordinates for the scale bar
            window_width = self.master.winfo_width() 
            window_height = self.master.winfo_height() 
            scale_bar_x0 = self.bbox2[0] + 20    # Adjust as needed
            scale_bar_y0 = self.bbox2[3] - 20  # Adjust as needed
            scale_bar_x1 = scale_bar_x0 + scale_bar_width   # Adjust as needed
            scale_bar_y1 = scale_bar_y0 - scale_bar_height  # Adjust as needed

            # Draw the scale bar
            self.scalebarwidget = self.canvas.create_rectangle(scale_bar_x0, scale_bar_y0, scale_bar_x1, scale_bar_y1, fill="white", tags="scalebar")

            # Optionally, add text to indicate the length of the scale bar
            scale_text_x = scale_bar_x0 + 50
            scale_text_y = scale_bar_y0 - 15  # Adjust as needed
            self.canvas.create_text(scale_text_x, scale_text_y, text=f"{scalebar_microns} um ({round(scale_bar_width,1)} px)", fill="white", tags="scalebar")
        print("FINISHED DRAWING SCALEBAR")   
    ########################################################
    ##################_FINALIZE_FUNCTIONS_##################    
    def finalize_ROI_picking(self):
        print("FINALIZING ROI PICKING")
     
        #Set Coordinates back to base

        self.F1_exp_arr = []   #array to iterate through for recalculating scaling and offset
        self.F2_exp_arr = []   #array to iterate through for recalculating scaling and offset

        for channel_arr in self.F1_ROI_dict.values():
            self.F1_exp_arr.append(channel_arr)

        #[kpID,kpOvalID,kp_x,kp_y,nucInd,polyID,polyTextID]

        for channel_arr in self.F2_ROI_dict.values():
            self.F2_exp_arr.append(channel_arr)

        #Move and rescale the canvas and kps
        dx = -self.canvas.coords(self.container)[0]
        dy = -self.canvas.coords(self.container)[1]
        self.canvas.move('all', dx,dy)
        self.canvas.scale('all', 0,0, 1/self.imscale, 1/self.imscale)
  
    
        #SHIFT KPS in F1 and F2 into final export array self.f_exp_arr
        self.f_exp_arr = []
        self.f1_full_exp_arr = []
        self.f2_full_exp_arr = []
        self.f1_draw_arr = [] #for drawing on output images
        self.f2_draw_arr = [] #for drawing on output images

        #File 1 recalculate kp_x and kp_y
        for i,chanarr in enumerate(self.F1_exp_arr):
            temparray = []
            file_id = 1
            channel = self.f1_cs[i]
            # Get channel ROI radius
            radius = self.F1_roi_radius_entry_dict[f"F1_{channel}"]
            chanrad = -1
            if radius is not None:
                radius_value = radius.get()
                chanrad = float(radius_value)
            # Get channel coloc threshold
            coloc = self.F1_coloc_entry_dict[f"F1_{channel}"]
            chan_e = -1
            if coloc is not None:
                coloc_value = coloc.get()
                chan_e = float(coloc_value)
            # Processs all ROIs in the channel set
            for row in chanarr:
               kpID,kpOvalID,kp_x,kp_y,nucInd,polyID,polyTextID = row #[kpID,kpOvalID,kp_x,kp_y,nucInd,polyID,polyTextID]

               kpcoords = self.canvas.coords(kpOvalID)
               t_kpx = (kpcoords[0]+kpcoords[2])/2
               t_kpy = (kpcoords[1]+kpcoords[3])/2
               f_kp_x = t_kpx - abs(self.padx)
               f_kp_y = t_kpy - abs(self.pady)

               if f_kp_x <= 0 or f_kp_y <= 0: pass
               else:
                temprow = file_id,channel,chanrad,chan_e,kpID,kpOvalID,f_kp_x,f_kp_y,nucInd,polyID,polyTextID
                temparray.append(temprow)

            self.f_exp_arr.append(temparray)
            self.f1_draw_arr.append(temparray)
            self.f1_full_exp_arr.append(temparray)
        #File 2 recalculate kp_x and kp_y
        for i,chanarr in enumerate(self.F2_exp_arr):
            temparray = []
            file_id = 2
            channel = self.f2_cs[i]
             # Get channel ROI radius
            radius = self.F2_roi_radius_entry_dict[f"F2_{channel}"]
            chanrad = -1
            if radius is not None:
                radius_value = radius.get()
                chanrad = float(radius_value)
            # Get channel coloc threshold
            coloc = self.F2_coloc_entry_dict[f"F2_{channel}"]
            chan_e = -1
            if coloc is not None:
                coloc_value = coloc.get()
                chan_e = float(coloc_value)
            # Processs all ROIs in the channel set
            for row in chanarr:
                kpID,kpOvalID,kp_x,kp_y,nucInd,polyID,polyTextID = row #[kpID,kpOvalID,kp_x,kp_y,nucInd,polyID,polyTextID]
        #        print(f"{kpID} - {kp_x},{kp_y}")
        #        print(f"padx,pady {self.padx},{self.pady}")

                kpcoords = self.canvas.coords(kpOvalID)
                t_kpx = (kpcoords[0]+kpcoords[2])/2
                t_kpy = (kpcoords[1]+kpcoords[3])/2
                f_kp_x = t_kpx - abs(self.padx)
                f_kp_y = t_kpy - abs(self.pady)

                if self.padx < 0:
                    f_kp_x = f_kp_x + abs(self.padx)
                if self.padx > 0:
                    f_kp_x = f_kp_x - abs(self.padx) 
                if self.padx == 0:
                    f_kp_x = f_kp_x

                if self.pady < 0:
                    f_kp_y = f_kp_y + abs(self.pady)
                if self.pady > 0:
                    f_kp_y = f_kp_y - abs(self.pady) 
                if self.pady == 0:
                    f_kp_y = f_kp_y

                if f_kp_x <= 0 or f_kp_y <= 0: pass
                else:
                    temprow = temprow = file_id,channel,chanrad,chan_e,kpID,kpOvalID,f_kp_x,f_kp_y,nucInd,polyID,polyTextID
                    temparray.append(temprow)

            self.f_exp_arr.append(temparray)
            self.f2_draw_arr.append(temparray)
            self.f2_full_exp_arr.append(temparray)
        
        #Arrows recalculate coords
        self.arrow_exp_arr = []
        for row in self.arrows_arr:
            temparray = []
            arrowind, arrowid,x1, y1, x2, y2,atextid,atextx,atexty = row #[arrowIndex,arrowID,x1,y1,x2,y2,arrowTextID,text_x,text_y]
            coords = self.canvas.coords(arrowid)
            f_x1 = coords[0] - abs(self.padx)
            f_y1 = coords[1] - abs(self.pady)
            f_x2 = coords[2] - abs(self.padx)
            f_y2 = coords[3] - abs(self.pady)
            temprow = arrowind, arrowid,f_x1, f_y1, f_x2, f_y2,atextid,atextx,atexty
            temparray.append(temprow)

            self.arrow_exp_arr.append(temparray)

        #1- EXPORT BUNDLE - KPS PER CHANNEL AND NUCLEI
        aparams.kpchan_kpnuc_xbundle = self.f_exp_arr
        aparams.f1_full_exp_arr = self.f1_full_exp_arr
        aparams.f2_full_exp_arr = self.f2_full_exp_arr
        #------------------------#
        #2- EXPORT BUNDLE - ROI RADIUS OF EACH CHANNEL
        # NEED TO DO

       # self.F1_roi_radius_entry_dict = {}
       # self.F2_roi_radius_entry_dict = {}

        self.ROI_xbundle = []

        for radius in self.F1_roi_radius_entry_dict.values():
            chanrad = -1
            if radius is not None:
                radius_value = radius.get()
                chanrad = float(radius_value)
                self.ROI_xbundle.append(chanrad)
        for radius in self.F2_roi_radius_entry_dict.values():
            chanrad = -1
            if radius is not None:
                radius_value = radius.get()
                chanrad = float(radius_value)
                self.ROI_xbundle.append(chanrad)


        aparams.kpchan_ROIradius_xbundle = self.ROI_xbundle

        #------------------------#
        #3- EXPORT BUNDLE - COLOCALIZATION TOLERANCE CUTOFF
        # NEED TO DO

       # self.F1_coloc_entry_dict = {}
       # self.F2_coloc_entry_dict = {}

        self.COLOC_xbundle = []

        for coloc in self.F1_coloc_entry_dict.values():
            chan_e = -1
            if coloc is not None:
                coloc_value = coloc.get()
                chan_e = float(coloc_value)
                self.COLOC_xbundle.append(chan_e)
        for coloc in self.F2_coloc_entry_dict.values():
            chan_e = -1
            if coloc is not None:
                coloc_value = coloc.get()
                chan_e = float(coloc_value)
                self.COLOC_xbundle.append(chan_e)


        aparams.kpchan_coloc_xbundle = self.COLOC_xbundle

        #------------------------#
        #4- EXPORT BUNDLE - ANALYSIS TOGGLES OUT

        #MOVED IN THIS VERSION
        #------------------------#
        #5- EXPORT BUNDLE - ARROWS
        aparams.arrows_xbundle = self.arrows_arr #[arrowIndex,arrowID,x1,y1,x2,y2,arrowTextID,text_x,text_y]

        #------------------------#

        # TODO NEED TO IMPLEMENT FINAL FULL CAPTURE
        #self.final_fullcapture()      
        #----
        print("FINISHED KP PICKING")
        self.switch_gui()
        #self.master.destroy()  
    def switch_gui(self):
        self.switch()  
    def final_fullcapture(self):

        #GENERATE FULL SIZED IMG COMPOSITE
        f1f2compimg = self.generate_final_f1f2compimg()
        f1compimg, f1mipbundle = self.generate_final_f1_imgs()
        f2compimg, f2mipbundle = self.generate_final_f2_imgs()

        self.draw_scalebar_imgs(f1f2compimg)
        self.draw_scalebar_imgs(f1compimg)
        self.draw_scalebar_imgs(f2compimg)
        for mip in f1mipbundle: self.draw_scalebar_imgs(mip)
        for mip in f2mipbundle: self.draw_scalebar_imgs(mip)
        #-------------------------------------------------------------------#
        #-------------------------------------------------------------------#
        #-------------------------------------------------------------------#
        #F1 - F2 COMPOSITE --------#
        #-NO WIDGET-#
        export_dir = "Outputs/IMAGES"
        output_name = f"f1f2_composite_NOwidget.tiff"
        output_path = os.path.join(export_dir, output_name)
        f1f2compimg.save(output_path, 'TIFF')
        print(f"Image saved at: {output_path}")
        #-WIDGET-#
        #DRAW F1F2 MIP COMPOSITE
        f1f2WIDGETimg = f1f2compimg.copy()
        self.drawf1f2composite(f1f2WIDGETimg)
        #SAVE the WIDGET VERSION ------
        # Define the export directory and output path
        export_dir = "Outputs/IMAGES"
        output_name = f"f1f2_composite_widget.tiff"
        output_path = os.path.join(export_dir, output_name)
        f1f2WIDGETimg.save(output_path, 'TIFF')
        print(f"Image saved at: {output_path}")
        #-------------------------------------------------------------------#
        #-------------------------------------------------------------------#
        #F1 COMPOSITE
        #-NO WIDGET-#
        export_dir = "Outputs/IMAGES"
        output_name = f"f1composite.tiff"
        output_path = os.path.join(export_dir, output_name)
        f1compimg.save(output_path, 'TIFF')
        print(f"Image saved at: {output_path}")
        #-WIDGET-#
        f1compWIDGETimg = f1compimg.copy()
        self.drawf1composite(f1compWIDGETimg)
        #SAVE the WIDGET VERSION ------
        # Define the export directory and output path
        export_dir = "Outputs/IMAGES"
        output_name = f"f1composite_WIDGET.tiff"
        output_path = os.path.join(export_dir, output_name)
        f1compWIDGETimg.save(output_path, 'TIFF')
        print(f"Image saved at: {output_path}")
        #-------------------------------------------------------------------#
        #F1 SINGLE MIPS
        #-NO WIDGET and WIDGET-#
        self.f1mipsaver(f1mipbundle)
        #-------------------------------------------------------------------#
        #-------------------------------------------------------------------#
        #F2 COMPOSITE
        #-NO WIDGET-#
        export_dir = "Outputs/IMAGES"
        output_name = f"f2composite.tiff"
        output_path = os.path.join(export_dir, output_name)
        f2compimg.save(output_path, 'TIFF')
        print(f"Image saved at: {output_path}")
        #-WIDGET-#
        f2compWIDGETimg = f2compimg.copy()
        self.drawf2composite(f2compWIDGETimg)
        #SAVE the WIDGET VERSION ------
        # Define the export directory and output path
        export_dir = "Outputs/IMAGES"
        output_name = f"f2composite_WIDGET.tiff"
        output_path = os.path.join(export_dir, output_name)
        f2compWIDGETimg.save(output_path, 'TIFF')
        print(f"Image saved at: {output_path}")
        #-------------------------------------------------------------------#
        #F2 SINGLE MIPS
        #-NO WIDGET and WIDGET-#
        self.f2mipsaver(f2mipbundle)
        #-------------------------------------------------------------------#
        self.master.update() 
        return
     
class AutoScrollbar(tk.Scrollbar):
    ''' A scrollbar that hides itself if it's not needed.
        Works only if you use the grid geometry manager '''
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            tk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with this widget')

    def place(self, **kw):
        raise tk.TclError('Cannot use place with this widget')
   