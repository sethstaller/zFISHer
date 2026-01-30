import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
#from nd2reader import ND2Reader

#import zFISHer.config.config_manager as cfgmgr
import zFISHer.processing.regXY as regXY
import zFISHer.utils.config as cfg

import time

"""
This creates the gui that let's the user view and modify 2D XY registration. 
"""

class RegistrationXYGUI():
    """
    The class that allows the user to set the XY offset between the two input files.
    """

    def __init__(self, master, switch_to_gui_two,logger):

        #GET XY OFFSET
        regXY.start_regXY(cfg.F1_REG_C,cfg.F2_REG_C)

        self.x_offset = cfg.OFFSET_X
        self.y_offset = cfg.OFFSET_Y
       # self.x_offset = cfgmgr.get_config_value("X_OFFSET")
       #self.y_offset = cfgmgr.get_config_value("Y_OFFSET")
        self.f2offset = self.x_offset,self.y_offset
        self.switch = switch_to_gui_two
        self.logger = logger
        #----
        self.f1_ntag = cfg.F1_NTAG
        self.f2_ntag = cfg.F2_NTAG
        #self.f1_channels = cfgmgr.get_config_value("FILE_2_NAMETAG")

        self.f1_cs = cfg.F1_C_LIST
        self.f2_cs = cfg.F2_C_LIST

        self.processing_dir = cfg.PROCESSING_DIR


        self.f1_reg_c = self.get_reg_channel(self.f1_cs)
        self.logger.log_message(f"Channel 1 registration channel: INDEX:{self.f1_reg_c} - NAME:{self.f1_cs[self.f1_reg_c]}")
        self.f2_reg_c= self.get_reg_channel(self.f2_cs)
        self.logger.log_message(f"Channel 2 registration channel: INDEX:{self.f2_reg_c} - NAME:{self.f2_cs[self.f2_reg_c]}")   

        self.F1_base_path = os.listdir(os.path.join(os.path.join(self.processing_dir,self.f1_ntag,self.f1_cs[self.f1_reg_c],"MIP"))[0])
        self.F2_base_path = os.listdir(os.path.join(os.path.join(self.processing_dir,self.f2_ntag,self.f2_cs[self.f2_reg_c],"MIP"))[0])

        self.f1dapichan = None
        self.f2dapichan = None
        for index, chan in enumerate(self.f1_cs):
            if chan == "DAPI":
                self.f1dapichan = index
                self.F1_DAPI_dir = f"C{self.f1dapichan}_DAPI"
                
        for index, chan in enumerate(self.f2_cs):
            if chan == "DAPI":
                self.f2dapichan = index
                self.F2_DAPI_dir = f"C{self.f2dapichan}_DAPI"

        self.F1_C0B_MIP_path = os.path.join(os.path.join(self.processing_dir,self.f1_ntag,self.f1_cs[self.f1_reg_c],"MIP"))
        self.F2_C0B_MIP_path = os.path.join(os.path.join(self.processing_dir,self.f2_ntag,self.f2_cs[self.f2_reg_c],"MIP"))

        #IMAGES h=2044 w= 2048
        self.F1_C0B_MIP = self.get_MIP_file(self.F1_C0B_MIP_path)
        self.F2_C0B_MIP = self.get_MIP_file(self.F2_C0B_MIP_path)
  
        #----------------------------------------------
       
        #-----------------------------------------------------------------------#
        #Offset on drawn canvas
        self.f2_offset_x = 0
        self.f2_offset_y = 0

        #Offset on base image --> passed to other classes
        self.f2_offset_x_scaled = 0
        self.f2_offset_y_scaled = 0
        #-----------------------------------------------------------------------#
        #Initial Opacity
        self.opac = 0.5
        self.myscalex = 1
        self.myscaley = 1

        #Define Master Window (Overlay)
        self.master = master
        self.master.title("zFISHer --- XY Registration")

        self.hwmultiplier = self.F1_C0B_MIP.height
        self.screen_width = master.winfo_screenwidth() * 0.8
        self.screen_height = master.winfo_screenheight() * 0.8
        self.hwmult = self.F1_C0B_MIP.width/self.F1_C0B_MIP.height

        #Create canvas
        self.canvas = tk.Canvas(master, width=self.screen_height*self.hwmult, height=self.screen_height)
        self.canvas.pack(side=tk.BOTTOM)
        self.master.update_idletasks()

        #region ---CREATE TOGGLE WINDOW---
        self.control_window = tk.Toplevel(self.master)
        self.control_window.title("Control Panel - Registration")

        #FILENAME FRAME
        self.fnameframe = tk.Frame(self.control_window)
        self.fnameframe.grid(row=1, column=1, columnspan=4)

        f1 = os.listdir(os.path.join(os.path.join(self.processing_dir,self.f1_ntag,self.f1_cs[self.f1_reg_c],"MIP")))[0]
        f2 = os.listdir(os.path.join(os.path.join(self.processing_dir,self.f2_ntag,self.f2_cs[self.f2_reg_c],"MIP")))[0]
        f1 = Image.open(os.path.join(self.F1_C0B_MIP_path,f1))
        f2 = Image.open(os.path.join(self.F2_C0B_MIP_path,f2))
        self.f1width, self.f1height = f1.size
        self.f2width, self.f2height = f2.size

        self.f1_filename_lab = tk.Label(self.fnameframe, text=f"{self.f1_ntag}(FIXED): {self.f1_ntag}")
        self.f1_filename_lab.grid(row=1, column=0, columnspan=2)
        self.f1_fdim_lab = tk.Label(self.fnameframe, text =f"{self.f1width}x{self.f1height}")
        self.f1_fdim_lab.grid(row=1, column=2, columnspan=2)

        self.f2_filename_lab = tk.Label(self.fnameframe, text=f"{self.f2_ntag}(MOVING): {self.f2_ntag}")
        self.f2_filename_lab.grid(row=2, column=0, columnspan=2)
        self.f2_fdim_lab = tk.Label(self.fnameframe, text =f"{self.f2width}x{self.f2height}")
        self.f2_fdim_lab.grid(row=2, column=2, columnspan=2)

        self.canvas_size_lab = tk.Label(self.fnameframe, text =f"Canvas Size: {self.canvas.winfo_width()} x {self.canvas.winfo_height()}")
        self.canvas_size_lab.grid(row=3, column=0, columnspan=4)
       #TOGGLES
        self.togglesframe = tk.Frame(self.control_window)
        self.togglesframe.grid(row=2, column=1, columnspan=4)
        self.list_label = tk.Label(self.togglesframe, text="-----FILE_CHANNEL_TOGGLE-----")
        self.list_label.grid(row=1, column=1, columnspan=4)
        self.F1_C0B_toggle = tk.BooleanVar(value=True)

        self.F2_C0B_toggle = tk.BooleanVar(value=True)

        #-spacer1
        self.spacer_label = tk.Label(self.control_window, text="------------------------------------")
        self.spacer_label.grid(row=4, column=1, columnspan=4)
        #F2 OPAC ENTRY
        self.opacframe = tk.Frame(self.control_window)
        self.opacframe.grid(row=6, column=1, columnspan=4)
        self.f2_opac_lab = tk.Label(self.opacframe, text=f"{self.f2_ntag} (F2)  Transparency: " )
        self.f2_opac_lab.grid(row=1,column=1)
        self.opac_e_text = tk.StringVar(value= self.opac)
        self.f2_opac_entry = tk.Entry(self.opacframe, width=5, textvariable=self.opac_e_text)
        self.f2_opac_entry.grid(row=1,column=2)
        self.f2_opac_e_butt = tk.Button(self.opacframe, text=f"Set {self.f2_ntag} (F2) % ", pady = 2, command=self.setf2opac)
        self.f2_opac_e_butt.grid(row=1,column=3)
        #SCALEBAR TOGGLE
        self.scalebar_toggle_var = tk.BooleanVar(value=True)
        self.scalebar_toggle_checkbox = tk.Checkbutton(self.control_window,text="Show Scalebar", variable=self.scalebar_toggle_var, command=self.toggle_scalebar)
        self.scalebar_toggle_checkbox.grid(row=7, column=1, columnspan=4)
        #-spacer2
        self.spacer2_label = tk.Label(self.control_window, text="------------------------------------")
        self.spacer2_label.grid(row=8, column=1, columnspan=4)
        #OFFSET ENTRY
        self.offentframe = tk.Frame(self.control_window)
        self.offentframe.grid(row=9, column=1,columnspan=4)
        self.c_os_x_l = tk.Label(self.offentframe, text="X:", fg="white")
        self.c_os_x_l.grid(row=9, column=1, sticky=tk.E)
        self.c_os_x_e_text = tk.StringVar(value= self.f2_offset_x)
        self.c_os_x_e = tk.Entry(self.offentframe, width=5, textvariable=self.c_os_x_e_text)
        
        self.c_os_x_e.grid(row=9, column=2, sticky=tk.W)
        self.c_os_x_e.insert(0,self.f2_offset_x)
        
        self.c_os_y_l = tk.Label(self.offentframe, text="Y:", fg="white")
        self.c_os_y_l.grid(row=9, column=3, sticky=tk.E)
        self.c_os_y_e_text = tk.StringVar(value= self.f2_offset_y)
        self.c_os_y_e = tk.Entry(self.offentframe, width=5, textvariable=self.c_os_y_e_text)
        
        self.c_os_y_e.grid(row=9, column=4, sticky=tk.W)
        self.c_os_y_e.insert(0,self.f2_offset_y)

        self.c_os_x_e_text.trace_add("write", self.on_offset_entry)
        self.c_os_y_e_text.trace_add("write", self.on_offset_entry)

        self.os_ent_but = tk.Button(self.offentframe, text="Set Offset", pady = 2, command=self.manualoffset)
        self.os_ent_but.grid(row=10, column=1, columnspan=5)
        #OFFSET LABEL
        self.offsetframe = tk.Frame(self.control_window)
        self.offsetframe.grid(row=10, column=1, columnspan=4)
        self.offset_label = tk.Label(self.offsetframe, text="Canvas Offset: (0,0)", fg="white")
        self.offset_label.grid(row=1, column=1, columnspan=4)
        self.f2offset_label = tk.Label(self.offsetframe, text=f"{self.f2_ntag} (F2) Offset: (0,0)", fg="white")
        self.f2offset_label.grid(row=2, column=0, columnspan=4)
        #-spacer3
        self.spacer2_label = tk.Label(self.control_window, text="------------------------------------")
        self.spacer2_label.grid(row=11, column=1, columnspan=4)
        #RESET ALL
        self.reset_button = tk.Button(self.control_window, text="RESET ALL", pady=4, command=self.reset_all)
        self.reset_button.grid(row=12, column=1, columnspan=4,pady=5)
        #-spacer4
        self.spacer2_label = tk.Label(self.control_window, text="------------------------------------")
        self.spacer2_label.grid(row=13, column=1, columnspan=4)

        #Finish Button
        self.finish_button = tk.Button(self.control_window, text=f"Finalize {self.f2_ntag} (F2) Offset", pady=12, command=self.finalize_offset)
        self.finish_button.grid(row=14, column=0, columnspan=5, pady=5)
        #endregion

        self.y_scaler = self.f1height/ self.canvas.winfo_height()
        self.x_scaler = self.f1width/ self.canvas.winfo_width()

        self.toggle_channels()

        self.draw_scalebar()
        self.scalebar_visible = True

        self.c_os_x_e_text.set(self.x_offset)
        self.c_os_y_e_text.set(self.y_offset)
        self.manualoffset()

    def setup_window(self):
        pass


    def get_reg_channel(self, channels):
        for index,channel in enumerate(channels):
            if channel == "DAPI":
                return index
        # Bind mouse events


    def setf2opac(self):
        newopac = float(self.opac_e_text.get())
        self.opac = newopac
        self.toggle_channels()


    def manualoffset(self):
        x = float(self.c_os_x_e_text.get())
        y = float(self.c_os_y_e_text.get())

        #The input x,y is in native image dimensions
        self.f2_offset_x = x / self.x_scaler
        self.f2_offset_y = y / self.y_scaler
        self.f2_offset_x_scaled = x 
        self.f2_offset_y_scaled = y      

        decimal_places = 2
        self.c_os_x_e_text.set(f"{self.f2_offset_x_scaled:.{decimal_places}f}")
        self.c_os_y_e_text.set(f"{self.f2_offset_y_scaled:.{decimal_places}f}")

        self.offset_label.config(text=f"Canvas Offset: ({round(self.f2_offset_x,2)},{round(self.f2_offset_y,2)})")
        self.f2offset_label.config(text=f"{self.f2_ntag} (F) Offset: ({round(self.f2_offset_x_scaled,2)},{round(self.f2_offset_y_scaled,2)})")

        self.canvas.coords(self.f2comp,self.f2_offset_x,self.f2_offset_y)


    def reset_all(self):
        self.f2_offset_x = 0
        self.f2_offset_y = 0
        self.f2_offset_x_scaled = 0
        self.f2_offset_y_scaled = 0
        self.c_os_x_e_text.set(self.f2_offset_x)
        self.c_os_y_e_text.set(self.f2_offset_y)
        self.offset_label.config(text=f"Offset: ({round(self.f2_offset_x,2)},{round(self.f2_offset_y,2)})")
        self.f2offset_label.config(text=f"{self.f2_ntag} (F2) Offset: ({round(self.f2_offset_x_scaled,2)},{round(self.f2_offset_y_scaled,2)})")
        self.opac = 0.5
        self.opac_e_text.set(self.opac)
        self.myscalex = 1
        self.myscaley = 1
        self.F1_C0B_toggle.set(True)
        self.F2_C0B_toggle.set(True)
        self.scalebar_toggle_var.set(True)
        self.scalebar_visible = True


        self.toggle_channels()
        self.draw_scalebar()


    def on_offset_entry(self, *args):
        pass
       #x_text = self.c_os_x_e_text.get()
       #y_text = self.c_os_y_e_text.get()
       #if x_text and y_text:
        #    self.f2_offset_x = float(self.c_os_x_e_text.get())
         #   self.f2_offset_y = float(self.c_os_y_e_text.get())


    def f1_toggle_imgprocessor(self):
        self.added_image = None
        #F1C0B
        if self.F1_C0B_MIP is not None and self.F1_C0B_toggle.get():
                    self.F1C0B_p = self.F1_C0B_MIP.copy()
                    self.F1C0B_p = np.array(self.F1C0B_p)
                    self.pil_image = Image.fromarray(self.F1C0B_p)
                    self.added_image = self.pil_image.copy()
        #F1C1G
        #if self.F1_C1G_MIP is not None and self.F1_C1G_toggle.get():  
        #    self.F1C1G_p = self.F1_C1G_MIP.copy()  
        #    if self.added_image is None:
        #        self.F1C1G_p = np.array(self.F1C1G_p)
        #        self.pil_image = Image.fromarray(self.F1C1G_p)
        #       self.added_image = self.pil_image.copy()
         #   else:
         #       self.added_image = np.array(self.added_image)
         #       self.F1C1G_p = np.array(self.F1C1G_p)
         #       self.sum_image = cv2.add(self.added_image,self.F1C1G_p)
         #       self.pil_image = Image.fromarray(self.sum_image)
        #        self.added_image = self.pil_image.copy()
        #F1C2R
       # if self.F1_C2R_MIP is not None and self.F1_C2R_toggle.get():  
       #     self.F1C2R_p = self.F1_C2R_MIP.copy()  
       #     if self.added_image is None:
       #         self.F1C2R_p = np.array(self.F1C2R_p)
       #         self.pil_image = Image.fromarray(self.F1C2R_p)
       #         self.added_image = self.pil_image.copy()
       #     else:
        #        self.added_image = np.array(self.added_image)
        #        self.F1C2R_p = np.array(self.F1C2R_p)
        #        self.sum_image = cv2.add(self.added_image,self.F1C2R_p)
        #        self.pil_image = Image.fromarray(self.sum_image)
        #        self.added_image = self.pil_image.copy()
        #F1C3M
       # if self.F1_C3M_MIP is not None and self.F1_C3M_toggle.get():  
       #     self.F1C3M_p = self.F1_C3M_MIP.copy()  
       #     if self.added_image is None:
        #        self.F1C3M_p = np.array(self.F1C3M_p)
        #        self.pil_image = Image.fromarray(self.F1C3M_p)
        #        self.added_image = self.pil_image.copy()
        #    else:
         #       self.added_image = np.array(self.added_image)
         ##       self.F1C3M_p = np.array(self.F1C3M_p)
        #        self.sum_image = cv2.add(self.added_image,self.F1C3M_p)
         #       self.pil_image = Image.fromarray(self.sum_image)
         #       self.added_image = self.pil_image.copy()

        return self.added_image
    
    
    def f2_toggle_imgprocessor(self):
        self.added_image = None
        #F2C0B
        if self.F2_C0B_MIP is not None and self.F2_C0B_toggle.get():
                    self.F2C0B_p = self.F2_C0B_MIP.copy()
                    self.F2C0B_p = np.array(self.F2C0B_p)
                    self.pil_image = Image.fromarray(self.F2C0B_p)
                    self.added_image = self.pil_image.copy()
        #F2C1G
      #  if self.F2_C1G_MIP is not None and self.F2_C1G_toggle.get():  
      #      self.F2C1G_p = self.F2_C1G_MIP.copy()  
      #      if self.added_image is None:
       #         self.F2C1G_p = np.array(self.F2C1G_p)
       #         self.pil_image = Image.fromarray(self.F2C1G_p)
        #        self.added_image = self.pil_image.copy()
        #    else:
        #        self.added_image = np.array(self.added_image)
       #         self.F2C1G_p = np.array(self.F2C1G_p)
        #        self.sum_image = cv2.add(self.added_image,self.F2C1G_p)
       #         self.pil_image = Image.fromarray(self.sum_image)
       #         self.added_image = self.pil_image.copy()
        #F2C2R
      #  if self.F2_C2R_MIP is not None and self.F2_C2R_toggle.get():  
      #      self.F2C2R_p = self.F2_C2R_MIP.copy()  
      #      if self.added_image is None:
       #         self.F2C2R_p = np.array(self.F2C2R_p)
       #         self.pil_image = Image.fromarray(self.F2C2R_p)
       #         self.added_image = self.pil_image.copy()
       #     else:
       #         self.added_image = np.array(self.added_image)
       #         self.F2C2R_p = np.array(self.F2C2R_p)
         #       self.sum_image = cv2.add(self.added_image,self.F2C2R_p)
         #       self.pil_image = Image.fromarray(self.sum_image)
       #         self.added_image = self.pil_image.copy()
        #F2C3M
      #  if self.F2_C3M_MIP is not None and self.F2_C3M_toggle.get():  
       #     self.F2C3M_p = self.F2_C3M_MIP.copy()  
       #     if self.added_image is None:
       #         self.F2C3M_p = np.array(self.F2C3M_p)
       #         self.pil_image = Image.fromarray(self.F2C3M_p)
       #         self.added_image = self.pil_image.copy()
       #     else:
       #         self.added_image = np.array(self.added_image)
       #         self.F2C3M_p = np.array(self.F2C3M_p)
       #         self.sum_image = cv2.add(self.added_image,self.F2C3M_p)
       #         self.pil_image = Image.fromarray(self.sum_image)
      #          self.added_image = self.pil_image.copy()
                
        return self.added_image


    def toggle_channels(self):
        # Generate F1 and F2 composites based on toggles
        self.f1img = self.f1_toggle_imgprocessor()
        self.f2img = self.f2_toggle_imgprocessor()

        # Ensure the image is in the correct mode
        #if self.f1img.mode not in ("RGB", "RGBA"):

        self.f1img = self.f1img.convert("L")

        self.f1img = self.f1img.convert("RGBA")


        # Put F1 on canvas
        self.f1img.thumbnail((self.canvas.winfo_width(), self.canvas.winfo_height()))
        self.f1photo = ImageTk.PhotoImage(self.f1img)
        self.f1comp = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.f1photo)

        # Ensure f2img is in a compatible mode before applying alpha
        #if self.f2img.mode not in ("RGBA", "RGB"):
        self.f2img = self.f2img.convert("RGBA")

        # Generate F2 Opacity 
        # Create a mask with the desired opacity
        self.f2mask = Image.new("L", self.f2img.size, int(255 * self.opac))
        self.f2imgwopac = self.f2img.copy()
        self.f2imgwopac.putalpha(self.f2mask)
        
        self.f2imgwopac.thumbnail((self.canvas.winfo_width(),self.canvas.winfo_height()))
        self.f2photo = ImageTk.PhotoImage(self.f2imgwopac)
        
        self.f2comp = self.canvas.create_image(self.f2_offset_x, self.f2_offset_y, anchor=tk.NW, image=self.f2photo)

        # Bind mouse events
        self.canvas.tag_bind(self.f2comp, '<Button-1>', self.on_drag_start)
        self.canvas.tag_bind(self.f2comp, '<B1-Motion>', self.on_drag_motion)
        self.canvas.tag_bind(self.f2comp, '<ButtonRelease-1>', self.on_drag_release)

        self.draw_scalebar()

    #region ---DRAGGING---
    def on_drag_start(self, event):
        self.dragging = True
        self.prev_x = event.x
        self.prev_y = event.y
        self.initial_x = event.x
        self.initial_y = event.y


    def on_drag_motion(self, event):
        if self.dragging:
            dx = event.x - self.prev_x
            dy = event.y - self.prev_y
            self.canvas.move(self.f2comp, dx, dy)
            self.prev_x = event.x
            self.prev_y = event.y

            self.update_offset_label(event.x, event.y)


    def on_drag_release(self, event):
        self.dragging = False
        #self.update_offset_label(event.x, event.y)


    def update_offset_label(self, x, y):
        offset_x = x - self.initial_x
        offset_y = y - self.initial_y

        self.f2_offset_x += offset_x 
        self.f2_offset_y += offset_y  

        decimal_places = 2

        self.f1_coords = self.canvas.coords(self.f1comp)
        self.f2_coords = self.canvas.coords(self.f2comp)
        self.relative_x = self.f2_coords[0] - self.f1_coords[0]
        self.relative_y = self.f2_coords[1] - self.f1_coords[1]

        self.f2_offset_x = self.relative_x
        self.f2_offset_y = self.relative_y

        self.f2_offset_x_scaled = self.relative_x * self.x_scaler
        self.f2_offset_y_scaled = self.relative_y * self.y_scaler       

        self.c_os_x_e_text.set(f"{self.f2_offset_x_scaled:.{decimal_places}f}")
        self.c_os_y_e_text.set(f"{self.f2_offset_y_scaled:.{decimal_places}f}")

        self.offset_label.config(text=f"Canvas Offset: ({round(self.relative_x,2)},{round(self.relative_y,2)})")
        self.f2offset_label.config(text=f"F2 Offset: ({round(self.f2_offset_x_scaled,2)},{round(self.f2_offset_y_scaled,2)})")
    #endregion
    #region ---Scalebar---
    def toggle_scalebar(self):
        if self.scalebar_visible:
            # Hide the scale bar
            self.canvas.delete("scalebar")
        else:
            # Show the scale bar
            self.draw_scalebar()  # Adjust the length of the scale bar as needed
        self.scalebar_visible = not self.scalebar_visible
    
    
    def draw_scalebar(self):
        #1px =  0.108333333333333 microns
        # Determine the position of the scale bar relative to the image
        # For example, you may want to place it at the bottom-right corner
        if self.scalebar_toggle_var.get():
            px_per_micron = 0.108333333333333
            scalebar_microns = 10
            scale_bar_width = scalebar_microns / px_per_micron
            scale_bar_height = 5  # Height of the scale bar rectangle

            # Calculate the coordinates for the scale bar
            image_width = self.canvas.winfo_width()
            image_height = self.canvas.winfo_height()
            scale_bar_x0 = image_width - scale_bar_width - 10  # Adjust as needed
            scale_bar_y0 = image_height - scale_bar_height - 10  # Adjust as needed
            scale_bar_x1 = image_width - 10  # Adjust as needed
            scale_bar_y1 = image_height - 10  # Adjust as needed

            # Draw the scale bar
            self.canvas.create_rectangle(scale_bar_x0, scale_bar_y0, scale_bar_x1, scale_bar_y1, fill="white", tags="scalebar")

            # Optionally, add text to indicate the length of the scale bar
            scale_text_x = scale_bar_x0 + (scale_bar_width / 2)
            scale_text_y = scale_bar_y0 - 7  # Adjust as needed
            self.canvas.create_text(scale_text_x, scale_text_y, text=f"{scalebar_microns} um ({round(scale_bar_width,1)} px)", fill="white", tags="scalebar")
    #endregion
    #------FINALIZE------------#
    def finalize_offset(self):
        #cfgmgr.set_config_value("X_OFFSET",self.f2_offset_x_scaled)
        #cfgmgr.set_config_value("Y_OFFSET",self.f2_offset_y_scaled)
        cfg.OFFSET_X = self.f2_offset_x_scaled
        cfg.OFFSET_Y = self.f2_offset_y_scaled
        
         # Close the current window
        self.master.destroy()

        # Switch to the next GUI
        self.switch()

    #------INITIALIZATION------#
    def get_MIP_file(self,target_dir):
        files = os.listdir(target_dir)
        if len(files) == 0: return None
        else:
            image_name= os.listdir(target_dir)[0]
            image_path = os.path.join(target_dir,image_name)
            image_in = Image.open(image_path)
            normalized_img = self.normalize_image(image_in)
            return normalized_img


    def normalize_image(self,image):
        # Convert the image to a numpy array
        image_array = np.array(image).astype(np.float32)
        
        # Find the minimum and maximum pixel values
        min_val = image_array.min()
        max_val = image_array.max()
        
        # Normalize the image to the range [0, 1]
        normalized_array = (image_array - min_val) / (max_val - min_val)
        
        # Optionally, scale to [0, 255] for 8-bit representation
        normalized_array = (normalized_array * 255).astype(np.uint8)
        
        # Convert back to PIL Image
        normalized_image = Image.fromarray(normalized_array)
        return normalized_image
    

    def metadata_processor(self):
        #FILE1
        with ND2Reader(self.f1filepath) as nd2_file:
            self.f1numchan = len(nd2_file.metadata['channels'])
            self.f1numslices = len(nd2_file)
            self.f1channels = nd2_file.metadata['channels']

            self.f1pmax = float(self.f1numchan*self.f1numslices)

        #FILE2
        with ND2Reader(self.f2filepath) as nd2_file:
            self.f2numchan = len(nd2_file.metadata['channels'])
            self.f2numslices = len(nd2_file)
            self.f2channels = nd2_file.metadata['channels']

            self.f2pmax = float(self.f2numchan*self.f2numslices)

        dyn_data.f1channels = self.f1channels
        dyn_data.f2channels = self.f2channels
        dyn_data.f1numslices = self.f1numslices
        dyn_data.f2numslices = self.f2numslices
        self.f1_ntag = dyn_data.f1nametag
        self.f2_ntag = dyn_data.f2nametag
