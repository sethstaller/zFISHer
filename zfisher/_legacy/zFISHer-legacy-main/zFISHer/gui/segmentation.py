import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image, ImageTk
#import zFISHer.config.config_manager as cfgmgr
import cv2
import numpy as np
import platform
from typing import Dict, Any, List, Tuple
import importlib.util

import zFISHer.utils.config as cfg
import zFISHer.data.parameters as aparams
import zFISHer.processing.normalize_img as norm

"""
Takes the desired nucleus image and attempts to run segementation on image. The image is 
"""


class AutoScrollbar(tk.Scrollbar):
    '''
    A scrollbar that hides itself if it's not needed.
        Works only if you use the grid geometry manager 
    '''
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
    

class SegmentationGUI(tk.Frame):

    def __init__(self, master, switch_to_new_gui,logger):

        self.master = master
        self.switch = switch_to_new_gui  # Save the function

        
        self.load_canvas_mip(master)
        self.setup_window(master)

        contours_arr = self.segment_mips()
        self.draw_init_polygons(contours_arr)
        self.master.update()
        self.show_image(master)

        self.set_user_input_controls()

        ###Fix later, variable declaration should not be in init
        self.manpolypoints = []
        return
  

    def load_canvas_mip(self,master):
        pass


    def setup_window(self, master):
        """
        Builds the GUI window and populates with widgets.

        Args:
        master (tkinter.Toplevel) : frame passed by GUImanager to build GUI.
        """

        # Title the window
        self.master.title('ZFISHER --- Nucleus Segmentation')

        ###tk.Frame.__init__(self, master=self.master)

        self.setup_canvas_window(master)
        self.setup_controls_window(master)


    def setup_canvas_window(self,master):
        """
        Creates the tkinter window that will display the MIP image and the nuclei segmentation polygons.
        """

        # Set the initial size of the main frame
        master.geometry("1024x1022")  # Example size: width=800, height=600

        # Vertical and horizontal scrollbars for canvas
        vbar = AutoScrollbar(self.master, orient='vertical')
        hbar = AutoScrollbar(self.master, orient='horizontal')
        vbar.grid(row=0, column=1, rowspan=2, sticky='ns')
        hbar.grid(row=2, column=0, sticky='we')

        # Create canvas 
        self.canvas = tk.Canvas(self.master, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        vbar.configure(command=self.scroll_y)  # bind scrollbars to the canvas
        hbar.configure(command=self.scroll_x)
        # Make the canvas expandable
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        self.load_seg_mips_imgs()


        self.width, self.height = self.f1_seg_c_n.size
        self.baseHeight = self.height
        self.baseWidth = self.width
        self.imscale = 1.0  # scale for the canvaas image
        self.scalefactor = 1.0
        self.delta = 1.3  # zoom magnitude
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)


    def get_seg_mips_paths(self):
        """
        Retrieve the MIP filepaths to be used for segmentation
        """

        # Verify paths before using them
        f1_seg_c_dir = cfg.F1_C_MIP_DIR_DICT[cfg.F1_SEG_C]
        f2_seg_c_dir = cfg.F2_C_MIP_DIR_DICT[cfg.F2_SEG_C]
        f1_seg_c_file = os.listdir(f1_seg_c_dir)[0]
        f2_seg_c_file = os.listdir(f2_seg_c_dir)[0]
        self.f1_seg_c_filepath = os.path.join(f1_seg_c_dir,f1_seg_c_file)
        self.f2_seg_c_filepath = os.path.join(f2_seg_c_dir,f2_seg_c_file)
        return self.f1_seg_c_filepath,self.f2_seg_c_filepath


    def load_seg_mips_imgs(self):
        f1_seg_c_filepath, f2_seg_c_filepath = self.get_seg_mips_paths()
        self.f1_seg_c_n = norm.normalize_mip_main(f1_seg_c_filepath)
        self.f2_seg_c_n = norm.normalize_mip_main(f2_seg_c_filepath)


    def setup_controls_window(self,master):
            #MAKE CONTROL PANEL
            self.control_window = tk.Toplevel(self.master)
            self.control_window.title("Control Panel - Nucleus Segmentation")

            self.list_label = tk.Label(self.control_window, text="-----Number of Nuclei:-----")
            self.list_label.grid(row=1, column=1, columnspan=2)

            self.count_label = tk.Label(self.control_window, text="0")
            self.count_label.grid(row=2, column=1, columnspan=2)

            self.spacer_label = tk.Label(self.control_window, text="------------------------------------")
            self.spacer_label.grid(row=6, column=1, columnspan=2)  

            self.mousepos_label = tk.Label(self.control_window, text="Mouse Position: (0000.00,0000.00)", font=("Courier"))
            self.mousepos_label.grid(row=7, column=1, columnspan=2)
            x_lab = 0
            y_lab = 0
            self.mousepos_label.config(text=f"Mouse Position: ({float(x_lab):07.2f}, {float(y_lab):07.2f})")
            
            self.finish_button = tk.Button(self.control_window, text="Finalize Nuclei Picking", command=self.finalize_nucpicking)
            self.finish_button.grid(row=50, column=1, columnspan=2)     

            self.ManPoly_toggle = tk.BooleanVar(value=False)
            self.polydraw_toggle_checkbox = tk.Checkbutton(self.control_window, text="Manual Polygon", variable=self.ManPoly_toggle)
            self.polydraw_toggle_checkbox.grid(row=10, column=1, columnspan=2) 

            self.seg_algo_header_l = tk.Label(self.control_window, text="Segmentation Algorithm:", font=("Helvetica", 16, "bold"))
            self.seg_algo_header_l.grid(row = 12, column = 1)
            
            self.seg_algo_path_e = tk.Entry(self.control_window, width=50)
            self.seg_algo_path_e.grid(row=12, column=2, sticky="ew", padx=10, pady=5)
             # Handle None or invalid NUC_SEG_ALGO_PATH
            self.set_default_algorithm()
            algo_path = cfg.NUC_SEG_ALGO_PATH if cfg.NUC_SEG_ALGO_PATH is not None else ""
            self.seg_algo_path_e.insert(0, algo_path)  # Set default algorithm path

            self.seg_algo_path_button = tk.Button(self.control_window, text="Browse", command=self.open_seg_file_select)
            self.seg_algo_path_button.grid(row=12, column=3, padx=10, pady=5)

            #self.seg_algo_path_e.insert(0, cfg.NUC_SEG_ALGO_PATH)  #set default algorithm path

            self.remove_all_button = tk.Button(self.control_window, text="Remove All Polygons", command=self.remove_all_polygons)
            self.remove_all_button.grid(row=20, column=1, padx=10, pady=5, columnspan=2)

            self.autosegment_button = tk.Button(self.control_window, text="Autosegment Image", command=self.auto_segment_polygons)
            self.autosegment_button.grid(row=21, column=1, padx=10, pady=5, columnspan=2)

    def set_default_algorithm(self):
        if cfg.NUC_SEG_DEFAULT_SCRIPT is not None:
            algo_path = os.path.join(cfg.SEG_ALGO_DIR, cfg.NUC_SEG_DEFAULT_SCRIPT)
            print(f"BASE DIR {cfg.BASE_DIR}")
            print(f"ALGORITHM PATH {algo_path}")
            if os.path.isfile(algo_path):
                self.seg_algo_path_e.delete(0, 'end')
                self.seg_algo_path_e.insert(0, algo_path)
                cfg.NUC_SEG_ALGO_PATH = algo_path
                print(f"DEFAULT NUC SEG ALGO PATH FOUND: {cfg.NUC_SEG_ALGO_PATH} ")
                return

        # If script is None or file does not exist
        print("NO SEGMENTATION ALGORITHM FOUND")
        self.seg_algo_path_e.delete(0, 'end')
        self.seg_algo_path_e.insert(0, "NO SEGMENTATION ALGORITHM FOUND")
        cfg.NUC_SEG_ALGO_PATH = None

    def open_seg_file_select(self):
        """
        Update file 1 path entry widget with selected path name.
        """
        filepath = filedialog.askopenfilename()
        self.seg_algo_path_e.delete(0, tk.END)
        self.seg_algo_path_e.insert(0, filepath)
        self.set_seg_algo()

    def set_seg_algo(self):
        """
        Set cfg.NUC_SEG_ALGO_PATH to the path in the seg_algo_path_e Entry widget.
        """
        #import os  # Ensure os is available for path validation
        filepath = self.seg_algo_path_e.get().strip()  # Get and clean the entered path
        
        if not filepath:
            #self.logger.warning("No segmentation algorithm path selected")
            print(f"No segmentation algorithm path selected")
            return
        
        if not os.path.exists(filepath):
            #self.logger.error(f"Selected segmentation script does not exist: {filepath}")
            print(f"Selected segmentation script does not exist: {filepath}")
            return
        
        cfg.NUC_SEG_ALGO_PATH = filepath
        #self.logger.info(f"Set NUC_SEG_ALGO_PATH to {filepath}")
        algo_path = cfg.NUC_SEG_ALGO_PATH if cfg.NUC_SEG_ALGO_PATH is not None else ""
        self.seg_algo_path_e.insert(0, algo_path)  # Set default algorithm path

    def show_image(self, event=None):
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
        #self.update_nuc_count()
        # Store bbox1 for later use
        self.bbox1 = bbox1

        x1 = max(bbox2[0] - bbox1[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]

        #Scaled positions for adding polygons from cropchunk add_polygon func
        self.si_x1 = x1
        self.si_y1 = y1
        self.si_x2 = x2
        self.si_y2 = y2

        self.grey_image_offset_x = bbox1[0] - bbox2[0]
        self.grey_image_offset_y = bbox1[1] - bbox2[1]       

        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            x = min(int(x2 / self.imscale), self.width)   # sometimes it is larger on 1 pixel...
            y = min(int(y2 / self.imscale), self.height)  # ...and sometimes not
            image = self.f1_seg_c_n.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            self.imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
                                            anchor='nw', image=imagetk)
            self.bgimage= self.canvas.lower(self.imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection


    def reset_canvas(self):
        """Reset the image to its initial state"""
        # Delete all existing canvas items
        self.canvas.delete("all")
        
        # Reset scroll region to initial state
        self.canvas.configure(scrollregion=(0, 0, self.width, self.height))
        
        # Clear any stored polygons or overlays (assuming these exist from your previous question)
        if hasattr(self, 'polygons'):
            self.polygons.clear()
        
        # Reset scale and position variables to initial values
        self.imscale = 1.0  # Assuming 1.0 is your initial scale
        self.si_x1 = 0
        self.si_y1 = 0
        self.si_x2 = self.width
        self.si_y2 = self.height
        self.grey_image_offset_x = 0
        self.grey_image_offset_y = 0
        
        # Recreate the initial image
        initial_image = self.f1_seg_c_n.crop((0, 0, self.width, self.height))
        imagetk = ImageTk.PhotoImage(initial_image)
        self.imageid = self.canvas.create_image(0, 0, anchor='nw', image=imagetk)
        self.bgimage = self.canvas.lower(self.imageid)
        self.canvas.imagetk = imagetk  # Keep reference to prevent garbage collection
        
        # Store initial bbox
        self.bbox1 = (1, 1, self.width - 1, self.height - 1)

        # Optional: Reset container if it exists
        if hasattr(self, 'container'):
            self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, 
                                                        width=0, fill="")
    #######################################################
    ##############_POLYGON-RELATED_FUNCTIONS_##############
    
    def auto_segment_polygons(self):
        contours = self.segment_mips()
        self.reset_canvas()
        self.draw_init_polygons(contours)
    
    def draw_init_polygons(self,contours_arr):
        """
        When GUI initializes, this runs the onboard nuclei segmentation algorithm
        and draws the polygon outputs.
        """
        self.polygons = []
        prev_column_value = None
        coordinates = []
        cell_index_table = [0, 0]
        for row in contours_arr:
            #print(self.arr_nucleus_contours_coordinates)
            if prev_column_value is None or row[0] == prev_column_value:
                temp_array = row[2:]
                coordinates = np.hstack((coordinates, temp_array))
            else:
                new_row = [row[0], str(coordinates)]
                cell_index_table = np.vstack((cell_index_table, new_row))
                scaled_coordinates = [coord  if i % 2 == 0 else coord  for i, coord in
                                    enumerate(coordinates)]
                self.draw_polygon(scaled_coordinates, prev_column_value)  # Pass index to draw_polygon function
                coordinates = row[2:]
            prev_column_value = row[0]
        
        #----Remove Edge of FOV Nucs from autopick-------
        temp_polygons = []
        print(f"len temp poly {len(temp_polygons)}")
        counter = 0
        for p_row in self.polygons:
            index, polygon, indx_label = p_row

            coords = self.canvas.coords(polygon)


            heightcutoff = self.height * 0.99
            if 0 in coords:
                print("FOUND A ZERO")
                counter += 1

                self.canvas.delete(indx_label)
                self.canvas.delete(polygon)
            elif any(x >= heightcutoff for x in coords):
                print("FOUND A MAX")
                counter += 1

                self.canvas.delete(indx_label)
                self.canvas.delete(polygon)                
            else:
                temp_polygons.append(p_row)

        self.polygons = temp_polygons
        self.update_nuc_count()

    def draw_added_polygons(self,nuc_cntrs):
        print(nuc_cntrs)
        prev_column_value = None
        coordinates = []
        cell_index_table = [0, 0]
        for row in nuc_cntrs:
            if prev_column_value is None or row[0] == prev_column_value:
                temp_array = row[2:]
                coordinates = np.hstack((coordinates, temp_array))
            else:
                new_row = [row[0], str(coordinates)]
                cell_index_table = np.vstack((cell_index_table, new_row))
                scaled_coordinates = [coord  if i % 2 == 0 else coord  for i, coord in
                                    enumerate(coordinates)]
                newindex = len(self.polygons) + 1
                    

            
                self.draw_polygon(scaled_coordinates, newindex)  # Pass index to draw_polygon function
                coordinates = row[2:]
            prev_column_value = row[0]
        scaled_coordinates = [coord if i % 2 == 0 else coord for i, coord in enumerate(coordinates)]
        newindex = len(self.polygons) + 1
        self.draw_polygon(scaled_coordinates, newindex)

        self.add_p_event_processed = False


    def add_polygon(self, event):
        if self.ManPoly_toggle.get() == True: return
        mousex = self.canvas.canvasx(event.x)
        mousey = self.canvas.canvasy(event.y) 

        # need to update this later, hardcoded filename with default segmentation 
        masked_mip_path = os.path.join(cfg.SEG_PROCESSING_DIR, f"nucleus_threshold_img.tif")    

        maskedmip = cv2.imread(masked_mip_path, cv2.IMREAD_GRAYSCALE)
        crop_width, crop_height = 300, 300  # Crop pic size

        scropx1 = max(0, int(mousex - crop_width * self.imscale // 2)) 
        scropy1 = max(0, int(mousey - crop_height * self.imscale // 2))
        scropx2 = min(maskedmip.shape[1], int(mousex + crop_width * self.imscale // 2))
        scropy2 = min(maskedmip.shape[0], int(mousey + crop_height * self.imscale // 2))

        print(f"six1_ {self.si_x1},{self.si_y1}")
        smousex = event.x_root - self.canvas.winfo_rootx()
        smousey = event.y_root - self.canvas.winfo_rooty()



        bbcoords = self.canvas.coords(self.container)

        if  self.imscale == 1.0:
            print("NO ZOOM")
            smousex = smousex*1/self.imscale + (max(bbcoords[0],self.si_x1))
            smousey = smousey*1/self.imscale + (max(self.grey_image_offset_y,self.si_y1))
        elif self.imscale > 1.0:
            print("ZOOMED IN")
            smousex = smousex*1/self.imscale + (max(self.grey_image_offset_x,self.si_x1*1/self.imscale))
            smousey = smousey*1/self.imscale + (max(self.grey_image_offset_y,self.si_y1*1/self.imscale))
        elif self.imscale < 1.0:
            print("ZOOMED OUT")
            smousex = ((smousex - max(0,self.grey_image_offset_x))*1/self.imscale) + self.si_x1*1/self.imscale
            smousey = ((smousey - max(0,self.grey_image_offset_y))*1/self.imscale) + self.si_y1*1/self.imscale


        socropx1 = max(0, int(smousex - (crop_width // 2))) 
        socropy1 = max(0, int(smousey - (crop_height // 2)))
        socropx2 = min(maskedmip.shape[0], int(smousex  + (crop_width // 2)))
        socropy2 = min(maskedmip.shape[1], int(smousey  + (crop_height // 2)))

        print(f"imscale {self.imscale}")
        print(f"grey offset {self.grey_image_offset_x}, {self.grey_image_offset_y}")
        print(f"mousepos {mousex},{mousey}")
        print(f"smousepos {smousex},{smousey}")
        print(f"scrop_ {scropx1},{scropy1},{scropx2},{scropy2}")
        print(f"socrop {socropx1},{socropy1},{socropx2},{socropy2}")
        print(f"Canvas position: x={event.x}, y={event.y}")
        print(f"Screen position: x={event.x_root}, y={event.y_root}")
        print(f"six1_{self.si_x1},{self.si_y1}")
        print(f"bbox {self.canvas.coords(self.container)}")
        canvas_position= [event.x,event.y]
        screen_position= [event.x_root, event.y_root]

       # self.canvas.create_rectangle(scropx1,scropy1,scropx2,scropy2, fill="", outline="white")

        cropped_image = maskedmip[socropy1:socropy2, socropx1:socropx2]

            #Save the cropped image
        print("cropped")
        cropfile_path = os.path.join(cfg.SEG_PROCESSING_DIR,"cropchunk.tif")
        cv2.imwrite(cropfile_path, cropped_image) 



        #Trace the cropped image
        cropfile= cv2.imread(cropfile_path, cv2.IMREAD_GRAYSCALE)
        contours, hierarchy = cv2.findContours(cropfile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #print(contours)
        contoured_cropfile = cropfile.copy()
        cv2.drawContours(contoured_cropfile, contours, -1, (0,255,0), 2, cv2.LINE_AA)

        cont_cropfile_path = os.path.join(cfg.SEG_PROCESSING_DIR,"contour_cropchunk.tif")
        cv2.imwrite(cont_cropfile_path, contoured_cropfile)

        index = 1
        isolated_count = 0
        cluster_count = 0

        arr_nucleus_contours = np.empty([1,4])

        for cntr in contours:
            epsilon = 0.001 * cv2.arcLength(cntr, True)
            approx = cv2.approxPolyDP(cntr, epsilon, True)
            n = approx.ravel() 
            i = 0
            for j in n : 
                if(i % 2 == 0): 
                    x = n[i]  
                    y = n[i + 1] 
                    #print(f"---{x},{y}")
                    #Generate contours cooridinates array to pass into tkinter visualization
                    arr_nucleus_contours = np.vstack((arr_nucleus_contours , [index,i,x,y]))
                i += 1
            index += 1
        arr_nucleus_contours = np.delete(arr_nucleus_contours,(0),axis=0)

        #print(len(arr_nucleus_contours))
        print("_________________________")
        if len(arr_nucleus_contours) <=0: 
            print("no contours found")
            self.add_p_event_processed = False
            return
        else: 

            if  self.imscale == 1.0:
                print("NO ZOOM")
                xo = scropx1
                yo = scropy1
                ofx = 0
                ofy = 0
            elif self.imscale > 1.0:
                print("ZOOMED IN")

                cmousex = self.canvas.canvasx(event.x)
                cmousey = self.canvas.canvasy(event.y)
                xo = int(cmousex - crop_width * self.imscale // 2)
                yo = int(cmousey - crop_height * self.imscale // 2)

                #self.canvas.create_oval(xo-10,yo-10,xo+10,yo+10, fill="", outline="white")

                
                ofx = 0
                ofy = 0

                bbcoords = self.canvas.coords(self.container)
                print (f"cmouse {cmousex},  -->{cmousex - crop_width*self.imscale // 2 }")

                if bbcoords[0]<0 : 
                    if (cmousex - crop_width*self.imscale // 2) < bbcoords[0]:
                        print("lilx")
                        ofx = (bbcoords[0])- (cmousex - crop_width*self.imscale // 2) 
                if bbcoords[1]<0 : 
                    print((cmousey - crop_height*self.imscale // 2))
                    if (cmousey - crop_height*self.imscale // 2) < bbcoords[1]:
                        print("lily")
                        ofy = (bbcoords[1]) - (cmousey - crop_height*self.imscale // 2) 


            elif self.imscale < 1.0:
                print("ZOOMED OUT")
                cmousex = self.canvas.canvasx(event.x)
                cmousey = self.canvas.canvasy(event.y)
                xo = int(cmousex - crop_width * self.imscale // 2)
                yo = int(cmousey - crop_height * self.imscale // 2)

              #  self.canvas.create_oval(xo-10,yo-10,xo+10,yo+10, fill="", outline="white")

                
                ofx = 0
                ofy = 0

                bbcoords = self.canvas.coords(self.container)
                print (f"cmouse {cmousex},  -->{cmousex - crop_width*self.imscale // 2 }")
                if bbcoords[0]>0 : 
                    if cmousex - crop_width*self.imscale // 2 < bbcoords[0]:
                        print("bigginx")
                        ofx = (bbcoords[0]) - (cmousex - crop_width*self.imscale // 2) 
                if bbcoords[1]>0 : 
                    if cmousey - crop_height*self.imscale // 2 < bbcoords[1]:
                        print("bigginy")
                        ofy = (bbcoords[1]) - (cmousey - crop_height*self.imscale // 2) 



                
            for row in arr_nucleus_contours:
                row[2] = row[2]*self.imscale + xo + ofx
                row[3] = row[3]*self.imscale + yo + ofy



            if len(arr_nucleus_contours) <= 0: 
                print("No contours")
                self.add_p_event_processed = False
                return
            else:
                print("ADD POLYGON")
                self.draw_added_polygons(arr_nucleus_contours)


    def remove_polygon(self, event):
        if self.ManPoly_toggle.get() == True: return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y) 
        item = self.canvas.find_closest(x, y)
        if item and item[0] in (polygon[1] for polygon in self.polygons):
            for polygon_index, (index, p, ilabel) in enumerate(self.polygons):
                if p == item[0]:
                    print(f"Deleted polygon {index}")
                    self.canvas.delete(ilabel)
                    self.canvas.delete(p)  
                    print(f"polylenb {len(self.polygons)}")
                    del self.polygons[polygon_index]
                    print(f"polylena {len(self.polygons)}")
                    
                    for row in self.polygons:
                        print(row)
                    break  # Break the loop after finding the polygon to delete
        if item and item[0] in (polygon[2] for polygon in self.polygons):
            for polygon_index, (index, p, ilabel) in enumerate(self.polygons):
                if ilabel == item[0]:
                    print(f"Deleted polygon {index}")
                    self.canvas.delete(ilabel)
                    self.canvas.delete(p)  
                    print(f"polylenb {len(self.polygons)}")
                    del self.polygons[polygon_index]
                    print(f"polylena {len(self.polygons)}")
                    for row in self.polygons:
                        print(row)
                    break  # Break the loop after finding the polygon to delete
        self.update_nuc_count()
   
    def man_poly_point(self,event):
        if self.ManPoly_toggle.get() == False: return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y) 
        #x,y = event.x, event.y
        self.manpolypoints.append((x,y))
        self.draw_mantemppoly()

    def man_poly_point_complete(self, event):
        if self.ManPoly_toggle.get() == False: return
        print("COMPLETE MANUAL POLYGON")
        if len(self.manpolypoints) > 2:
            #self.canvas.create_line(self.manpolypoints, fill='white', width=2, tags='tempmanpolygon')
            self.canvas.delete('tempmanpolygon')
            newindex = len(self.polygons) + 1
            self.draw_polygon(self.manpolypoints, newindex)
            self.manpolypoints = []


    def draw_mantemppoly(self):
        # Redraw polygon based on current points
        self.canvas.delete('tempmanpolygon')
        if len(self.manpolypoints) > 1:
            self.canvas.create_line(self.manpolypoints, fill='green', width=2, tags='tempmanpolygon')
          

    def update_nuc_count(self):
        """
        Updates GUI polygon count label on the control window.
        """
        self.nuccount = len(self.polygons)
        self.count_label.config(text=self.nuccount)
    def display_index_on_polygon(self, polygon, index):
        coords = self.canvas.coords(polygon)
        x_coords = [int(coords[i]) for i in range(0, len(coords), 2)]
        y_coords = [int(coords[i]) for i in range(1, len(coords), 2)]
        x_center = sum(x_coords) // len(x_coords)  # Using integer division
        y_center = sum(y_coords) // len(y_coords)  # Using integer division
        text_item = self.canvas.create_text(x_center, y_center, text=str(index), font=('Arial', 12), fill='red')
        return text_item
    def draw_polygon(self, coordinates, index):
        index = int(index)
        self.polygons = sorted(self.polygons, key=lambda x: x[0])

        scaled_coordinates = [coord * 1 for coord in coordinates]
        #print(scaled_coordinates)
       #print(len(scaled_coordinates))
        polygon = self.canvas.create_polygon(scaled_coordinates, outline='red', fill="", tags=('polygon'))

 
        newpolycoords = self.canvas.coords(polygon)
            
        #REMOVE polygon if OVERLAPPING
        n_num_points = len(newpolycoords) // 2
        n_avg_x = sum(newpolycoords[i] for i in range(0, len(newpolycoords), 2)) / n_num_points
        n_avg_y = sum(newpolycoords[i] for i in range(1, len(newpolycoords), 2)) / n_num_points

        bboxwidth = 20
        bboxheight = 20

        n_bbox = self.canvas.create_rectangle(n_avg_x - bboxwidth/2, n_avg_y - bboxheight/2, n_avg_x + bboxwidth/2, n_avg_y + bboxheight/2)

        overlapping_items = self.canvas.find_overlapping(n_avg_x - bboxwidth/2, n_avg_y - bboxheight/2, n_avg_x + bboxwidth/2, n_avg_y + bboxheight/2)
        for item_id in overlapping_items:
            tag = self.canvas.gettags(item_id)
            if "polygon" in tag:
                if item_id != polygon: 
                    print(f"OLD POLY OVERLAP {item_id}")
                    self.canvas.delete(polygon)
                    self.canvas.delete(n_bbox)
                    return
        self.canvas.delete(n_bbox)


        #Re-index the polygon if there is a gap in the self.polygons ID list
        missing_numbers = []
        expected_number = 1
        for row in self.polygons:
            while row[0] > expected_number:
                missing_numbers.append(expected_number)
                expected_number += 1
            expected_number = row[0] + 1
        if len(missing_numbers) > 0: index = missing_numbers[0]

        print(f"polygon index {index}")
        for rowinfo in self.polygons:   # Storing index, polygon, and text index
            p_index,p_poly,p_tindex = rowinfo
            if index == p_index:
                print("dupe")


        indx_label = self.display_index_on_polygon(polygon, int(index))  # Display index on polygon

        self.polygons.append((index, polygon, indx_label))  # Storing index, polygon, and text index
        self.update_nuc_count()
        
        # Disable mouse events on the polygon
        self.canvas.tag_bind(polygon, '<Button-1>', lambda event: "break")
        self.canvas.tag_bind(polygon, '<ButtonRelease-1>', lambda event: "break")
        self.canvas.tag_bind(polygon, '<B1-Motion>', lambda event: "break")
        self.canvas.tag_bind(polygon, '<Enter>', lambda event, p=polygon: self.highlight_polygon(p))
        self.canvas.tag_bind(polygon, '<Leave>', lambda event, p=polygon: self.unhighlight_polygon(p))
    def highlight_polygon(self, polygon):
        self.canvas.itemconfig(polygon, fill='yellow')
    def unhighlight_polygon(self, polygon):
        self.canvas.itemconfig(polygon, fill='') 
    def remove_all_polygons(self):
        print(len(self.polygons))
        while self.polygons:  # While list is not empty
            (index, p, ilabel) = self.polygons[0]  # Get first element
            self.canvas.delete(ilabel)
            self.canvas.delete(p)
            del self.polygons[0]  # Remove first element
        print(len(self.polygons))
        self.update_nuc_count()
    #######################################################
    ##################_MOVEMENT_CONTROLS_##################
    def set_user_input_controls(self):
        """
        Create key binding-function linking for user-input for different platforms.
        """

        self.canvas.bind("<Motion>", self.motion)
        self.canvas.bind('<Configure>', self.show_image)  # canvas is resized
        
        if platform.system() == "Darwin":  # macOS
            self.canvas.bind("<Button-1>", self.remove_polygon) #click polygon to remove
            self.canvas.bind("<Button-1>", self.man_poly_point, add='+') #add Vertex for manual polygon 
            self.canvas.bind('<Button-2>', self.add_polygon) # right-click for macOS
            self.canvas.bind('<Button-2>', self.man_poly_point_complete, add='+') # close manual polygon
            self.canvas.bind('<ButtonPress-3>', self.move_from)
            self.canvas.bind('<B3-Motion>',     self.move_to)
            self.canvas.bind("<MouseWheel>", self.zoom)
        elif platform.system() == "Linux":  # Linux
            self.canvas.bind("<Button-1>", self.remove_polygon) #click polygon to remove
            self.canvas.bind("<Button-1>", self.man_poly_point, add='+') #add Vertex for manual polygon 
            self.canvas.bind('<Button-3>', self.add_polygon) # right-click for Linux
            self.canvas.bind('<Button-3>', self.man_poly_point_complete, add='+') # close manual polygon
            self.canvas.bind('<ButtonPress-2>', self.move_from)
            self.canvas.bind('<B2-Motion>',     self.move_to)
            self.canvas.bind("<Button-4>", self.zoom)
            self.canvas.bind("<Button-5>", self.zoom)
        elif platform.system() == "Windows": #Windows
            self.canvas.bind("<Button-1>", self.remove_polygon) #left-click to remove polygon
            self.canvas.bind("<Button-1>", self.man_poly_point, add='+') #left-click to add vertex
            self.canvas.bind('<Button-3>', self.add_polygon) #right-click to add polygon
            self.canvas.bind('<Button-3>', self.man_poly_point_complete, add='+') #right-click to complete polygon
            self.canvas.bind('<ButtonPress-2>', self.move_from) #middle-click press to start moving
            self.canvas.bind('<B2-Motion>', self.move_to) #middle-click drag to move
            self.canvas.bind("<MouseWheel>", self.zoom) #mouse wheel to zoom


    def move_from(self, event):
        ''' Remember previous coordinates for scrolling with the mouse '''
        self.canvas.scan_mark(event.x, event.y)
    def move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.show_image()  # redraw the image
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
        self.canvas.scale('all', x, y, scale, scale)  # rescale all canvas objects
        self.show_image()
   
   

    def scroll_y(self, *args, **kwargs):
        self.canvas.yview(*args, **kwargs)  # scroll vertically
        self.show_image()  # redraw the image


    def scroll_x(self, *args, **kwargs):
        ''' Scroll canvas horizontally and redraw the image '''
        self.canvas.xview(*args, **kwargs)  # scroll horizontally
        self.show_image()  # redraw the image


    def motion(self, event):
        """
        Handles motion tracking when the canvas is clicked and dragged.

        Args:
        event () = 
        """
        pass


    def segment_mips(self):
        f1_seg_path = cfg.F1_SEG_C

        input_data = {"file_#": 1, "seg_c": f1_seg_path}
        print(f"Processing file: {input_data['seg_c']}")
        print(f"Type of nuclei_seg_script: {type(cfg.NUC_SEG_ALGO_PATH)}")
        print(f"Value of nuclei_seg_script: {cfg.NUC_SEG_ALGO_PATH}")

        # Check if the segmentation script path is valid
        if cfg.NUC_SEG_ALGO_PATH is None or not os.path.exists(cfg.NUC_SEG_ALGO_PATH):
            print("NO VALID SEGMENTATION SCRIPT FOUND.")
            return []

        print(f"Script file exists: {os.path.exists(cfg.NUC_SEG_ALGO_PATH)}")
        print(f"Current working directory: {os.getcwd()}")

        try:
            result = self._execute_segmentation_script(cfg.NUC_SEG_ALGO_PATH, input_data)
            self.nuclei_contours = result.get("contours", [])
            print(f"Segmentation complete. Found {len(self.nuclei_contours)} contours")
            return self.nuclei_contours
        except Exception as e:
            print(f"Segmentation failed: {e}")
            return []


    def finalize_nucpicking(self):
        self.update_nuc_count
        aparams.seg_nuc_count = self.nuccount
        cfg.SEG_NUC_COUNT = self.nuccount

        #Set Coordinates back to base
        dx = -self.canvas.coords(self.container)[0]
        dy = -self.canvas.coords(self.container)[1]
        self.canvas.move('all', dx,dy)
        self.canvas.scale('all', 0,0, 1/self.imscale, 1/self.imscale)

        #updates polygons
        polygon_ids = self.canvas.find_withtag('polygon')
        self.polygons = []

        nucnum = 1
        for poly_id in polygon_ids:
            coords = self.canvas.coords(poly_id)
            self.polygons.append([nucnum]+ [coords])
            print(coords)
            nucnum += 1

        aparams.nuc_polygons = self.polygons
        cfg.SEG_NUC_POLYGONS = self.polygons

        print("FINISHED NUC PICK")
        print(cfg.SEG_NUC_POLYGONS)

        self.generate_polygon_raster_mask()     #raster mask used to mark each [pixel for punctafinder]
        self.switch()



    def _execute_segmentation_script(self, script_path: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely executes a segmentation script with provided input data.
        
        Args:
            script_path: Path to the segmentation script
            input_data: Dictionary containing input parameters
            
        Returns:
            Dictionary containing processed results
        """
        print(f"Executing script with script_path type: {type(script_path)}")
        print(f"script_path value: {script_path}")
        
        try:
            spec = importlib.util.spec_from_file_location("segmenter", script_path)
            if spec is None:
                raise ImportError(f"Could not load script at {script_path}")
            module = importlib.util.module_from_spec(spec)
            print("Module spec created successfully")
            spec.loader.exec_module(module)
            print("Module loaded successfully")
            result = module.main(input_data)
            print("Script executed successfully")
            return result
        except AttributeError:
            print("Error: Segmentation script must define a 'main' function")
            return input_data
        except Exception as e:
            print(f"Script execution error: {e}")
            return input_data




    def point_in_polygon(self,x, y, poly_coords):
        """Check if point (x, y) is inside polygon defined by poly_coords [x1, y1, x2, y2, ...]"""
        n = len(poly_coords) // 2  # Number of vertices
        inside = False
        x_coords = poly_coords[0::2]  # Every even index
        y_coords = poly_coords[1::2]  # Every odd index
        
        p1x, p1y = x_coords[0], y_coords[0]
        for i in range(n + 1):
            p2x, p2y = x_coords[i % n], y_coords[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def create_mask_from_polygons(self, polygons, width, height):
        # Create an empty mask with zeros (background)
        mask = np.zeros((height, width), dtype=np.uint16)  # Use uint16 for more than 255 nuclei
        
        # Process each polygon
        for poly_data in polygons:
            nucnum = poly_data[0]  # The polygon ID
            coords = poly_data[1]  # List of coordinates [x1, y1, x2, y2, ...]
            
            # Get bounding box
            x_coords = coords[0::2]  # Every even index
            y_coords = coords[1::2]  # Every odd index
            minx, maxx = int(max(0, min(x_coords))), int(min(width, max(x_coords)))
            miny, maxy = int(max(0, min(y_coords))), int(min(height, max(y_coords)))
            
            # Fill the mask
            for y in range(miny, maxy + 1):
                for x in range(minx, maxx + 1):
                    if self.point_in_polygon(x, y, coords):
                        mask[y, x] = nucnum
        
        return mask



    def generate_polygon_raster_mask(self):
        print("DOING RASTER MASK")
        # Example usage
        width = self.baseWidth  # Replace with your canvas width
        height = self.baseHeight  # Replace with your canvas height

        # Assuming self.polygons is your list from earlier
        mask = self.create_mask_from_polygons(self.polygons, width, height)

        # Save as TIFF using PIL
        mask_img = Image.fromarray(mask)  # 16-bit for more nuclei
        #mask_img.save('mask.tif')
        mask_array = np.array(mask_img)
        output_path = os.path.join(cfg.SEG_PROCESSING_DIR, "raster_mask.tif")    
        print(output_path)
        cv2.imwrite(output_path,mask_array)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#LOGIC TO LOAD SEGMENTATION ALGORITHM TO RUN IMAGES THROUGH