import tkinter as tk
from tkinter import filedialog, ttk
import os

import zFISHer.utils.config as cfg
import zFISHer.utils.makedir as mkdir
import zFISHer.processing.process_nd2 as process_nd2


class FileInputGUI:
    def __init__(self, master, switch_to_new_gui):
        self.master = master
        self.switch = switch_to_new_gui
        self.f1_valid = False
        self.f2_valid = False

        self.setup_window()

    def setup_window(self):
        self.master.title("zFISHer — Select Input Files")

        tk.Label(self.master, text="Select Fixed and Moving Image Files", font=("Helvetica", 16, "bold")).pack(pady=10)

        main_frame = tk.Frame(self.master)
        main_frame.pack(padx=10, pady=10)

        self.fixed_frame = tk.LabelFrame(main_frame, text="Fixed Image")
        self.fixed_frame.grid(row=0, column=0, padx=10, sticky="n")

        self.moving_frame = tk.LabelFrame(main_frame, text="Moving Image")
        self.moving_frame.grid(row=0, column=1, padx=10, sticky="n")

        self.build_file_section(
            self.fixed_frame,
            is_fixed=True,
            tag_default="File1",
            open_file_callback=self.open_file_1
        )

        self.build_file_section(
            self.moving_frame,
            is_fixed=False,
            tag_default="File2",
            open_file_callback=self.open_file_2
        )

        self.finalize_btn = tk.Button(self.master, text="Finalize Inputs", command=self.finalize_inputs, state=tk.DISABLED)
        self.finalize_btn.pack(pady=20)

    def build_file_section(self, frame, is_fixed, tag_default, open_file_callback):
        prefix = "f1" if is_fixed else "f2"

        # Path
        path_row = tk.Frame(frame)
        path_row.pack(pady=5)
        tk.Label(path_row, text="PATH:").pack(side=tk.LEFT)
        entry = tk.Entry(path_row, width=35)
        entry.pack(side=tk.LEFT, padx=(5, 0))
        tk.Button(path_row, text="Browse", command=open_file_callback).pack(side=tk.LEFT, padx=(5, 0))
        setattr(self, f"{prefix}_path_e", entry)

        # Nametag
        tag_row = tk.Frame(frame)
        tag_row.pack(pady=(2, 5))
        tk.Label(tag_row, text="NAMETAG:").pack(side=tk.LEFT)
        tag_var = tk.StringVar(value=tag_default)
        tag_entry = tk.Entry(tag_row, textvariable=tag_var, width=30)
        tag_entry.pack(side=tk.LEFT, padx=(5, 0))
        setattr(self, f"{prefix}_tag_var", tag_var)

        # Registration Channel (start empty)
        reg_row = tk.Frame(frame)
        reg_row.pack(pady=(0, 2))
        tk.Label(reg_row, text="XY 2D Registration Channel:").pack(side=tk.LEFT)
        reg_channel_var = tk.StringVar()
        reg_channel = ttk.Combobox(reg_row, values=[], state="readonly", width=25, textvariable=reg_channel_var)
        # No default selection here, so no .current()
        reg_channel.pack(side=tk.LEFT, padx=(5, 0))
        setattr(self, f"{prefix}_reg_channel", reg_channel)

        # Segmentation Channel (start empty)
        seg_row = tk.Frame(frame)
        seg_row.pack(pady=(0, 10))
        tk.Label(seg_row, text="Nuclei Segmentation Channel:").pack(side=tk.LEFT)
        seg_channel_var = tk.StringVar()
        seg_channel = ttk.Combobox(seg_row, values=[], state="readonly", width=25, textvariable=seg_channel_var)
        # No default selection here
        seg_channel.pack(side=tk.LEFT, padx=(5, 0))
        setattr(self, f"{prefix}_seg_channel", seg_channel)

        # Divider
        divider = tk.Frame(frame, height=2, bd=1, relief=tk.SUNKEN)
        divider.pack(fill=tk.X, pady=5)

        # Info Labels
        path_label = tk.Label(frame, text="PATH: —")
        path_label.pack()
        name_label = tk.Label(frame, text="NAME: —")
        name_label.pack()
        type_label = tk.Label(frame, text="TYPE: —")
        type_label.pack()
        tag_label = tk.Label(frame, text=f"NAMETAG: {tag_default}")
        tag_label.pack()
        reg_label = tk.Label(frame, text="REGISTRATION CHANNEL: —")
        reg_label.pack()
        seg_label = tk.Label(frame, text="SEGMENTATION CHANNEL: —")
        seg_label.pack()

        # Traces to update labels on changes
        tag_var.trace_add("write", lambda *args: tag_label.config(text=f"NAMETAG: {tag_var.get() or '—'}"))
        reg_channel_var.trace_add("write", lambda *args: reg_label.config(text=f"REGISTRATION CHANNEL: {reg_channel_var.get() or '—'}"))
        seg_channel_var.trace_add("write", lambda *args: seg_label.config(text=f"SEGMENTATION CHANNEL: {seg_channel_var.get() or '—'}"))

        # Store references for later
        setattr(self, f"{prefix}_path_label", path_label)
        setattr(self, f"{prefix}_name_label", name_label)
        setattr(self, f"{prefix}_type_label", type_label)
        setattr(self, f"{prefix}_tag_label", tag_label)
        setattr(self, f"{prefix}_reg_label", reg_label)
        setattr(self, f"{prefix}_seg_label", seg_label)


    def open_file_1(self):
        preferred_dir = "/research_jude/rgs01_jude/shres/CYTOG/common/cell-microinjection"
        alt_dir = "/Volumes/cytogenetics/common/cell-microinjection"

        if os.path.exists(preferred_dir):
            initial_dir = preferred_dir
        elif os.path.exists(alt_dir):
            initial_dir = alt_dir
        else:
            initial_dir = os.getcwd()

        filepath = filedialog.askopenfilename(initialdir=initial_dir)
        if filepath:
            self.f1_path_e.delete(0, tk.END)
            self.f1_path_e.insert(0, filepath)
            self.set_file_labels(filepath, file_num=1)

    def open_file_2(self):
        preferred_dir = "/research_jude/rgs01_jude/shres/CYTOG/common/cell-microinjection"
        alt_dir = "/Volumes/cytogenetics/common/cell-microinjection"

        if os.path.exists(preferred_dir):
            initial_dir = preferred_dir
        elif os.path.exists(alt_dir):
            initial_dir = alt_dir
        else:
            initial_dir = os.getcwd()

        print(initial_dir)
        filepath = filedialog.askopenfilename(initialdir=initial_dir)
        if filepath:
            self.f2_path_e.delete(0, tk.END)
            self.f2_path_e.insert(0, filepath)
            self.set_file_labels(filepath, file_num=2)

    def set_file_labels(self, filepath, file_num):
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filepath)[1].lower().strip(".")

        is_valid = ext in ["nd2", "tif", "tiff"]
        file_type = f"{ext.upper()} File {'✔' if is_valid else '✘'}" if ext else "Unknown file ✘"

        if file_num == 1:
            self.f1_valid = is_valid
            self.f1_path_label.config(text=f"PATH: {filepath}")
            self.f1_name_label.config(text=f"NAME: {filename}")
            self.f1_type_label.config(text=f"TYPE: {file_type}")

            if ext == "nd2":
                self.f1_metadata = self.process_nd2_metadata(filepath)
                if self.f1_metadata:
                    channels = self.f1_metadata['c_list']
                    self.f1_reg_channel['values'] = channels
                    self.f1_seg_channel['values'] = channels

                    # Default reg channel to DAPI if present else first channel
                    if "DAPI" in channels:
                        self.f1_reg_channel.set("DAPI")
                    elif channels:
                        self.f1_reg_channel.current(0)
                    else:
                        self.f1_reg_channel.set('')

                    # Default seg channel to DAPI if present else second channel else first
                    if "DAPI" in channels:
                        self.f1_seg_channel.set("DAPI")
                    elif len(channels) > 1:
                        self.f1_seg_channel.current(1)
                    elif channels:
                        self.f1_seg_channel.current(0)
                    else:
                        self.f1_seg_channel.set('')
            else:
                self.f1_metadata = None

        else:
            self.f2_valid = is_valid
            self.f2_path_label.config(text=f"PATH: {filepath}")
            self.f2_name_label.config(text=f"NAME: {filename}")
            self.f2_type_label.config(text=f"TYPE: {file_type}")

            if ext == "nd2":
                self.f2_metadata = self.process_nd2_metadata(filepath)
                if self.f2_metadata:
                    channels = self.f2_metadata['c_list']
                    self.f2_reg_channel['values'] = channels
                    self.f2_seg_channel['values'] = channels

                    # Default reg channel to DAPI if present else first channel
                    if "DAPI" in channels:
                        self.f2_reg_channel.set("DAPI")
                    elif channels:
                        self.f2_reg_channel.current(0)
                    else:
                        self.f2_reg_channel.set('')

                    # Default seg channel to DAPI if present else second channel else first
                    if "DAPI" in channels:
                        self.f2_seg_channel.set("DAPI")
                    elif len(channels) > 1:
                        self.f2_seg_channel.current(1)
                    elif channels:
                        self.f2_seg_channel.current(0)
                    else:
                        self.f2_seg_channel.set('')
            else:
                self.f2_metadata = None

        self.update_finalize_state()


    def process_nd2_metadata(self, filepath):
        try:
            metadata = process_nd2.nd2_metadata_processor(filepath)
            print(metadata)
            return metadata
        except Exception as e:
            print(f"Error processing ND2 metadata: {e}")
            return None

    def update_finalize_state(self):
        if self.f1_valid and self.f2_valid:
            self.finalize_btn.config(state=tk.NORMAL)
        else:
            self.finalize_btn.config(state=tk.DISABLED)

    def finalize_inputs(self):

        # Establish file config values for directory generation
        cfg.F1_PATH = self.f1_path_e.get()
        cfg.F2_PATH = self.f2_path_e.get()
        cfg.F1_NTAG = self.f1_tag_var.get()
        cfg.F2_NTAG = self.f2_tag_var.get()
        cfg.F1_REG_C = self.f1_reg_channel.get()
        cfg.F2_REG_C = self.f2_reg_channel.get()
        cfg.F1_SEG_C = self.f1_seg_channel.get()
        cfg.F2_SEG_C = self.f2_seg_channel.get()
        cfg.F1_C_LIST = self.f1_metadata['c_list']
        cfg.F2_C_LIST = self.f2_metadata['c_list']
        cfg.F1_C_NUM = self.f1_metadata['c_num']
        cfg.F2_C_NUM = self.f2_metadata['c_num']
        cfg.F1_Z_NUM = self.f1_metadata['z_num']
        cfg.F2_Z_NUM = self.f2_metadata['z_num']
        
        # Make output directories
        mkdir.create_processing_directories()

        # Generate nd2 metadata JSONs only if file is .nd2
        if cfg.F1_PATH.lower().endswith('.nd2'):
            process_nd2.extract_nd2_metadata_nd2lib(cfg.F1_PATH, cfg.OUTPUT_DIR)
        if cfg.F2_PATH.lower().endswith('.nd2'):
            process_nd2.extract_nd2_metadata_nd2lib(cfg.F2_PATH, cfg.OUTPUT_DIR)

        self.switch()
