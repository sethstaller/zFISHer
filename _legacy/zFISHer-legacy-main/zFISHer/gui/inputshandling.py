import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import re
import zFISHer.utils.config as cfg


class InputsHandlingGUI:
    def __init__(self, master, switch_back_to_file_gui, switch_forward):
        self.master = master
        self.switch_back = switch_back_to_file_gui
        self.switch_forward = switch_forward

        self.fixed_channel_entries = {}
        self.moving_channel_entries = {}

        self.setup_window()

    def setup_window(self):
        self.master.title("zFISHer — Input Channel Metadata")

        # Set window size to 80% of screen width and height
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        width = int(screen_width * 0.8)
        height = int(screen_height * 0.8)
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.master.geometry(f"{width}x{height}+{x}+{y}")

        # Top label
        tk.Label(self.master, text="Channel Metadata Entry", font=("Helvetica", 16, "bold")).pack(pady=10)

        # Scrollable canvas
        canvas = tk.Canvas(self.master)
        scrollbar_y = ttk.Scrollbar(self.master, orient="vertical", command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(self.master, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        canvas.pack(side="top", fill="both", expand=True)
        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")

        self.scrollable_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # Main panel with fixed and Moving channel sections side-by-side
        main_panel = tk.Frame(self.scrollable_frame)
        main_panel.pack(padx=10, pady=10, fill="both", expand=True)

        # Left Frame (fixed Channels)
        left_frame = tk.LabelFrame(main_panel, text="Fixed Image Channels")
        left_frame.grid(row=0, column=0, padx=20, sticky="nw")

        # Add filename and nametag label once at the top of left_frame using grid
        filename_fixed = getattr(cfg, "F1_PATH", "UnknownFile")
        nametag_fixed = getattr(cfg, "F1_NTAG", "UnknownTag")
        info_text_fixed = f"File: {filename_fixed}\nNametag: {nametag_fixed}"
        tk.Label(left_frame, text=info_text_fixed, justify="left", fg="gray")\
            .grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(5, 15))

        # Right Frame (Moving Channels)
        right_frame = tk.LabelFrame(main_panel, text="Moving Image Channels")
        right_frame.grid(row=0, column=1, padx=20, sticky="nw")

        # Add filename and nametag label once at the top of right_frame using grid
        filename_moving = getattr(cfg, "F2_PATH", "UnknownFile")
        nametag_moving = getattr(cfg, "F2_NTAG", "UnknownTag")
        info_text_moving = f"File: {filename_moving}\nNametag: {nametag_moving}"
        tk.Label(right_frame, text=info_text_moving, justify="left", fg="gray")\
            .grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(5, 15))

        # Create channel grids starting at row 1 (below info labels)
        self.create_channel_grid(left_frame, cfg.F1_C_LIST, self.fixed_channel_entries, "fixed", start_row=1)
        self.create_channel_grid(right_frame, cfg.F2_C_LIST, self.moving_channel_entries, "moving", start_row=1)

        # Navigation buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(side="bottom", pady=15)

        tk.Button(button_frame, text="Back", command=self.switch_back).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Next", command=self.finalize_details).pack(side=tk.LEFT, padx=10)

    def create_channel_grid(self, parent, channels, storage_dict, channel_type, start_row=0):
        for idx, ch in enumerate(channels):
            row = start_row + (idx // 2)
            col = idx % 2
            self.create_channel_panel(parent, ch, storage_dict, row, col, channel_type)

    def create_channel_panel(self, parent, channel_name, storage_dict, row, col, channel_type):
        channel_frame = tk.LabelFrame(parent, text=f"{channel_name}")
        channel_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nw")

        # Removed filename and nametag label from here (moved to parent frame)

        labels = [
            ("Probe Name:", "entry"),
            ("Probe Type:", [
                "NONE",
                "piFISH", "FISH-BAC", "FISH-Fosmid", "FISH-Direct Label",
                "Immunofluorescence-Endogeneous",
                "Immunofluorescence-Primary Antibody",
                "Immunofluorescence-Secondary Antibody"
            ]),
            ("Probe Fluorophore:", [
                "NONE",
                "Alexa Fluor 488", "Alexa Fluor 546", "Alexa Fluor 555", "Alexa Fluor 568",
                "Alexa Fluor 594", "Alexa Fluor 647", "Cy3", "Cy3B", "Cy5", "Cy7",
                "FITC", "Rhodamine", "DAPI", "Seebright", "Enzo", "Atto 488",
                "Atto 550", "Atto 647N", "TAMRA", "Pacific Blue", "Quasar 570", "Quasar 670"
            ]),
            ("Target Name:", "entry"),
            ("Target Type:", ["NONE", "DNA", "RNA", "Protein"]),
            ("Cell/Organism Type:", "entry"),
            ("Chromosome #:", "entry"),
            ("Start Position:", "entry"),
            ("End Position:", "entry"),
        ]

        entries = {}

        start_row = 0  # Now row=0 available because no label at top of channel panel

        for i, (label_text, input_type) in enumerate(labels):
            r = start_row + i

            # Label
            label = tk.Label(channel_frame, text=label_text)
            label.grid(row=r, column=0, sticky="w", pady=2)

            # Input widget + store
            if isinstance(input_type, list):
                var = tk.StringVar()
                var.set("NONE")  # default to NONE
                dropdown = ttk.OptionMenu(channel_frame, var, "NONE", *input_type)
                dropdown.grid(row=r, column=1, pady=2, sticky="w")
                entries[label_text] = var
            else:
                entry = tk.Entry(channel_frame, width=25)
                entry.grid(row=r, column=1, pady=2, sticky="w")
                entries[label_text] = entry

            # Copy button
            btn = tk.Button(
                channel_frame,
                text="⇄",
                width=2,
                command=lambda lt=label_text, ch=channel_name, ct=channel_type: self.copy_to_all(ct, lt, ch)
            )
            btn.grid(row=r, column=2, padx=5, sticky="w")

        storage_dict[channel_name] = entries

    def copy_to_all(self, channel_type, label_text, source_channel):
        # Determine source entries dictionary
        if channel_type == "fixed":
            source_entries = self.fixed_channel_entries
        else:
            source_entries = self.moving_channel_entries

        source_widget = source_entries[source_channel][label_text]
        val = source_widget.get() if isinstance(source_widget, tk.StringVar) else source_widget.get()

        # Decide scope of copy: to all, or only same-named channels
        if label_text == "Cell/Organism Type:":
            # Copy to ALL rows in both fixed and moving
            for entries_dict in (self.fixed_channel_entries, self.moving_channel_entries):
                for entries in entries_dict.values():
                    widget = entries[label_text]
                    if isinstance(widget, tk.StringVar):
                        widget.set(val)
                    else:
                        widget.delete(0, tk.END)
                        widget.insert(0, val)
        else:
            # Copy only to rows with the same channel name (excluding the source)
            for entries_dict in (self.fixed_channel_entries, self.moving_channel_entries):
                for ch_name, entries in entries_dict.items():
                    if ch_name != source_channel:
                        continue  # Only copy to same-named channels

                    # Skip the exact row we copied from
                    if entries_dict is source_entries and ch_name == source_channel:
                        continue

                    widget = entries[label_text]
                    if isinstance(widget, tk.StringVar):
                        widget.set(val)
                    else:
                        widget.delete(0, tk.END)
                        widget.insert(0, val)



    def validate_entries(self, entries_dict):
        # Return True if all fields valid; else False
        pattern = re.compile(r"\w")  # at least one word char

        for ch_name, entries in entries_dict.items():
            for label, widget in entries.items():
                if isinstance(widget, tk.StringVar):
                    val = widget.get().strip()
                else:
                    val = widget.get().strip()

                if val == "" or val.upper() == "NONE" or not pattern.search(val):
                    return False, f"Invalid or missing value for '{label.strip(':')}' in channel '{ch_name}'."
        return True, ""

    def finalize_details(self):
        valid_fixed, msg_fixed = self.validate_entries(self.fixed_channel_entries)
        valid_moving, msg_moving = self.validate_entries(self.moving_channel_entries)

        if not valid_fixed:
            messagebox.showerror("Validation Error", msg_fixed)
            return

        if not valid_moving:
            messagebox.showerror("Validation Error", msg_moving)
            return

        # Collect metadata for fixed channels
        # Collect metadata for fixed channels
        fixed_data = {}
        for ch_name, entries in self.fixed_channel_entries.items():
            fixed_data[ch_name] = {
                "Parent File": os.path.basename(getattr(cfg, "F1_PATH", "UnknownFile")),
                "Image Role": "fixed"
            }
            for label, widget in entries.items():
                if isinstance(widget, tk.StringVar):
                    val = widget.get().strip()
                else:
                    val = widget.get().strip()
                fixed_data[ch_name][label.strip(":")] = val

        # Collect metadata for moving channels
        moving_data = {}
        for ch_name, entries in self.moving_channel_entries.items():
            moving_data[ch_name] = {
                "Parent File": os.path.basename(getattr(cfg, "F2_PATH", "UnknownFile")),
                "Image Role": "moving"
            }
            for label, widget in entries.items():
                if isinstance(widget, tk.StringVar):
                    val = widget.get().strip()
                else:
                    val = widget.get().strip()
                moving_data[ch_name][label.strip(":")] = val


        # Combine both into a single dictionary
        all_data = {
            "fixed": fixed_data,
            "moving": moving_data
        }

        # Save to cfg for later access elsewhere
        cfg.CH_METADATA = all_data

        # Write JSON files
        fixed_filename = os.path.splitext(os.path.basename(cfg.F1_PATH))[0]
        moving_filename = os.path.splitext(os.path.basename(cfg.F2_PATH))[0]

        fixed_json_path = os.path.join(cfg.OUTPUT_DIR, f"{fixed_filename}_chmetadata.json")
        moving_json_path = os.path.join(cfg.OUTPUT_DIR, f"{moving_filename}_chmetadata.json")

        with open(fixed_json_path, "w") as f:
            json.dump(fixed_data, f, indent=4)

        with open(moving_json_path, "w") as f:
            json.dump(moving_data, f, indent=4)

        print(f"Saved fixed channels metadata to {fixed_json_path}")
        print(f"Saved moving channels metadata to {moving_json_path}")

        self.switch_forward()
