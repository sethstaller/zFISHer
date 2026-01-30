import tkinter as tk
from tkinter import font
from zFISHer import version

ASCII_ART = r"""
      _____ ___ ____  _   _           
  ___|  ___|_ _/ ___|| | | | ___ _ __ 
 |_  / |_   | |\___ \| |_| |/ _ \ '__|
  / /|  _|  | | ___) |  _  |  __/ |   
 /___|_|   |___|____/|_| |_|\___|_|   
                                      
"""

class WelcomeWindowGUI():
    def __init__(self, master, switch_to_gui_two):
        self.master = master
        self.master.title("ZFISHER --- Welcome")
        self.master.config(bg="#121212")

        screen_width, screen_height = self.get_screen_size()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        x_offset = (screen_width // 2) - (window_width // 2)
        y_offset = (screen_height // 2) - (window_height // 2)
        self.master.geometry(f"{window_width}x{window_height}+{x_offset}+{y_offset}")

        for i in range(6):
            self.master.grid_rowconfigure(i, weight=1)
        for i in range(3):
            self.master.grid_columnconfigure(i, weight=1)

        # Fonts
        self.title_font = font.Font(family="Helvetica", size=36, weight="bold")
        self.ascii_font = font.Font(family="Courier", size=14)
        self.subtitle_font = font.Font(family="Helvetica", size=18, slant="italic")
        self.button_font = font.Font(family="Helvetica", size=24, weight="bold")

        # Title
        self.titlelabel = tk.Label(master, text=f"Welcome to ZFISHER v{version.__version__}!", 
                                   bg="#121212", fg="#00FF00", font=self.title_font)
        self.titlelabel.grid(row=0, column=1, sticky="nsew", pady=(10,5))

        # ASCII Art
        self.ascii_label = tk.Label(master, text=ASCII_ART, bg="#121212", fg="#33FF33", font=self.ascii_font, justify="center")
        self.ascii_label.grid(row=1, column=1, sticky="nsew")

        # Subtitle
        self.subtitle = tk.Label(master, text="Advanced RNA Imaging Analysis Made Easy",
                                 bg="#121212", fg="#66FF66", font=self.subtitle_font)
        self.subtitle.grid(row=2, column=1, sticky="nsew")

        # Button
        self.imageprocessbutton = tk.Button(master, text="Begin Analysis",
                                            command=lambda: self.tofileselect(switch_to_gui_two),
                                            bg="#004400", fg="#AAFFAA", font=self.button_font,
                                            activebackground="#00FF00", activeforeground="#003300",
                                            relief=tk.RAISED, bd=4, cursor="hand2")
        self.imageprocessbutton.grid(row=4, column=1, sticky="nsew", padx=80, pady=40)

        self.imageprocessbutton.bind("<Enter>", self.on_button_hover)
        self.imageprocessbutton.bind("<Leave>", self.on_button_leave)

        master.protocol("WM_DELETE_WINDOW", self.disable_close)

    def on_button_hover(self, event):
        event.widget.config(bg="#00FF00", fg="#003300")

    def on_button_leave(self, event):
        event.widget.config(bg="#004400", fg="#AAFFAA")

    def tofileselect(self, switch) -> None:
        switch()

    def disable_close(self) -> None:
        print("Window close disabled.")

    def get_screen_size(self) -> None:
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        return screen_width, screen_height
