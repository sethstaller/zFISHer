import tkinter as tk
import os
import logging
import zFISHer.utils.config as cfg
#import zFISHer.config.config_manager as cfgmgr

from datetime import datetime
from zFISHer import version

log_file = "log_output.txt"

class Logger:
    _instance = None  # Class-level attribute to store the singleton instance

    def __new__(cls, master=None):
        """
        Singleton pattern to prevent duplicate Logger instances.
        """
        # Check if an instance already exists
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            # Only initialize the logger if it's the first time the instance is created
            #cls._instance.__init__(master)
        return cls._instance
    
    def __init__(self, master=None):
        # Create a new top-level window for the logger
        self.logger_window = tk.Toplevel(master)
        self.logger_window.title("zFISHer - Log")
        self.logger_window.geometry("500x300")

        # Override the close button behavior
        self.logger_window.protocol("WM_DELETE_WINDOW", self.disable_close)

        # Create a Text widget to display log messages
        self.log_text = tk.Text(self.logger_window, wrap=tk.WORD, height=15, width=60)
        self.log_text.pack(padx=10, pady=10)

        # Disable text widget editing so it's read-only
        self.log_text.config(state=tk.DISABLED)

        # Configure log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #self.log_file = os.path.join(cfg.get_config_value("LOGS_DIR"), f"log_{timestamp}.txt")
        self.log_file = os.path.join(cfg.LOGS_DIR, f"log_{timestamp}.txt")
     #  logging.basicConfig(
     #      filename=self.log_file,
     #      filemode="a",  # Append to file
     #       format="%(asctime)s - %(levelname)s - %(message)s",
     #      datefmt="%Y-%m-%d %H:%M:%S",
      #      level=logging.INFO,
      #  )
        print(f"Logging to: {self.log_file}")

    def log_message(self, message, level="info"):
        """Logs a message to the log file and the log window."""
        if level == "info":
            logging.info(message)
        elif level == "warning":
            logging.warning(message)
        elif level == "error":
            logging.error(message)
        else:
            logging.debug(message)
        
        self.log_message_to_window(message, level)

    def log_message_to_window(self, message, level="info"):
        """Add a message to the log text widget."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state=tk.DISABLED)
        self.log_text.yview(tk.END)

    def disable_close(self):
        """Disable the close button action."""
        print("Close button disabled. Logger window cannot be closed.")