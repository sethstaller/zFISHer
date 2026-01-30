import tkinter as tk
from tkinter import ttk

def apply_theme(root):
    """Apply a consistent theme to the given Tkinter root or Toplevel."""
    style = ttk.Style(root)
    style.theme_use("aqua")
    
    style.configure("TButton", font=("Helvetica", 12), background="#4CAF50", foreground="white", padding=10)
    style.configure("TLabel", font=("Helvetica", 14), foreground="#333")
    style.configure("TFrame", background="#F0F0F0")