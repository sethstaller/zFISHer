import tkinter as tk
from zFISHer.gui.tktheme import apply_theme
from zFISHer.gui.welcome import WelcomeWindowGUI
from zFISHer.gui.inputs import FileInputGUI
from zFISHer.gui.inputshandling import InputsHandlingGUI
from zFISHer.gui.process import ProcessingGUI
from zFISHer.gui.registrationXY import RegistrationXYGUI
from zFISHer.gui.registrationZ import RegistrationZGUI
from zFISHer.gui.segmentation import SegmentationGUI
from zFISHer.gui.punctapick import PunctaPickGUI
from zFISHer.gui.analysis import AnalysisGUI
from zFISHer.gui.calculations import CalculationsGUI
from zFISHer.gui.finish import FinishGUI

from zFISHer.gui.logger import Logger
from zFISHer import version


class GuiManager:
    """
    Manager of swapping and initializing each GUI of the workflow.
    """


    def __init__(self):
        """
        Initialize the manager class and the root tk instance.
        """
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the root window
        self.current_window = None 
        self.start_logger()


    def to_welcome_gui(self):
        """Switch to the Welcome GUI."""
        self.switch_to_gui(WelcomeWindowGUI, self.to_inputs_gui)
        #self.switch_to_gui(RegistrationZGUI, self.to_segmentation_gui, self.logger)


    def to_inputs_gui(self):
        """
        Switch to the File Input GUI.
        """
        self.switch_to_gui(FileInputGUI, self.to_inputs_handling_gui)


    def to_inputs_handling_gui(self):
        """
        Switch to the Inputs Handling GUI.
        """
        self.switch_to_gui(InputsHandlingGUI,self.to_inputs_gui,self.to_processing_gui)


    def to_processing_gui(self):
        """
        Switch to the File Inputs Processing GUI.
        """
        self.switch_to_gui(ProcessingGUI, self.to_registration_xy_gui,self.logger)
    

    def to_registration_xy_gui(self):
        """
        Switch to the XY registration GUI.
        """
        self.switch_to_gui(RegistrationXYGUI, self.to_segmentation_gui, self.logger)


    def to_registration_z_gui(self):
        """
        Switch to the Z registration GUI.
        """
        self.switch_to_gui(RegistrationZGUI, self.to_segmentation_gui, self.logger)


    def to_segmentation_gui(self):
        """
        Switch to the nuclei segmentation GUI.
        """
        self.switch_to_gui(SegmentationGUI, self.to_punctapick_gui, self.logger)


    def to_punctapick_gui(self):
        """
        Switch to the ROI picking GUI.
        """
        self.switch_to_gui(PunctaPickGUI, self.to_analysis_gui, self.logger)


    def to_analysis_gui(self):
        """
        Switch to the analysis starts and parameters pre-calculations GUI.
        """
        self.switch_to_gui(AnalysisGUI, self.to_calculations_gui, self.logger)


    def to_calculations_gui(self):
        """
        Switch to the calculations progress window.
        """
        self.switch_to_gui(CalculationsGUI, self.to_finish_gui, self.logger)


    def to_finish_gui(self):
        """
        Switch to the finished analysis window.
        """
        self.switch_to_gui(FinishGUI, self.to_welcome_gui, self.logger)


    def switch_to_gui(self, gui_class, *args):
        """
        Switch to a new Toplevel GUI window.
        :param gui_class: The class of the new GUI.
        :param args: Additional arguments to pass to the GUI class.
        """
        # Destroy the current window if it exists
        if self.current_window is not None:
            self.current_window.master.destroy()

        # Create a new Toplevel window
        new_window = tk.Toplevel(self.root)
        self.current_window = gui_class(new_window, *args)  # Pass the new Toplevel to the GUI class

        # Apply the theme to the new Toplevel window
        #apply_theme(new_window)


    def start_logger(self):
        """
        Initialize the logger singleton.
        """
        if not hasattr(self, 'logger'):
            self.logger = Logger()
            self.logger.log_message("Application started successfully.")
            self.logger.log_message(f"Welcome to zFISHer v{version.__version__}.")
        else:
            self.logger.log_message("Logger already initialized.")


    def start_gui(self):
        """
        Start the Tkinter main event loop.
        """
        self.root.mainloop()
