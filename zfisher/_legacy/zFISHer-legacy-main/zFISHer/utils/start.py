from zFISHer.gui.guimanager import GuiManager
from zFISHer.gui.welcome import WelcomeWindowGUI
from zFISHer.gui.logger import Logger
import zFISHer.utils.config as cfg
import zFISHer.utils.makedir as mkdir

"""
Run all logic and intialize objects and classes to start the processing pipeline.
"""


def start_app() -> None:
    """
    Entry point for initializing the GUI application.
    This function sets up the GuiManager and Logger starts the application.
    Start up and initialize the application components and pass control to the GUI manager for GUI swapping.
    """

    mkdir.set_core_dir_values()
    #mkdir.make_output_directories
    manager = GuiManager()
    manager.to_welcome_gui()
    manager.start_gui()

    



