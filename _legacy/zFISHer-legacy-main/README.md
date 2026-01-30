# zFISHer

Advanced image processing for multiplexed sequential FISH (Fluorescence In Situ Hybridization).

zFISHer is a Python package designed to streamline and enhance the analysis of multiplexed sequential FISH data through advanced image processing techniques. [Optionally add more: e.g., "It’s ideal for researchers in genomics and molecular biology looking to automate and refine their imaging workflows."]

Note: This package is currently in the process of being refactored and fully commented for official Python package publication.

## Installation
You can install zFISHer by following the steps below. Choose the method that best suits your setup.

Option 1: Run in a Conda Environment
Clone or download the repository to your local machine.

Navigate to the package directory:
cd zFISHer

Create and activate a Conda environment (recommended Python version: 3.8+):
conda create -n zfisher_env python=3.8
conda activate zfisher_env
Install dependencies (if any—update this list as needed):
pip install -r requirements.txt
Note: A requirements.txt file will be added upon package publication.
Option 2: Run as a Local Executable
Ensure Python (3.8+) is installed on your system.
Navigate to the zFISHer folder:
cd zFISHer

Run the package directly:
python -m zFISHer
Dependencies will be formalized in a future release. For now, ensure common image processing libraries like numpy, scipy, and Pillow are installed if errors occur.

## Usage
To start using zFISHer, navigate to the package directory and initialize the program:

cd zFISHer
python -m zFISHer
This will launch the main workflow for processing multiplexed sequential FISH data.

Analysis Steps

### 1- Welcome
zFISHer progrm successfully initializes and displays first GUI.


### 2- Input Selection
Select your two input files (e.g., raw FISH images or data files).
zFISHer natively supports .nd2 files, but other microscopy images can be preconverted into 4D TIFF stacks prior to starting analysis. See the tests directory for an example file.


### 3- Processing
Automated preprocessing of inputs before registration, segmentation, and ROI picking. Builds directories, saves each file-channel slice as a separate .tif in the proper output directory. Generates MIPs for registration and segmentation.

### 4- XY_Registration
zFISHer uses an automated algorithm to align the specified channels in XY space based on their MIPs.

### 5- Z_Registration
zFISHer currently does not support automated 3D registration, and assumes the mid slice of each stack as properly aligned.


### 6- Nuclei Segmentation

Segment/ outline and number the nuclei from the specified channel.

zFISHer has a default watershed algorithm for nuclei segmentation. However, it contains the option to path




Processing Passes

[Describe what happens in the "PASS" step—e.g., "The package aligns and processes the images to extract meaningful signals." Add more details as refactoring progresses.]
[Additional steps like output generation or visualization can be added here as the package develops.]

Example
Here’s a placeholder example (update with real code once available):


# In the zFISHer directory
python -m zFISHer --input1 path/to/fish_image1.tiff --input2 path/to/fish_image2.tiff
Output: [Describe what the user should expect, e.g., "Processed images saved to output/ folder."]

Contributing
This project is under active development. Contributions, bug reports, and feature requests are welcome! Please submit a Pull Request or open an issue on the repository (link to be added upon publication).

License
[Specify your license here—e.g., MIT, GPL, or leave as TBD for now.]