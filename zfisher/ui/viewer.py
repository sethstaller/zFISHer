import napari
from magicgui import magicgui
from pathlib import Path
from zfisher.core.io import load_nd2
from zfisher.core.registration import segment_nuclei_3d
import numpy as np

# Define your paths as constants at the top for easy editing later
DEFAULT_R1 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-19-24Fdecon.nd2")
DEFAULT_R2 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-17-24Adecon.nd2")

# Helper to map metadata names to colors
CHANNEL_COLORS = {
    "DAPI": "blue",
    "FITC": "green",
    "CY3": "yellow",
    "CY5": "red",
    "TXRED": "magenta"
}

@magicgui(
    call_button="Load Data",
    round1_path={"label": "Round 1 (.nd2)", "filter": "*.nd2"},
    round2_path={"label": "Round 2 (.nd2)", "filter": "*.nd2"},
)
def file_selector_widget(
    viewer: "napari.viewer.Viewer",
    round1_path: Path = DEFAULT_R1,
    round2_path: Path = DEFAULT_R2
):
    """Loads ND2 files, fixes swapped Z/Channel axes, and restores Z-scrolling."""
    
    for path, prefix in [(round1_path, "R1"), (round2_path, "R2")]:
        if not path.exists():
            print(f"Error: {path} not found.")
            continue
            
        session = load_nd2(str(path))
        
        # YOUR DATA SHAPE: (71, 3, 2044, 2048) -> (Z, C, Y, X)
        # NAPARI EXPECTS CHANNELS AT INDEX 1 IF WE WANT TO SPLIT THEM
        # We move axis 1 (Channels) to the front so it becomes (C, Z, Y, X)
        data_swapped = np.moveaxis(session.data, 1, 0)
        
        # Now shape is (3, 71, 2044, 2048)
        # Axis 0 = 3 channels
        # Axis 1 = 71 Z-slices
        
        new_layers = viewer.add_image(
            data_swapped,
            name=[f"{prefix} - {ch}" for ch in session.channels],
            channel_axis=0,        # Now correctly sees 3 channels
            scale=session.voxels,   # Matches the (71, 2044, 2048) ZYX stack
            blending="additive"
        )

        # Apply colors
        for layer in new_layers:
            for ch_name, color in CHANNEL_COLORS.items():
                if ch_name.upper() in layer.name.upper():
                    layer.colormap = color
            
            if "DAPI" not in layer.name.upper():
                layer.visible = False

    # Force the Z-slider to appear for the 71 slices
    viewer.dims.axis_labels = ("z", "y", "x")
    viewer.reset_view()
    print(f"Success! Found {data_swapped.shape[0]} channels and {data_swapped.shape[1]} Z-slices.")

@magicgui(call_button="1. Run AI Segmentation")
def dapi_segmentation_widget(viewer: 'napari.viewer.Viewer', layer: 'napari.layers.Image'):
    if layer is None: return
    
    # Show the user we are working
    viewer.status = "AI Segmenting... please wait."
    
    # Call the logic from our other script
    masks, centroids = segment_nuclei_3d(layer.data)
    
    # Add to viewer
    if centroids is not None:
        viewer.add_points(
            centroids, 
            name=f"{layer.name}_centroids", 
            size=5, 
            face_color='orange'
        )
    
def launch_zfisher():
    viewer = napari.Viewer(title="zFISHer - 3D Colocalization", ndisplay=2) # Force 2D slice mode
    viewer.window.add_dock_widget(file_selector_widget, area="right", name="1. File Selection")
    viewer.window.add_dock_widget(dapi_segmentation_widget, area="right", name="2. AI Nuclei Finder")
    
    napari.run()