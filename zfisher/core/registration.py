import numpy as np
from cellpose import models, core
from skimage.measure import regionprops

def segment_nuclei_3d(image_data, gpu=True):
    """
    Runs Cellpose 3D segmentation on a single channel.
    """
    # 1. Properly check for GPU (especially important on Mac/Miniforge)
    use_gpu = core.use_gpu() if gpu else False
    print(f"GPU check: {use_gpu}")

    # 2. Use the 'CellposeModel' class, which is more universal than the 'Cellpose' alias
    # This specifically bypasses the naming conflict we keep hitting
    model = models.CellposeModel(gpu=use_gpu, model_type='nuclei')

    print("Cellpose is starting 3D segmentation... (71 slices)")
    
    # 3. Use eval()
    # For CellposeModel, we don't need the channels list if it's already a single-channel image
    masks, flows, styles = model.eval(
        image_data[0:10], 
        do_3D=True, 
        z_axis=0,
        diameter=100,      
        channels=[0, 0] # Standard for single-channel DAPI
    )
    # 4. Extract anchors
    props = regionprops(masks)
    centroids = np.array([p.centroid for p in props])
    
    print(f"Found {len(centroids)} nuclei.")
    return masks, centroids

def calculate_warp_field(fixed_points, moving_points):
    """
    This is where we'll eventually put the math to map R2 to R1.
    """
    pass