import numpy as np
import logging
from cellpose import models, core
from skimage.measure import regionprops
from skimage.transform import rescale

# Set logging to see the progress bar in terminal
logging.basicConfig(level=logging.INFO)

def segment_nuclei_3d(image_data, gpu=True):
    # 1. SETUP
    use_gpu = core.use_gpu() if gpu else False
    model = models.CellposeModel(gpu=use_gpu, model_type='nuclei')

    # 2. SUBSAMPLE Z
    z_step = 5
    subsampled_data = image_data[::z_step, :, :]
    
    # 3. DOWNSAMPLE X/Y
    scale_factor = 0.25
    small_data = rescale(
        subsampled_data, 
        (1, scale_factor, scale_factor), 
        preserve_range=True, 
        anti_aliasing=True
    ).astype(np.float32) # Cellpose prefers float32

    # 4. EVALUATE (Fixing the ValueError)
    masks_small, flows, styles = model.eval(
        small_data,
        channels=[0,0],      # Grayscale DAPI
        diameter=None,       # We already handled scaling manually
        rescale=1.0,         # <--- CRITICAL: Prevents internal resizing
        do_3D=False,                
        stitch_threshold=0.5,       
        z_axis=0,
        batch_size=16,               
        progress=True,
        resample=False       # <--- CRITICAL: Prevents internal resampling
    )

    # 5. SCALE CENTROIDS
    props = regionprops(masks_small)
    centroids = np.array([
        [p.centroid[0] * z_step, 
         p.centroid[1] / scale_factor, 
         p.centroid[2] / scale_factor] 
        for p in props
    ])
    
    return None, centroids