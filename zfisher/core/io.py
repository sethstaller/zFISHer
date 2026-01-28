import nd2
import numpy as np
from dataclasses import dataclass

@dataclass
class FISHSession:
    data: np.ndarray      # (C, Z, Y, X)
    voxels: tuple        # (dz, dy, dx) in microns
    channels: list       # ['DAPI', 'FISH1', ...]
    path: str

def load_nd2(path: str) -> FISHSession:
    with nd2.ND2File(path) as f:
        img = f.asarray()
        # Voxel size handling
        v_size = (f.voxel_size().z, f.voxel_size().y, f.voxel_size().x)
        try:
            ch_names = [c.channel.name for c in f.metadata.channels]
        except AttributeError:
            ch_names = [f"Channel_{i}" for i in range(img.shape[0])]
    return FISHSession(data=img, voxels=v_size, channels=ch_names, path=path)