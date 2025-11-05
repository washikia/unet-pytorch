"""
Module containing utility functions for image processing.
"""
import torch
from skimage import io, exposure, util, color
from torchvision import transforms


def im_to_tensor(im_path: str) -> torch.Tensor:
    """ Read a TIF image and return a tensor (converted to grayscale if multi-channel)
    
    Args:
        im_path: Path to load the image.
    
    Returns:
        Tensor: Torch tensor containing the image as grayscale (1 channel).
    """
    im = io.imread(im_path)
    im = util.img_as_ubyte(exposure.rescale_intensity(im))
    
    # Convert to grayscale if image has multiple channels
    # skimage.io.imread typically loads as (H, W) for grayscale or (H, W, C) for RGB
    if len(im.shape) == 3:
        # Multi-channel image - convert to grayscale
        if im.shape[2] == 3:  # RGB
            im = color.rgb2gray(im)
        else:
            # Other multi-channel - take mean across channels
            im = im.mean(axis=2)
    
    # Convert to tensor (ToTensor adds channel dimension and scales to [0,1])
    im = transforms.ToTensor()(im)
    
    # Ensure float32 dtype (not double/float64)
    return im.type(torch.float32)
