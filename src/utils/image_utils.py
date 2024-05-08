import numpy as np
from PIL import Image
import tifffile
import glob
import os
from pathlib import Path
import skimage.io as skio

def extend_stack(stack, margin):
    """
    Extends the boundaries of each slice in the stack by a specified margin.
    """
    return np.pad(stack, ((0, 0), (margin, margin), (margin, margin)), mode='reflect')


def bin_image(image, factor=2):
    # Get the shape of the original image
    height, width = image.shape

    # Calculate the new shape after binning
    new_height = height // factor
    new_width = width // factor

    # Reshape the image into non-overlapping 2x2 blocks
    reshaped_image = image[:new_height * factor, :new_width * factor].reshape(new_height, factor, new_width, factor)

    # Compute the mean along the last two axes to bin the image
    binned_image = np.mean(reshaped_image, axis=(1, 3))

    return binned_image

def load_anatomy_stack(anatomy_stack_path, n_channels=2, channel_num=None):
    #TODO: use load_tiff from hyperstack extension
    """
    Load a specific channel from a multi-channel anatomy stack.

    Parameters:
    - anatomy_stack_path (str): Path to the TIFF file containing the anatomy stack.
    - n_channels (int): Total number of channels in the stack.
    - channel_num (int, optional): Specific channel number to load. If None, all channels are returned.

    Returns:
    - numpy.ndarray: 3D array with the selected channel(s) of the stack.
    """
    with tifffile.TiffFile(anatomy_stack_path) as tif:
        # Read the whole stack
        images = tif.asarray()
        if images.ndim == 3:  # Assuming the shape is (frames, height, width)
            if channel_num is not None:
                # Select every nth frame where n is the number of channels, starting at channel_num
                return images[channel_num::n_channels]
            else:
                # Reshape the stack to separate channels only if channel_num is None
                return images.reshape(-1, n_channels, images.shape[1], images.shape[2]).swapaxes(0, 1)
        elif images.ndim == 4:  # Assuming the shape is (z, channel, height, width)
            if channel_num is not None:
                # Return only the specified channel across all z planes
                return images[:, channel_num, :, :]
            else:
                return images

def load_planes_from_folder(anatomy_dir, n_planes, doubling=True):
    """
    Loads image planes from a directory, considering doubling if specified.
    """
    planes = []
    anatomy_dir = Path(anatomy_dir)
    for i in range(n_planes):
        path = anatomy_dir / f"plane{i + 1:02d}.tif"
        with Image.open(path) as img:
            plane = np.array(img)
            if doubling:
                plane = np.vstack([plane[:plane.shape[0] // 2], plane[plane.shape[0] // 2:]])
            planes.append(plane)
    return np.stack(planes)


def split_double_planes(hyperstack_img):
    """
    Load planes from a (hyper)stack TIF file and reshape based on input parameters.

    Parameters:
    - tif_path (str): Path to the TIF file.
    - n_planes (int): Total number of planes.
    - doubling (int): Doubling factor.
    - ignore_frames (int, optional): Number of frames to be ignored at the start.

    Returns:
    - numpy.ndarray: 3D array containing the reshaped planes in (slice,width,height) or(z,y,x) shape.
    """
    doubling = 2
    n_planes = hyperstack_img.shape[0]
    planes_stack = np.zeros((n_planes * doubling , hyperstack_img.shape[1], hyperstack_img.shape[2] // doubling, hyperstack_img.shape[3]))
    # Split and assign the planes based on the doubling factor
    for plane in range(n_planes):
        planes_stack[2 * plane, :,:,:] = hyperstack_img[plane,:, :hyperstack_img.shape[2] // doubling, :]
        planes_stack[2 * plane + 1,:, :, :] = hyperstack_img[plane,:, hyperstack_img.shape[2] // doubling:, :]

    return planes_stack


def load_images_to_stack(path, ending="*.tif"):
    """
    Loads all images from a directory into a single stack.
    """
    files = sorted(Path(path).glob(ending), key=lambda x: int(x.stem.split('_')[-1]))
    return np.array([np.array(Image.open(file)) for file in files])



def load_tiff_as_hyperstack(file_path, n_slices=1, n_channels=1, doubling=False):
    """
    Load a TIFF stack and reshape it into a hyperstack.

    Parameters:
        file_path (str): Path to the TIFF file.
        n_channels (int): Number of channels.
        n_slices (int): Number of z-slices.

    Returns:
        numpy.ndarray: A hyperstack array with dimensions [channel, slice, time, y, x].
    """
    # read tiff file
    images = tifffile.imread(file_path)
    n_frames = images.shape[0]

    # Reshape and reorder to (channels, slices, time, y, x)
    reshaped_images = images.reshape(n_channels,n_slices,n_frames//n_channels//n_slices,images.shape[1], images.shape[2],order="F")

    hyperstack = np.squeeze(reshaped_images)
    # Splitting doubled planes if set
    if doubling:
        return split_double_planes(hyperstack)
    else:
        return hyperstack

