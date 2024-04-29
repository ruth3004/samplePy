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

def bin_image_2x2(image):
    """
    Reduces the size of an image by 2x2 binning, averaging groups of four pixels.
    """
    new_shape = (image.shape[0]//2, image.shape[1]//2)
    return image.reshape(new_shape[0], 2, new_shape[1], 2).mean(axis=(1, 3))

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


def load_planes_from_tif(tif_path, n_planes, doubling=True, ignore_frames=0):
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


    doubling = 2 if doubling else 1

    hyperstack_img = load_tiff_as_hyperstack(tif_path,1,n_planes,375)
    print(hyperstack_img.shape)

    planes_stack = np.zeros((n_planes * doubling, hyperstack_img.shape[1], hyperstack_img.shape[2] // doubling, hyperstack_img.shape[3]))
    print(planes_stack.shape)
    # Split and assign the planes based on the doubling factor
    for plane in range(n_planes):
        planes_stack[2 * plane, :,:,:] = hyperstack_img[plane,:, :hyperstack_img.shape[2] // doubling, :]
        planes_stack[2 * plane + 1,:, :, :] = hyperstack_img[plane,:, hyperstack_img.shape[2] // doubling:, :]

    return planes_stack


def load_images_convert_to_stack(path, ending="*.tif"):
    """
    Loads all images from a directory into a single stack.
    """
    files = sorted(Path(path).glob(ending), key=lambda x: int(x.stem.split('_')[-1]))
    return np.array([np.array(Image.open(file)) for file in files])



def load_tiff_as_hyperstack(file_path, n_channels, n_slices, n_time=None):
    """
    Load a TIFF stack and reshape it into a hyperstack.

    Parameters:
        file_path (str): Path to the TIFF file.
        n_channels (int): Number of channels.
        n_slices (int): Number of z-slices.
        n_time (int): Number of time points.

    Returns:
        numpy.ndarray: A hyperstack array with dimensions [channel, slice, time, y, x].
    """
    with tifffile.TiffFile(file_path) as tif:
        images = tif.asarray()

    # Reshape and reorder to (channels, slices, time, y, x)
    reshaped_images = images.reshape(n_time, n_channels, n_slices, images.shape[1], images.shape[2])
    hyperstack = np.moveaxis(reshaped_images, 0, 2)  # Move time to the third axis

    return np.squeeze(hyperstack)

def load_raw_acquisition(file_path, n_channels, n_slices, n_time):
    """
    Load raw acquisition data from a TIFF file formatted as a hyperstack.

    Parameters:
        file_path (str): Path to the raw acquisition TIFF file.
        n_channels (int): Number of channels in the TIFF file.
        n_slices (int): Number of z-slices per time point.
        n_time (int): Number of time points.

    Returns:
        numpy.ndarray: A hyperstack array [channel, slice, time, y, x].
    """
    return load_tiff_as_hyperstack(file_path, n_channels, n_slices, n_time)

def load_anatomy_stack(file_path, n_channels, n_slices):
    """
    Load anatomy data from a TIFF file formatted as a hyperstack, assuming one time point.

    Parameters:
        file_path (str): Path to the anatomy TIFF file.
        n_channels (int): Number of channels in the TIFF file.
        n_slices (int): Number of z-slices.

    Returns:
        numpy.ndarray: A hyperstack array [channel, slice, y, x].
    """
    return load_tiff_as_hyperstack(file_path, n_channels, n_slices, 1)
