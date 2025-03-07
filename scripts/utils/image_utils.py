import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

import tifffile
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.interpolate import griddata
from scipy.signal import medfilt

import random

from skimage import feature
from skimage.measure import manders_coloc_coeff, regionprops, label
from skimage.filters import threshold_otsu
from skimage.transform import SimilarityTransform
from skimage.exposure import match_histograms, equalize_adapthist, rescale_intensity
plt.set_cmap('binary')

def extend_stack(stack, margin):
    """
    Extends the boundaries of each slice in the stack by a specified margin.

    Parameters:
    - stack: 3D numpy array representing the image stack
    - margin: Integer specifying the size of the margin to add

    Returns:
    - Extended stack with added margins
    """
    if not isinstance(stack, np.ndarray) or stack.ndim != 3:
        raise ValueError("Stack must be a 3D numpy array")

    if not isinstance(margin, int) or margin < 0:
        raise ValueError("Margin must be a non-negative integer")

    return np.pad(stack, ((0, 0), (margin, margin), (margin, margin)), mode='minimum')


def bin_image(image, factor=2):
    """
    Bins an image by a specified factor.

    Parameters:
    - image: 2D numpy array representing the image
    - factor: Integer specifying the binning factor (default: 2)

    Returns:
    - Binned image
    """

    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Image must be a 2D numpy array")

    if not isinstance(factor, int) or factor < 1:
        raise ValueError("Factor must be a positive integer")

    # Get the shape of the original image
    height, width = image.shape

    # Calculate the new shape after binning
    new_height = height // factor
    new_width = width // factor

    if new_height == 0 or new_width == 0:
        raise ValueError(f"Binning factor {factor} is too large for the image dimensions {image.shape}")

    assert height % factor == 0 and width % factor == 0, "Image dimensions must be divisible by the factor for even binning"

    # Reshape the image into non-overlapping 2x2 blocks
    reshaped_image = image[:new_height * factor, :new_width * factor].reshape(new_height, factor, new_width, factor)

    # Compute the mean along the last two axes to bin the image
    binned_image = np.mean(reshaped_image, axis=(1, 3))

    return binned_image


import numpy as np

def split_double_planes(image):
    """
    Split planes in the image vertically (top and bottom).

    Parameters:
    - image (numpy.ndarray): 2D, 3D, or 4D array.
        If 2D: shape (height, width)
        If 3D: shape (n_planes, height, width)
        If 4D: shape (n_planes, time, height, width)

    Returns:
    - numpy.ndarray: Array with doubled planes (top and bottom split)
        If input is 2D: shape (2, height // 2, width)
        If input is 3D: shape (2 * n_planes, height // 2, width)
        If input is 4D: shape (2 * n_planes, time, height // 2, width)

    Raises:
    - ValueError: If input is not a 2D, 3D, or 4D numpy array or if height is not even.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")

    if image.ndim not in [2, 3, 4]:
        raise ValueError(f"Input must be a 2D, 3D, or 4D array, got {image.ndim}D")

    if image.ndim == 2:
        h, w = image.shape
        if h % 2 != 0:
            raise ValueError(f"Height must be even, got {h}")

        planes_stack = np.zeros((2, h // 2, w), dtype=image.dtype)
        planes_stack[0] = image[:h // 2, :]
        planes_stack[1] = image[h // 2:, :]

    elif image.ndim == 3:
        n_planes, h, w = image.shape
        if h % 2 != 0:
            raise ValueError(f"Height must be even, got {h}")

        planes_stack = np.zeros((n_planes * 2, h // 2, w), dtype=image.dtype)
        planes_stack[0::2] = image[:, :h // 2, :]
        planes_stack[1::2] = image[:, h // 2:, :]

    else:  # 4D case
        n_planes, t, h, w = image.shape
        if h % 2 != 0:
            raise ValueError(f"Height must be even, got {h}")

        planes_stack = np.zeros((n_planes * 2, t, h // 2, w), dtype=image.dtype)
        planes_stack[0::2] = image[:, :, :h // 2, :]
        planes_stack[1::2] = image[:, :, h // 2:, :]

    return planes_stack



def read_metadata(tifffile_path: str) -> dict:
    """
    Reads metadata from the description field of a TIFF file and returns it as a dictionary.

    Args:
        tifffile_path (str): Path to the TIFF file

    Returns:
        dict: Metadata dictionary from the TIFF file description
    """
    with tifffile.TiffFile(str(tifffile_path)) as tif:
        description = tif.pages[0].description
        if description:
            return json.loads(description)
        return {}

def load_images_to_stack(path, beginning="", ending="*.tif"):
    """
    Loads all images from a directory into a single stack.

    Parameters:
        path (str): The directory path where the images are stored.
        beginning (str): The prefix that the files should start with.
        ending (str): The suffix pattern for the files (default is "*.tif").

    Returns:
        np.ndarray: A stack of images loaded from the directory.
    """
    if not isinstance(path, (str, Path)):
        raise ValueError("Path must be a string or a Path object")

    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"The specified path '{path}' is not a directory")

    # Get all files matching the ending pattern
    files = sorted(Path(path).glob(ending))

    # Filter files that start with the specified beginning
    filtered_files = [file for file in files if file.stem.startswith(beginning)]

    # Check if any files were found
    if not filtered_files:
        raise ValueError(f"No files found matching the pattern '{beginning}*{ending}' in '{path}'")

    # Sort files by their name
    filtered_files.sort()

    # Load images into a stack
    try:
        stack = np.array([tifffile.imread(file) for file in filtered_files])
    except Exception as e:
        raise ValueError(f"Error reading image files: {str(e)}")

    # Internal consistency check
    assert len(stack) == len(filtered_files), "Number of loaded images doesn't match number of files"
    assert all(img.shape == stack[0].shape for img in stack), "All images must have the same shape"

    # Load images into a stack
    return stack


def load_anatomy_stack(anatomy_stack_path, n_channels=2, channel_num=None):
    """
    Load a specific channel from a multi-channel anatomy stack.

    Parameters:
    - anatomy_stack_path (str): Path to the TIFF file containing the anatomy stack.
    - n_channels (int): Total number of channels in the stack.
    - channel_num (int, optional): Specific channel number to load. If None, all channels are returned.

    Returns:
    - numpy.ndarray: 3D or 4D array with the selected channel(s) of the stack.
    """
    try:
        # Load the entire stack using load_tiff_as_hyperstack
        hyperstack = load_tiff_as_hyperstack(anatomy_stack_path, n_channels=n_channels)

        if channel_num is not None:
            if channel_num < 0 or channel_num >= n_channels:
                raise ValueError(f"Invalid channel number. Must be between 0 and {n_channels - 1}")
            return hyperstack[channel_num]
        else:
            return hyperstack
    except Exception as e:
        raise IOError(f"Error loading anatomy stack: {e}")


def load_tiff_as_hyperstack(file_path, n_slices=1, n_channels=1, doubling=False):
    """
    Load a TIFF stack and reshape it into a hyperstack.

    Parameters:
        file_path (str): Path to the TIFF file.
        n_channels (int): Number of channels.
        n_slices (int): Number of z-slices.
        doubling (bool): Whether to split doubled planes.

    Returns:
        numpy.ndarray: A hyperstack array with dimensions [channel, slice, time, y, x].
    """
    # Read tiff file
    try:
        images = tifffile.imread(file_path)
    except Exception as e:
        raise IOError(f"Error reading TIFF file: {e}")

    if images.ndim < 3:
        raise ValueError("Input TIFF must have at least 3 dimensions")

    n_frames = images.shape[0]
    expected_frames = n_channels * n_slices
    if n_frames % expected_frames != 0:
        raise ValueError(f"Total frames ({n_frames}) is not divisible by channels ({n_channels}) * slices ({n_slices})")

    # Reshape and reorder to (channels, slices, time, y, x) Changed below!
    #reshaped_images = images.reshape(n_channels,n_slices,n_frames//n_channels//n_slices,images.shape[1], images.shape[2],order="F")

    # Reshape and reorder to (channels, slices, time, y, x)
    try:
        reshaped_images = images.reshape(n_channels, n_slices, -1, *images.shape[1:], order="F")
    except ValueError as e:
        raise ValueError(f"Reshaping failed. Check if n_channels and n_slices are correct: {e}")

    hyperstack = np.squeeze(reshaped_images)
    print(f"{file_path} loaded. Shape: {hyperstack.shape}")

    # Splitting doubled planes if set
    if doubling:
        return split_double_planes(hyperstack)
    else:
        return hyperstack


def compute_reference_trial_images(trials_path, n_planes, ignore_until_frame=0, save_path=None):
    """
    Compute reference images from trial data.

    Parameters:
    - trials_path (str): Path to the directory containing raw trial data.
    - n_planes (int): Number of planes in each trial.
    - ignore_until_frame (int): Number of initial frames to ignore (default: 0).
    - save_path (str): Path to save the resulting array (default: None).

    Returns:
    - np.ndarray: Array of reference images.
    """
    # Input validation
    if not isinstance(trials_path, str) or not os.path.isdir(trials_path):
        raise ValueError(f"Invalid trials_path: {trials_path}")
    if not isinstance(n_planes, int) or n_planes <= 0:
        raise ValueError(f"Invalid n_planes: {n_planes}")
    if not isinstance(ignore_until_frame, int) or ignore_until_frame < 0:
        raise ValueError(f"Invalid ignore_until_frame: {ignore_until_frame}")

    raw_trial_paths = os.listdir(os.path.join(trials_path, "raw"))
    if not raw_trial_paths:
        raise FileNotFoundError(f"No raw trial data found in {trials_path}")

    ref_images = []

    for trial_path in tqdm(raw_trial_paths, desc="Computing reference trial images"):
        full_trial_path = os.path.join(trials_path, "raw", trial_path)
        if not os.path.isfile(full_trial_path):
            raise FileNotFoundError(f"Trial file not found: {full_trial_path}")

        try:
            trial_images = load_tiff_as_hyperstack(full_trial_path,
                                                   n_channels=1,
                                                   n_slices=n_planes,
                                                   doubling=True)
        except Exception as e:
            raise ValueError(f"Error loading trial images from {full_trial_path}: {str(e)}")

        ref_planes = []

        for plane in trial_images:
            assert plane.ndim == 3, f"Expected 3D plane, got shape {plane.shape}"

            max_frames = plane.shape[0] - ignore_until_frame
            if max_frames <= 0:
                raise ValueError(f"No frames left after ignoring {ignore_until_frame} frames")

            random_frames = random.sample(range(ignore_until_frame, plane.shape[0]),
                                          min(250, max_frames))
            images_array = [plane[frame].flatten() for frame in random_frames]
            corr_mat = np.corrcoef(images_array)
            avg_corr = np.mean(corr_mat, axis=1)
            top_n = min(100, len(avg_corr))
            top_indices = np.argsort(avg_corr)[-top_n:]
            top_images = [plane[idx] for idx in top_indices]
            sum_top_images = np.sum(top_images, axis=0)
            ref_planes.append(sum_top_images)

        ref_images.append(np.stack(ref_planes, axis=0))

    # Convert list of arrays to a single numpy array
    ref_images_array = np.stack(ref_images, axis=1)

    if save_path is not None:
        try:
            save_array_as_hyperstack_tiff(save_path, ref_images_array)
        except Exception as e:
            print(f"Warning: Failed to save array to {save_path}: {str(e)}")

    return ref_images_array



def coarse_plane_detection_in_stack(plane, stack, plot_all_correlations=False, z_range=None):
    """
    Correlate a given plane with slices in a 3D stack.

    Parameters:
    - plane (np.ndarray): The 2D plane to be matched.
    - stack (np.ndarray): 3D stack where the plane is to be searched.
    - plot_all_correlations (bool): If True, plot correlations for all slices.
    - z_range (None, tuple, list, or range): Range of z-slices to search.
      If None, search all slices. If tuple or list, should be [start, end].

    Returns:
    - tuple: (max_corr, max_position, all_correlations)
      max_corr (float): Maximum correlation value.
      max_position (tuple): Position of the best match (z, y, x).
      all_correlations (list): Correlation values for all processed slices.

    """
    if not isinstance(plane, np.ndarray) or plane.ndim != 2:
        raise ValueError("Plane must be a 2D numpy array")
    if not isinstance(stack, np.ndarray) or stack.ndim != 3:
        raise ValueError("Stack must be a 3D numpy array")

    # Determine slices to process
    if z_range is None:
        slices = range(stack.shape[0])
    elif isinstance(z_range, (tuple, list)) and len(z_range) == 2:
        start, end = z_range
        if start < 0 or end > stack.shape[0] or start >= end:
            raise ValueError("Invalid z_range")
        slices = range(start, end)
    elif isinstance(z_range, range):
        slices = z_range
    else:
        raise ValueError("z_range must be None, a tuple/list of (start, end), or a range object")

    max_corr = -np.inf
    max_position = None
    all_correlations = []

    for slice_idx in slices:
        corr = feature.match_template(stack[slice_idx], plane, pad_input=True, mode="mean")
        max_value = np.max(corr)
        position = np.unravel_index(np.argmax(corr), corr.shape)
        all_correlations.append(max_value)

        if max_value > max_corr:
            max_corr = max_value
            max_position = (slice_idx, *position)

    if plot_all_correlations:
        plt.figure(figsize=(10, 6))
        plt.scatter(list(slices), all_correlations)
        plt.xlabel('Slice')
        plt.ylabel('Correlation')
        plt.title('Correlation of Each Slice')
        plt.show()

    return max_corr, max_position, all_correlations


def crop_stack_to_matched_plane(stack, plane, position):
    """
    Crop a stack to match a plane at a given position, padding if necessary.

    Parameters:
    - stack (np.ndarray): 3D array representing the image stack.
    - plane (np.ndarray): 2D array representing the plane to match.
    - position (tuple): (slice_idx, y_offset, x_offset) of the matched position.

    Returns:
    - np.ndarray: Cropped slice from the stack, padded if necessary.

    Raises:
    - ValueError: If inputs are invalid.
    """
    if not isinstance(stack, np.ndarray) or stack.ndim != 3:
        raise ValueError("Stack must be a 3D numpy array")
    if not isinstance(plane, np.ndarray) or plane.ndim != 2:
        raise ValueError("Plane must be a 2D numpy array")
    if not isinstance(position, tuple) or len(position) != 3:
        raise ValueError("Position must be a tuple of (slice_idx, y_offset, x_offset)")

    slice_idx, y_offset, x_offset = position
    height, width = plane.shape
    stack_depth, stack_height, stack_width = stack.shape

    # Calculate crop boundaries
    y_start = y_offset - height // 2
    y_end = y_start + height
    x_start = x_offset - width // 2
    x_end = x_start + width

    # Calculate padding
    pad_top = max(0, -y_start)
    pad_bottom = max(0, y_end - stack_height)
    pad_left = max(0, -x_start)
    pad_right = max(0, x_end - stack_width)

    # Adjust crop boundaries if they're outside the stack
    y_start = max(0, y_start)
    y_end = min(stack_height, y_end)
    x_start = max(0, x_start)
    x_end = min(stack_width, x_end)

    # Crop the slice
    cropped_slice = stack[slice_idx, y_start:y_end, x_start:x_end]

    # Pad the cropped slice if necessary
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cropped_slice = np.pad(cropped_slice, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='mean')

    # Ensure the cropped slice has the same shape as the plane
    assert cropped_slice.shape == plane.shape, "Cropped slice shape does not match plane shape"

    return cropped_slice



def plot_matched_plane_and_cropped_slice(stack, plane, position, match_hist=True, return_plot=False):
    """
        Visualize the matched plane from the stack and a cropped slice from the rotated stack.

    Parameters:
    - stack (np.ndarray): 3D image stack.
    - plane (np.ndarray): 2D plane to be visualized.
    - position (tuple): (slice index, y_offset, x_offset) - describes where in the rotated stack the match was found.
    - match_hist (bool): Whether to match histograms of the cropped slice to the plane.
    - return_plot (bool): Whether to return the plot object instead of displaying it.

    Returns:
    - If return_plot is True, returns the matplotlib figure object. Otherwise, displays the plot.
    """
    try:
        cropped_slice = crop_stack_to_matched_plane(stack, plane, position)
    except ValueError as e:
        raise ValueError(f"Error in cropping stack:{str(e)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(plane, cmap='gray')
    ax1.set_title("Plane")

    if match_hist:
        try:
            matched_slice = match_histograms(cropped_slice, plane)
        except Exception as e:
            print(f"Warning: Histogram matching failed. Displaying original cropped slice. Error: {str(e)}")
            matched_slice = cropped_slice
        ax2.imshow(matched_slice, cmap='gray')
    else:
        ax2.imshow(cropped_slice, cmap='gray')
    ax2.set_title("Matched cropped slice")
    ax2.axis('off')

    plt.tight_layout()

    if return_plot:
        return fig
    else:
        plt.show()


# def fine_plane_detection_in_stack_by_tiling(tiles, stack, tiles_filter, z_range=None):
#     """
#     Find the best matching plane in a stack for each tile, based on a filter.
#
#     Parameters:
#     - tiles: A 4D numpy array of tiles with shape (ny, nx, tile_height, tile_width).
#     - stack: A 3D numpy array representing the stack to search through.
#     - tiles_filter: A 2D numpy array (mask) with shape (ny, nx), where 1 indicates a tile to process and 0 a tile to ignore.
#
#     Returns:
#     - best_plane_matrix: A 2D numpy array storing the best plane's position for each tile.
#     - all_correlations_matrix: A 3D numpy array storing all correlations for each tile across the stack.
#     """
#     # E.g. best_plane_matrix, all_correlations_matrix = find_best_planes(tiles, stack, tiles_filter)
#
#     ny, nx = tiles.shape[:2]
#     best_plane_matrix = np.zeros(shape=(ny, nx, 3), dtype=int)
#     if z_range == None:
#         all_correlations_matrix = np.zeros(
#             (ny, nx, stack.shape[0]))  # Assuming the third dimension of stack is the depth (z)
#     else:
#         all_correlations_matrix = np.zeros((ny, nx, stack[z_range[0]:z_range[-1]].shape[0]))
#
#     for i in range(ny):
#         for j in range(nx):
#             if tiles_filter[i, j] == 1:
#
#                 max_corr, max_position, all_correlations = coarse_plane_detection_in_stack(tiles[i, j], stack, z_range=z_range)
#
#                 best_plane_matrix[i, j] = max_position
#                 all_correlations_matrix[i, j, :] = all_correlations
#
#     return best_plane_matrix, all_correlations_matrix




def fine_plane_detection_in_stack_by_tiling(tiles, stack, z_range=None, min_tiles=3, correlation_threshold=0.5):
    """
    Find the best matching plane in a stack for each tile, selecting the best-matching tiles.

    Parameters:
    - tiles (np.ndarray): A 4D numpy array of tiles with shape (ny, nx, tile_height, tile_width).
    - stack (np.ndarray): A 3D numpy array representing the stack to search through.
    - z_range (tuple or None): Optional range of z-slices to search, given as (start, end).
    - min_tiles (int): Minimum number of tiles to select (default: 3).
    - correlation_threshold (float): Minimum correlation value to consider a tile (default: 0.5).

    Returns:
    - best_plane_matrix (np.ndarray): A 3D numpy array storing the best plane's position for each tile.
    - all_correlations_matrix (np.ndarray): A 3D numpy array storing all correlations for each tile across the stack.
    - selected_tiles (np.ndarray): A 2D boolean array indicating which tiles were selected.

    Raises:
    - ValueError: If inputs are invalid or incompatible.
    """
    # Input validation
    if not isinstance(tiles, np.ndarray) or tiles.ndim != 4:
        raise ValueError("Tiles must be a 4D numpy array")
    if not isinstance(stack, np.ndarray) or stack.ndim != 3:
        raise ValueError("Stack must be a 3D numpy array")

    ny, nx = tiles.shape[:2]

    # Handle z_range
    if z_range is None:
        z_start, z_end = 0, stack.shape[0]
    else:
        if not isinstance(z_range, (tuple, list)) or len(z_range) != 2:
            raise ValueError("z_range must be a tuple or list of (start, end)")
        z_start, z_end = z_range
        if z_start < 0 or z_end > stack.shape[0] or z_start >= z_end:
            raise ValueError("Invalid z_range")

    best_plane_matrix = np.zeros((ny, nx, 3), dtype=int)
    all_correlations_matrix = np.zeros((ny, nx, z_end - z_start))
    max_correlations = np.zeros((ny, nx))

    for i in range(ny):
        for j in range(nx):
            max_corr, max_position, correlations = coarse_plane_detection_in_stack(tiles[i, j], stack,
                                                                                       z_range=z_range)
            # Apply median filter to smooth correlations
            smoothed_correlations = medfilt(correlations, kernel_size=7)

            # Find the maximum correlation using smoothed data
            max_corr_index = np.argmax(smoothed_correlations)
            max_corr = smoothed_correlations[max_corr_index]
            max_position = (z_start + max_corr_index, *max_position[1:])

            best_plane_matrix[i, j] = max_position
            all_correlations_matrix[i, j, :] = smoothed_correlations
            max_correlations[i, j] = max_corr

    # Select the best tiles
    selected_tiles = (max_correlations >= correlation_threshold)
    if np.sum(selected_tiles) < min_tiles:
        # If we don't have enough tiles above the threshold, select the top n tiles
        flat_indices = np.argsort(max_correlations.ravel())[::-1][:min_tiles]
        selected_tiles = np.zeros_like(selected_tiles, dtype=bool)
        selected_tiles.ravel()[flat_indices] = True
        logging.info(f"Not enough tiles above the threshold {correlation_threshold}, top {min_tiles} tiles selected")
    else:
        logging.info(f"Selected {np.sum(selected_tiles)} tiles above {correlation_threshold} out of {ny * nx} tiles")

    return best_plane_matrix, all_correlations_matrix, selected_tiles




# %%
def plot_image_correlation(tiles, stack, best_plane_matrix, all_correlations_matrix, selected_tiles=None,
                           return_plot=False):
    """
    Plots a grid of images, cropped slices, and correlation scatter plots.

    Parameters:
    - tiles: 4D numpy array of images.
    - stack: 3D numpy array representing an interpolated stack of images.
    - best_plane_matrix: 2D numpy array with the best plane indices for each tile.
    - all_correlations_matrix: 3D numpy array containing correlation data for scatter plots.
    - selected_tiles: 2D boolean array indicating which tiles were selected. If None, all tiles are selected.
    - return_plot: Boolean, whether to return the plot object instead of displaying it.

    Returns:
    - If return_plot is True, returns the matplotlib figure object. Otherwise, displays the plot.
    """
    num_rows, num_cols = tiles.shape[:2]

    # If selected_tiles is None, create a boolean array with all tiles selected
    if selected_tiles is None:
        selected_tiles = np.ones((num_rows, num_cols), dtype=bool)

    fig = plt.figure(figsize=(16, 14))
    gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
    gs_sub1 = GridSpecFromSubplotSpec(nrows=num_rows * 3, ncols=num_cols, subplot_spec=gs_main[0], hspace=0.5,
                                      wspace=0.3)

    global_y_min, global_y_max = np.inf, -np.inf
    for row in range(num_rows):
        for col in range(num_cols):
            local_min = all_correlations_matrix[row, col].min()
            local_max = all_correlations_matrix[row, col].max()
            global_y_min = min(global_y_min, local_min)
            global_y_max = max(global_y_max, local_max)

    for row in range(num_rows):
        for col in range(num_cols):
            # Original tile
            ax = fig.add_subplot(gs_sub1[3 * row, col])
            ax.imshow(tiles[row, col], cmap="binary")
            ax.set_title(f'Image ({row},{col})')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # Cropped slice
            ax = fig.add_subplot(gs_sub1[3 * row + 1, col])
            cropped_slice = crop_stack_to_matched_plane(stack, tiles[row, col],
                                                        tuple(best_plane_matrix[row, col].astype(int)))
            ax.imshow(cropped_slice, cmap="binary")
            ax.set_title(f'Slice {best_plane_matrix[row, col][0] - stack.shape[0] // 2}')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # Correlation plot
            ax = fig.add_subplot(gs_sub1[3 * row + 2, col])
            correlations = all_correlations_matrix[row, col]
            smoothed_correlations = medfilt(correlations, kernel_size=7)
            ax.plot(range(len(correlations)), correlations, alpha=0.5, label='Raw')
            ax.plot(range(len(smoothed_correlations)), smoothed_correlations, label='Smoothed')

            max_corr_index = np.argmax(smoothed_correlations)
            ax.scatter(max_corr_index, smoothed_correlations[max_corr_index],
                       color='red', s=50, zorder=5, label='Best match')
            ax.set_ylim(global_y_min, global_y_max)
            ax.set_title(f'Correlation ({row},{col})')
            ax.set_xticklabels([])
            if col > 0:
                ax.set_yticklabels([])

            # Highlight selected tiles
            if selected_tiles[row, col]:
                for subplot in range(3):
                    for spine in ax.spines.values():
                        spine.set_edgecolor('green')
                        spine.set_linewidth(2)

    # Add a legend to the last subplot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if return_plot:
        return fig
    else:
        plt.show()


def warp_stack_to_plane(stack, plane, transformation, thickness):
    """
    Warp a stack of images to a specified plane using a given transformation.

    Parameters:
    - stack (np.ndarray): 3D numpy array representing the stack of images.
    - plane (np.ndarray): 2D numpy array representing the target plane.
    - transformation (SimilarityTransform): Transformation to apply to warp the stack to the plane.
    - thickness (int): Thickness of the warped stack.

    Returns:
    - np.ndarray: Interpolated stack warped to the specified plane.
    """
    # Input validation
    if not isinstance(stack, np.ndarray) or stack.ndim != 3:
        raise ValueError("Stack must be a 3D numpy array")
    if not isinstance(plane, np.ndarray) or plane.ndim != 2:
        raise ValueError("Plane must be a 2D numpy array")
    if not isinstance(transformation, SimilarityTransform):
        raise ValueError("Transformation must be a SimilarityTransform object")
    if not isinstance(thickness, int) or thickness <= 0:
        raise ValueError("Thickness must be a positive integer")

    # Calculate half thickness
    t_2 = thickness // 2

    try:
        # Create meshgrid for Z, Y, and X coordinates
        zz, yy, xx = np.meshgrid(
            np.linspace(-t_2, t_2 + 1, num=int(thickness * transformation.scale) + 1, endpoint=False),
            np.linspace(0, plane.shape[0], num=int(transformation.scale * plane.shape[0]) + 1, endpoint=False),
            np.linspace(0, plane.shape[1], num=int(transformation.scale * plane.shape[1]) + 1, endpoint=False),
            indexing='ij'
        )

        # Plane coordinates +/- half thickness e.g. -3, -2, -1, 0, 1, 2, 3 in Z
        plane_coords = np.stack([zz.flatten(), yy.flatten(), xx.flatten()], axis=1)

        # Calculate target stack coordinates in the overview stack
        plane_coords_in_stack = transformation(plane_coords)

        # Round target stack coordinates to integer grid
        plane_stack_source_coordinates = np.round(plane_coords_in_stack).astype(np.int16)

        # Remove coordinates outside of the overview stack
        valid_coords = plane_stack_source_coordinates[np.all(plane_stack_source_coordinates >= 0, axis=1)]
        valid_coords = valid_coords[np.all(valid_coords < stack.shape, axis=1)]

        # Convert coordinates into index which works on the flattened stack
        idx = np.ravel_multi_index((valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]), stack.shape)

        # Get coordinates for target stack (thickness - 2 (-1 on both sides of Z))
        zz, yy, xx = np.meshgrid(
            range(-t_2 + 1, t_2), range(plane.shape[0]), range(plane.shape[1]), indexing='ij'
        )
        xi = transformation(np.stack([zz.flatten(), yy.flatten(), xx.flatten()], axis=1))

        # Get interpolated target stack
        interpolated_stack = griddata(
            points=valid_coords,
            values=stack.flatten()[idx],
            xi=xi,
            method='nearest'
        ).reshape(zz.shape)

        return interpolated_stack

    except Exception as e:
        raise RuntimeError(f"Error during stack warping: {str(e)}")


def warp_plane_to_stack(plane_2d, stack, transformation, thickness=3):
    """
    Warp a 2D plane into a 3D stack using a given transformation.
    The 2D plane is first converted to a 3D volume by stacking identical planes.

    Parameters:
    - plane_2d (np.ndarray): 2D numpy array representing the source plane.
    - stack (np.ndarray): 3D numpy array representing the target stack.
    - transformation (SimilarityTransform): Transformation to apply to warp the plane to the stack.
    - thickness (int): Number of identical planes to stack together.

    Returns:
    - np.ndarray: 3D array representing the plane warped into the stack space.
    """
    # Create a 3D volume by stacking identical planes
    # upsampled_plane = scipy.ndimage.zoom(plane_2d, zoom=(10, 4), order=1)
    plane_3d = np.stack([plane_2d] * thickness, axis=0)

    # Input validation
    if not isinstance(plane_3d, np.ndarray) or plane_3d.ndim != 3:
        raise ValueError("Plane must be a 3D numpy array")
    if not isinstance(stack, np.ndarray) or stack.ndim != 3:
        raise ValueError("Stack must be a 3D numpy array")
    if not isinstance(transformation, SimilarityTransform):
        raise ValueError("Transformation must be a SimilarityTransform object")

    try:
        # Create a result array with the same shape as the stack
        result = np.zeros_like(stack)

        # Create a denser grid of coordinates for the 3D plane
        zz, yy, xx = np.meshgrid(
            np.arange(plane_3d.shape[0]),  # Keep z-axis at original resolution
            np.linspace(0, plane_3d.shape[1] - 1, plane_3d.shape[1] * 4),  # 4x denser sampling in y
            np.linspace(0, plane_3d.shape[2] - 1, plane_3d.shape[2] * 4),  # 4x denser sampling in x
            indexing='ij'
        )

        # Create 3D coordinates for the plane
        plane_coords = np.column_stack((
            zz.flatten(),  # z coordinates
            yy.flatten(),  # y coordinates
            xx.flatten()  # x coordinates
        ))

        # print(f"Plane coordinates shape: {plane_coords.shape}")

        # Adjust z-coordinates to center the plane stack
        # This shifts the z-coordinate so that the center of the plane stack is at z=0
        plane_coords[:, 0] = plane_coords[:, 0] - (thickness - 1) / 2

        # Add homogeneous coordinate for transformation
        homogeneous_coords = np.column_stack((plane_coords, np.ones(plane_coords.shape[0])))

        # Apply transformation to get coordinates in the stack
        stack_coords_homogeneous = homogeneous_coords @ transformation.params.T
        stack_coords = stack_coords_homogeneous[:, :3]

        # print(f"Stack coordinates shape: {stack_coords.shape}")

        # Round to get integer indices
        stack_indices = np.round(stack_coords).astype(np.int32)

        # Filter out coordinates outside the stack bounds
        valid_mask = (
                (stack_indices[:, 0] >= 0) & (stack_indices[:, 0] < stack.shape[0]) &
                (stack_indices[:, 1] >= 0) & (stack_indices[:, 1] < stack.shape[1]) &
                (stack_indices[:, 2] >= 0) & (stack_indices[:, 2] < stack.shape[2])
        )

        # Get valid indices
        valid_stack_indices = stack_indices[valid_mask]

        # Get original plane coordinates (adjusted back to positive indices)
        valid_plane_indices = plane_coords[valid_mask].astype(np.int32)
        valid_plane_indices[:, 0] += (thickness - 1) // 2  # Adjust z back to original range

        # Ensure plane indices are within bounds
        valid_plane_indices = np.clip(
            valid_plane_indices,
            [0, 0, 0],
            [plane_3d.shape[0] - 1, plane_3d.shape[1] - 1, plane_3d.shape[2] - 1]
        )

        # Get the values from the plane
        valid_plane_values = plane_3d[
            valid_plane_indices[:, 0],
            valid_plane_indices[:, 1],
            valid_plane_indices[:, 2]
        ]

        # Place the plane values into the result array
        for i, (z, y, x) in enumerate(valid_stack_indices):
            result[z, y, x] = valid_plane_values[i]

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Error during plane warping: {str(e)}")

def slice_into_uniform_tiles(image, nx, ny, plot=True, return_plot=False):
    """
    Slice an image into uniformly sized tiles.

    Parameters:
    - image (np.ndarray): 2D NumPy array representing the image to be sliced.
    - nx (int): Number of tiles along the x-axis (width).
    - ny (int): Number of tiles along the y-axis (height).
    - plot (bool): If True, plot the tiles.
    - return_plot (bool): If True, return the plot objects instead of displaying.

    Returns:
    - np.ndarray: A 4D NumPy array with shape (ny, nx, tile_height, tile_width).
    - tuple: Tile size (height, width).
    - tuple: Adjusted image size (height, width).
    - (optional) matplotlib objects: fig, axs (if return_plot is True)

    """

    # Input validation
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Image must be a 2D NumPy array")
    if not isinstance(nx, int) or not isinstance(ny, int) or nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers")

    height, width = image.shape
    print(f"Original image size: {height}x{width}")

    # Calculate the size of each tile
    M, N = height // ny, width // nx
    if M == 0 or N == 0:
        raise ValueError("Tile size is too small. Reduce nx or ny.")
    print(f"Tile size: {M}x{N}")

    # Adjust the height and width to ensure all tiles are of equal size
    adjusted_height = M * ny
    adjusted_width = N * nx
    print(f"Adjusted image size for perfect slicing: {adjusted_height}x{adjusted_width}")

    # Adjust image size for slicing
    adjusted_image = image[:adjusted_height, :adjusted_width]

    # Generate perfectly sized tiles
    tiles = np.array([adjusted_image[x:x + M, y:y + N]
                      for x in range(0, adjusted_height, M)
                      for y in range(0, adjusted_width, N)])

    print(f"Number of tiles: {len(tiles)}")

    reshaped_tiles = tiles.reshape(ny, nx, M, N)

    if plot:
        fig, axs = plt.subplots(nrows=ny, ncols=nx, figsize=(nx * 3, ny * 3))
        if ny == 1 and nx == 1:
            axs = np.array([[axs]])
        elif ny == 1 or nx == 1:
            axs = axs.reshape(ny, nx)

        for i in range(ny):
            for j in range(nx):
                axs[i, j].imshow(match_histograms(reshaped_tiles[i, j], reshaped_tiles[0, 0]), cmap='gray')
                axs[i, j].axis('off')
                axs[i, j].set_title(f'Tile ({i},{j})')

        plt.tight_layout()

        if return_plot:
            return reshaped_tiles, (M, N), (adjusted_height, adjusted_width), fig, axs
        else:
            plt.show()

    return reshaped_tiles, (M, N), (adjusted_height, adjusted_width)



def save_array_as_hyperstack_tiff(path, array, transpose=False):
    if not isinstance(array, np.ndarray):
            array = np.array(array)
    if transpose:
        array_reshaped = array.transpose(1, 0, 2, 3).astype(np.float32)
        tifffile.imwrite(path, array_reshaped, imagej=True)
    else:
        array = array.astype(np.float32)
        tifffile.imwrite(path, array, imagej=True)


## Suggestion:
# def save_array_as_hyperstack_tiff(path, array, transpose=False, compression='zlib', metadata=None):
#     """
#     Save a numpy array as a hyperstack TIFF file.
#
#     Parameters:
#     - path (str): The file path where the TIFF will be saved.
#     - array (np.ndarray): The array to be saved. Should be 3D or 4D.
#     - transpose (bool): If True, transpose the array before saving.
#     - compression (str): Compression method. Options: 'zlib', 'lzw', None.
#     - metadata (dict): Optional metadata to include in the TIFF file.
#
#     Raises:
#     - ValueError: If the input array is not 3D or 4D.
#     - IOError: If there's an issue saving the file.
#     """
#     try:
#         # Input validation
#         if not isinstance(array, np.ndarray):
#             array = np.array(array)
#
#         if array.ndim not in [3, 4]:
#             raise ValueError("Input array must be 3D or 4D")
#
#         # Ensure the directory exists
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#
#         # Prepare the array
#         if transpose:
#             if array.ndim == 4:
#                 array_reshaped = array.transpose(1, 0, 2, 3).astype(np.float32)
#             else:  # 3D array
#                 array_reshaped = array.transpose(1, 0, 2).astype(np.float32)
#         else:
#             array_reshaped = array.astype(np.float32)
#
#         # Prepare metadata
#         imagej_metadata = {'hyperstack': True}
#         if metadata:
#             imagej_metadata.update(metadata)
#
#         # Save the file
#         tifffile.imwrite(
#             path,
#             array_reshaped,
#             imagej=True,
#             metadata=imagej_metadata,
#             compression=compression
#         )
#         print(f"Successfully saved hyperstack TIFF to {path}")
#
#     except Exception as e:
#         raise IOError(f"Error saving hyperstack TIFF: {str(e)}")



def plane_detection_with_iterative_alignment(plane, stack, equalize=True, binning=True, plot=True, nx=2, ny=3,
                                             tiles_filter=None, thickness_values=None, return_plot=False):
    """
    Detects a plane within a 3D image stack using iterative alignment of tiles.

    This function performs plane detection in a 3D image stack by iteratively aligning tiles
    from a 2D plane image to the stack. It uses a combination of image processing techniques
    including adaptive histogram equalization, binning, coarse detection, and fine alignment
    using tile-based correlation.

    Parameters:
    -----------
    plane : numpy.ndarray
        2D array representing the plane to be detected in the stack.
    stack : numpy.ndarray
        3D array representing the image stack to search for the plane.
    equalize : bool, optional (default=True)
        If True, apply adaptive histogram equalization to the images.
    binning : bool, optional (default=True)
        If True, apply binning to reduce image size for initial coarse detection.
    plot : bool, optional (default=True)
        If True, generate plots at various stages of the detection process.
    nx : int, optional (default=2)
        Number of tiles in the x-direction for fine alignment.
    ny : int, optional (default=3)
        Number of tiles in the y-direction for fine alignment.
    tiles_filter : numpy.ndarray, optional (default=None)
        2D boolean array to select which tiles to use in the alignment process.
        If None, a default filter is used.
    thickness_values : list of int, optional (default=None)
        List of thickness values to use in iterative alignment.
        If None, default values [100, 50, 50, 30, 30, 20] are used.
    return_plot: Boolean, whether to return the plot object instead of displaying it.

    Returns:
    --------
    current_tform : numpy.ndarray
        4x4 transformation matrix representing the final alignment.
    all_transformation_matrices : list of numpy.ndarray
        List of all intermediate transformation matrices.

    Notes:
    ------
    The function performs the following main steps:
    1. Image preprocessing (equalization and binning)
    2. Coarse detection of the plane in the stack
    3. Iterative fine alignment using tile-based correlation
    4. Transformation estimation and refinement

"""

    # Set default tiles_filter if not provided
    if tiles_filter is None:
        tiles_filter = np.array([[1, 1],
                                 [1, 1],
                                 [0, 0]])

    # Equalization
    if equalize:
        try:
            equalized_plane = equalize_adapthist(plane, clip_limit=0.03)
        except:
            equalized_plane = equalize_adapthist(rescale_intensity(plane, out_range=(0, 1)), clip_limit=0.03)

        equalized_stack = np.array([
            equalize_adapthist(rescale_intensity(slice, out_range=(0, 1)), clip_limit=0.03)
            for slice in stack
        ])
    else:
        equalized_plane = plane
        equalized_stack = stack

    # Binning
    if binning:
        binned_plane = bin_image(equalized_plane)
        binned_stack = np.array([bin_image(slice) for slice in equalized_stack])
    else:
        binned_plane = equalized_plane
        binned_stack = equalized_stack

    # Coarse detection
    max_corr_coarse, max_position_coarse, _ = coarse_plane_detection_in_stack(binned_plane, binned_stack,
                                                                              plot_all_correlations=plot)

    if plot:
        plot_matched_plane_and_cropped_slice(binned_stack, binned_plane, max_position_coarse)

    if thickness_values is None:
        thickness_values = [100, 50, 50, 30, 30, 20, 20]

    # Slice into tiles
    tiles, tile_size, adj_image_size = slice_into_uniform_tiles(equalized_plane, nx, ny, plot=plot)

    x = np.arange(tile_size[1] // 2, adj_image_size[1], tile_size[1])
    y = np.arange(tile_size[0] // 2, adj_image_size[0], tile_size[0])
    xv, yv = np.meshgrid(x, y)

    points_filter = np.where(tiles_filter.flatten() == 1)

    all_transformation_matrices = [np.eye(4)]
    current_tform = np.eye(4)

    # Iterative alignment of tiles
    for thickness in thickness_values:
        print(f"Thickness {thickness}")
        if len(all_transformation_matrices) == 1:
            min_z_range = max(0, max_position_coarse[0] - thickness // 2)
            max_z_range = min(stack.shape[0], max_position_coarse[0] + thickness // 2)
            best_plane_matrix, all_correlations_matrix = fine_plane_detection_in_stack_by_tiling(
                tiles, equalized_stack, tiles_filter, z_range=[min_z_range, max_z_range])

            if plot:
                plot_image_correlation(tiles, equalized_stack, best_plane_matrix, all_correlations_matrix)

            lm_plane_points = np.array([(0, yv[i, j], xv[i, j])
                                        for i in range(ny) for j in range(nx)])
        else:
            interpolated_stack = warp_stack_to_plane(stack, plane, SimilarityTransform(current_tform), thickness)

            best_plane_matrix, all_correlations_matrix = fine_plane_detection_in_stack_by_tiling(
                tiles, interpolated_stack, tiles_filter)

            if plot:
                plot_image_correlation(tiles, interpolated_stack, best_plane_matrix, all_correlations_matrix)

            lm_plane_points = np.array([(interpolated_stack.shape[0] // 2, yv[i, j], xv[i, j])
                                        for i in range(ny) for j in range(nx)])

        lm_stack_points = np.array([best_plane_matrix[i, j] for i in range(ny) for j in range(nx)])

        source = lm_plane_points[points_filter]
        target = lm_stack_points[points_filter]

        tform = SimilarityTransform()
        tform.estimate(source, target)

        all_transformation_matrices.append(tform.params)
        current_tform = np.linalg.multi_dot(all_transformation_matrices[::-1])

    # Plot the last matched slice with plane

    last_matched_slice = warp_stack_to_plane(stack, plane, SimilarityTransform(current_tform), thickness_values[-1])[
        thickness_values[-1] // 2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(plane, cmap='gray')
    ax1.set_title('Original Plane')
    ax1.axis('off')

    ax2.imshow(last_matched_slice, cmap='gray')
    ax2.set_title('Last Matched Slice')
    ax2.axis('off')

    plt.tight_layout()
    if return_plot:
        return fig
    else:
        plt.show()

    return current_tform, all_transformation_matrices

# TODO: why red dot is wrong?
# TODO: Should give the original slice index on the plots also give the correlation max value
# TODO: check if only selected tiles are used for warp
# TODO: Correlation check should be done with the whole plane
# TODO: Last plot with: correlation development until stagnation. And additional image with differece of original plane and last matched slice.
# TODO: Iteration stop when stagnated three times.


def calculate_manders_coefficient_3d(mask_stack, channel_stack):
    # Label the mask stack
    labels = label(mask_stack)
    props = regionprops(labels)

    # Initiate Manders' coefficients dictionary and Manders' coefficient based colored stack
    manders_results = {}
    mask_colored_stack = np.zeros_like(mask_stack, dtype=float)

    # Binarize channel stack
    channel_threshold = threshold_otsu(channel_stack)
    binary_channel_stack = channel_stack > channel_threshold

    # Binarize the mask (assuming it's already binary)
    binary_mask_stack = mask_stack > 0

    for prop in props:
        # Get the bounding box of the region
        minr, minc, minz, maxr, maxc, maxz = prop.bbox

        # Extract the corresponding region from the red channel and the mask
        channel_region = binary_channel_stack[minr:maxr, minc:maxc, minz:maxz]
        mask_region = binary_mask_stack[minr:maxr, minc:maxc, minz:maxz]

        # Calculate Manders' colocalization coefficients
        manders_coeff = manders_coloc_coeff(mask_region, channel_region)

        # Store the results
        manders_results[prop.label] = manders_coeff

        # Color the stack based on the Manders' coefficients
        mask_colored_stack[minr:maxr, minc:maxc, minz:maxz][mask_region] = manders_coeff

    return manders_results, mask_colored_stack

# Similarity visualization tools

def local_similarity_map(img1, img2, window_size=7):
    similarity = np.zeros_like(img1, dtype=float)
    pad = window_size//2
    padded1 = np.pad(img1, pad)
    padded2 = np.pad(img2, pad)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            window1 = padded1[i:i+window_size, j:j+window_size]
            window2 = padded2[i:i+window_size, j:j+window_size]
            similarity[i,j] = np.corrcoef(window1.flat, window2.flat)[0,1]

    return similarity


def create_difference_maps(img1, img2):
    # Difference map
    diff = np.abs(img1 - img2)

    # Heat map using normalized cross-correlation
    heat_map = feature.match_template(img1, img2, pad_input=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax1.imshow(diff, cmap='RdBu_r')
    ax1.set_title('Difference Map')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(heat_map, cmap='hot')
    ax2.set_title('Heat Map')
    plt.colorbar(im2, ax=ax2)


def create_checkerboard(img1, img2, tile_size=50):
    checker = np.zeros_like(img1)
    for i in range(0, img1.shape[0], tile_size):
        for j in range(0, img1.shape[1], tile_size):
            i_end = min(i + tile_size, img1.shape[0])
            j_end = min(j + tile_size, img1.shape[1])
            if (i//tile_size + j//tile_size) % 2:
                checker[i:i_end, j:j_end] = img1[i:i_end, j:j_end]
            else:
                checker[i:i_end, j:j_end] = img2[i:i_end, j:j_end]
    return checker


def edge_overlay(img1, img2):
    edges1 = feature.canny(img1)
    edges2 = feature.canny(img2)

    overlay = np.zeros((*img1.shape, 3))
    overlay[edges1] = [1, 0, 0]  # Red for img1 edges
    overlay[edges2] = [0, 1, 0]  # Green for img2 edges

    return overlay


def visualize_registration_quality(img1, img2, window_size=20):
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(2,2)
    # Checkerboard
    ax1.imshow(create_checkerboard(img1, img2, tile_size=window_size), cmap='gray')
    ax1.set_title('Checkerboard Pattern')

    # Difference & Heat maps
    diff = np.abs(img1 - img2)
    im2 = ax2.imshow(diff, cmap='RdBu_r')
    ax2.set_title('Difference Map')
    plt.colorbar(im2, ax=ax2)

    # Local similarity
    sim_map = local_similarity_map(img1, img2, window_size=window_size)
    im3 = ax3.imshow(sim_map, cmap='viridis')
    ax3.set_title('Local Similarity')
    plt.colorbar(im3, ax=ax3)

    # Edge overlay
    ax4.imshow(edge_overlay(img1, img2))
    ax4.set_title('Edge Overlay')

    plt.tight_layout()
    plt.show()