import os
import numpy as np
from pathlib import Path

import tifffile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.interpolate import griddata
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
    """
    return np.pad(stack, ((0, 0), (margin, margin), (margin, margin)), mode='constant')


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




def split_double_planes(hyperstack_img):
    """
    Split planes in the hyperstack based on doubling factor.

    Parameters:
    - hyperstack_img (numpy.ndarray): 4D array of shape (n_planes, time, width, height).

    Returns:
    - numpy.ndarray: 4D array with doubled planes (2 * n_planes, time, width // 2, height).
    """
    n_planes = hyperstack_img.shape[0]
    doubling = 2
    _, t, w, h = hyperstack_img.shape

    # Efficient slicing
    planes_stack = np.zeros((n_planes * doubling, t, w // doubling, h), dtype=np.float32)
    planes_stack[0::2] = hyperstack_img[:, :, :w // doubling, :]
    planes_stack[1::2] = hyperstack_img[:, :, w // doubling:, :]

    return planes_stack


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
    # Get all files matching the ending pattern
    files = sorted(Path(path).glob(ending))

    # Filter files that start with the specified beginning
    filtered_files = [file for file in files if file.stem.startswith(beginning)]

    # Sort files by their name
    filtered_files.sort()

    # Load images into a stack
    return np.array([np.array(tifffile.imread(file)) for file in filtered_files])


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
    print(f"{file_path} loaded.")
    # Splitting doubled planes if set
    if doubling:
        return split_double_planes(hyperstack)
    else:
        return hyperstack


def compute_reference_trial_images(trials_path, n_planes, ignore_until_frame=0, save_path=None):
    raw_trial_paths = os.listdir(os.path.join(trials_path, "raw"))
    ref_images = []

    for trial_path in raw_trial_paths:
        trial_images = load_tiff_as_hyperstack(os.path.join(trials_path, "raw", trial_path),
                                               n_channels=1,
                                               n_slices=n_planes,
                                               doubling=True)
        ref_planes = []

        for plane in trial_images:
            random_frames = random.sample(range(ignore_until_frame, plane.shape[0]), min(250, plane.shape[0] - ignore_until_frame))
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
        save_array_as_hyperstack_tiff(save_path, ref_images_array)

    return ref_images_array


def coarse_plane_detection_in_stack(plane, stack, plot_all_correlations=False, z_range=None):
    """
    Correlate a given plane with slices in a 3D stack.
    If an angle is provided, the stack is rotated before correlation along the plane given.

    Parameters:
    - stack: 3D stack where the plane is to be searched.
    - plane: The 2D plane to be matched.

    Returns:
    - (max_corr, max_position): Tuple containing the position of the best match and its coordinates.
    """

    max_corr = -np.inf
    max_position = None

    all_correlations = []

    slices = [*range(stack.shape[0])] if z_range == None else [*range(z_range[0], z_range[-1])]

    # Loop through each slice in the stack (or rotated stack)
    for slice in slices:
        corr = feature.match_template(stack[slice], plane, pad_input=True)
        max_value = np.max(corr)
        position = np.unravel_index(np.argmax(corr), corr.shape)
        all_correlations.append(max_value)

        if max_value > max_corr:
            max_corr = max_value
            max_position = (slice, *position)

    if plot_all_correlations:
        plt.scatter(slices, all_correlations)
        plt.xlabel('Slice')
        plt.ylabel('Correlation')
        plt.title('Correlation of Each Slice')
        plt.show()

    return (max_corr, max_position, all_correlations)


def crop_stack_to_matched_plane(stack, plane, position, blank=200):
    slice_idx, x_offset, y_offset = position
    extended_stack = extend_stack(stack, blank)

    # Calculate the bounds for cropping the matched slice
    width, height = plane.shape
    y_start, y_end = y_offset - height // 2 + blank, y_offset + height // 2 + blank
    x_start, x_end = x_offset - width // 2 + blank, x_offset + width // 2 + blank

    cropped_slice = extended_stack[slice_idx, x_start:x_end, y_start:y_end]

    return cropped_slice


def plot_matched_plane_and_cropped_slice(stack, plane, position, match_hist=True):
    """
    Visualize the matched plane from the stack and a cropped slice from the rotated stack.

    Parameters:
    - stack: 3D image stack.
    - plane: 2D plane to be visualized.
    - position: (slice index, x_offset, y_offset) - describes where in the rotated stack the match was found.

    Returns:
    - Displays a side-by-side visualization of the plane and the matched slice.
    """

    cropped_slice = crop_stack_to_matched_plane(stack, plane, position)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(plane, cmap='gray')
    ax1.set_title(f"Plane")
    if match_hist:
        ax2.imshow(match_histograms(cropped_slice, plane), cmap='gray')
    else:
        ax2.imshow(cropped_slice, cmap='gray')
    ax2.set_title(f"Matched cropped slice ")
    plt.show()


def fine_plane_detection_in_stack_by_tiling(tiles, stack, tiles_filter, z_range=None):
    """
    Find the best matching plane in a stack for each tile, based on a filter.

    Parameters:
    - tiles: A 4D numpy array of tiles with shape (ny, nx, tile_height, tile_width).
    - stack: A 3D numpy array representing the stack to search through.
    - tiles_filter: A 2D numpy array (mask) with shape (ny, nx), where 1 indicates a tile to process and 0 a tile to ignore.

    Returns:
    - best_plane_matrix: A 2D numpy array storing the best plane's position for each tile.
    - all_correlations_matrix: A 3D numpy array storing all correlations for each tile across the stack.
    """
    # E.g. best_plane_matrix, all_correlations_matrix = find_best_planes(tiles, stack, tiles_filter)

    ny, nx = tiles.shape[:2]
    best_plane_matrix = np.zeros(shape=(tiles.shape[0], tiles.shape[1], 3), dtype=int)
    if z_range == None:
        all_correlations_matrix = np.zeros(
            (ny, nx, stack.shape[0]))  # Assuming the third dimension of stack is the depth (z)
    else:
        all_correlations_matrix = np.zeros((ny, nx, stack[z_range[0]:z_range[-1]].shape[0]))

    for i in range(ny):
        for j in range(nx):
            if tiles_filter[i, j] == 1:
                # Replace this with the actual function call and its return values
                max_corr, max_position, all_correlations = coarse_plane_detection_in_stack(tiles[i, j], stack, z_range=z_range)

                best_plane_matrix[i, j] = max_position
                all_correlations_matrix[i, j, :] = all_correlations

    return best_plane_matrix, all_correlations_matrix


# +

def plot_image_correlation(tiles, stack, best_plane_matrix, all_correlations_matrix):
    """
    Plots a grid of images, cropped slices, and correlation scatter plots.

    Parameters:
    - tiles: 4D numpy array of images.
    - stack: 3D numpy array representing an interpolated stack of images.
    - best_plane_matrix: 2D numpy array with the best plane indices for each tile.
    - all_correlations_matrix: 3D numpy array containing correlation data for scatter plots.
    """
    # Define grid shape
    num_rows, num_cols = tiles.shape[:2]  # Assumes tiles is a 4D array with shape (num_rows, num_cols, height, width)

    # Create figure and gridspec
    fig = plt.figure(figsize=(14, 12))
    gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
    gs_sub1 = GridSpecFromSubplotSpec(nrows=num_rows * 3, ncols=num_cols, subplot_spec=gs_main[0], hspace=0.5)

    # Compute global Y-axis limits for scatter plots
    global_y_min, global_y_max = np.inf, -np.inf
    for row in range(num_rows):
        for col in range(num_cols):
            local_min = all_correlations_matrix[row, col].min()
            local_max = all_correlations_matrix[row, col].max()
            global_y_min = min(global_y_min, local_min)
            global_y_max = max(global_y_max, local_max)

    # Populate the grid with plots
    for row in range(num_rows):
        for col in range(num_cols):
            ax = fig.add_subplot(gs_sub1[3 * row, col])
            ax.imshow(tiles[row, col], cmap="gray")
            ax.set_title(f'Image ({row},{col})')
            ax.set_xticklabels([])

            ax = fig.add_subplot(gs_sub1[3 * row + 1, col])
            cropped_slice = crop_stack_to_matched_plane(stack, tiles[row, col],
                                                        tuple(best_plane_matrix[row, col].astype(int)))
            ax.imshow(cropped_slice, cmap="gray")
            ax.set_title(f'Slice {best_plane_matrix[row, col][0] - stack.shape[0] // 2}')
            ax.set_xticklabels([])

            ax = fig.add_subplot(gs_sub1[3 * row + 2, col])
            ax.scatter(range(all_correlations_matrix.shape[2]), all_correlations_matrix[row, col])
            ax.set_ylim(global_y_min, global_y_max)
            ax.set_title(f'Correlation ({row},{col})')
            ax.set_xticklabels([])
            if col > 0:
                ax.set_yticklabels([])

    plt.tight_layout()
    plt.show()


# Creating meshgrid
def warp_stack_to_plane(stack, plane, transformation, thickness):
    """
    Warp a stack of images to a specified plane using a given transformation.

    Parameters:
    - stack: 3D numpy array representing the stack of images.
    - plane: 2D numpy array representing the target plane.
    - transformation: Transformation function to apply to warp the stack to the plane.
    - thickness: Thickness of the warped stack.

    Returns:
    - interpolated_stack: Interpolated stack warped to the specified plane.
    """
    # Calculate half thickness
    t_2 = thickness // 2

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


def slice_into_uniform_tiles(image, nx, ny, plot=True):
    """
    Slice an image into uniformly sized tiles.

    Parameters:
    - image: 2D NumPy array representing the image to be sliced.
    - nx: Number of tiles along the x-axis (width).
    - ny: Number of tiles along the y-axis (height).
    - plot: Boolean if the tiles should be ploted.

    Returns:
    - A 2D NumPy array with desired shape, each space representing a uniformly sized tile.
    """
    height, width = image.shape
    print(f"Original image size: {height}x{width}")

    # Calculate the size of each tile
    M, N = (height // ny, width // nx)
    print(f"Tile size: {M}x{N}")

    # Adjust the height and width to ensure all tiles are of equal size
    adjusted_height = M * ny
    adjusted_width = N * nx
    print(f"Adjusted image size for perfect slicing: {adjusted_height}x{adjusted_width}")

    # Adjust image size for slicing
    adjusted_image = image[:adjusted_height, :adjusted_width]

    # Generate perfectly sized tiles
    tiles = np.array(
        [adjusted_image[x:x + M, y:y + N] for x in range(0, adjusted_height, M) for y in range(0, adjusted_width, N)])

    print(f"Number of tiles: {len(tiles)}")

    reshaped_tiles = tiles.reshape(ny, nx, tiles.shape[1], tiles.shape[2])

    if plot:
        fig, axs = plt.subplots(ncols=nx, nrows=ny)

        count = 0
        for i in range(ny):
            for j in range(nx):
                axs[i, j].imshow(reshaped_tiles[i, j], cmap='gray')
                count += 1

        return reshaped_tiles, (M, N), (adjusted_height, adjusted_width), fig, axs

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




def plane_detection_with_iterative_alignment(plane, stack, equalize=True, binning=True, plot=True, nx=2, ny=3,
                                             tiles_filter=None, thickness_values=None):
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
    plt.show()

    return current_tform, all_transformation_matrices


def calculate_manders_coefficient_3d(mask_stack, channel_stack):
    # Label the mask stack
    labels = label(mask_stack)
    props = regionprops(labels)

    # Initiate Manders' coefficients dictionary and Manders' coefficient based colored stack
    manders_results = {}
    mask_colored_stack = np.zeros_like(anatomy_mask, dtype=float)

    # Binarize channel stack
    channel_threshold = threshold_otsu(channel_stack)
    binary_channel_stack = channel_stack > channel_threshold

    # Binarize the mask (assuming it's already binary)
    binary_mask_stack = anatomy_mask > 0

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