import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from skimage import feature
from skimage.exposure import match_histograms, equalize_adapthist
from image_processing import extend_stack, bin_2x2_grayscale
from scipy.interpolate import griddata

def find_plane_in_stack(plane, stack, reshape=True, plot_all_correlations=False, z_range=None):
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

    slices = [*range(stack.shape[0])] if z_range == None else [*range(z_range[0],z_range[-1])]

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

def crop_stack_to_matched_plane(stack, plane, position, blank=100):
    """
    Crop a matched slice from the stack based on the position of the matched plane.
    
    Parameters:
    - stack: 3D numpy array representing the stack of images.
    - plane: 2D numpy array representing the plane.
    - position: Tuple (slice index, x_offset, y_offset) describing the position of the matched plane.
    - blank: Number of blank pixels to add around the cropped slice.

    Returns:
    - cropped_slice: Cropped slice from the stack.
    """
    slice_idx, x_offset, y_offset = position
    extended_stack = extend_stack(stack, blank)

    # Calculate the bounds for cropping the matched slice
    width, height = plane.shape
    y_start, y_end = y_offset - height//2 + blank, y_offset + height//2 + blank
    x_start, x_end = x_offset - width//2 + blank, x_offset + width//2 + blank

    cropped_slice = extended_stack[slice_idx, x_start:x_end, y_start:y_end]
    
    return cropped_slice

def plot_matched_plane_and_cropped_slice(stack, plane, position, match_hist=True):
    """
    Visualize the matched plane from the stack and a cropped slice from the rotated stack.

    Parameters:
    - stack: 3D image stack.
    - plane: 2D plane to be visualized.
    - position: (slice index, x_offset, y_offset) - describes where in the rotated stack the match was found.
    - match_hist: Boolean indicating whether to plot the match histograms.

    Returns:
    - Displays a side-by-side visualization of the plane and the matched slice.
    """
    cropped_slice = crop_stack_to_matched_plane(stack, plane, position)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(plane, cmap='gray')
    ax1.set_title(f"Plane")
    if match_hist:
        ax2.imshow(match_histograms(cropped_slice, plane), cmap='gray')  # match_histograms is not defined in provided code
    else:
        ax2.imshow(cropped_slice, cmap='gray')
    ax2.set_title(f"Matched cropped slice ")
    plt.show()

def find_best_planes(tiles, stack, tiles_filter, z_range=None):
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
    best_plane_matrix = np.zeros(shape=(tiles.shape[0],tiles.shape[1],3), dtype=int)
    if z_range==None:
        all_correlations_matrix = np.zeros((ny, nx, stack.shape[0]))  # Assuming the third dimension of stack is the depth (z)
    else:
        all_correlations_matrix = np.zeros((ny, nx, stack[z_range[0]:z_range[-1]].shape[0]))

    for i in range(ny):
        for j in range(nx):
            if tiles_filter[i, j] == 1:
                max_corr, max_position, all_correlations = find_plane_in_stack(tiles[i, j], stack, z_range=z_range)
                
                best_plane_matrix[i, j] = max_position
                all_correlations_matrix[i, j, :] = all_correlations

    return best_plane_matrix, all_correlations_matrix



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
            ax = fig.add_subplot(gs_sub1[3*row, col])
            ax.imshow(tiles[row, col])
            ax.set_title(f'Image ({row},{col})')
            ax.set_xticklabels([])

            ax = fig.add_subplot(gs_sub1[3*row+1, col])
            cropped_slice = crop_stack_to_matched_plane(stack, tiles[row, col], tuple(best_plane_matrix[row, col].astype(int)))
            ax.imshow(cropped_slice)
            ax.set_title(f'Slice {best_plane_matrix[row, col][0]-stack.shape[0]//2}')
            ax.set_xticklabels([])

            ax = fig.add_subplot(gs_sub1[3*row+2, col])
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
        np.linspace(-t_2, t_2+1, num=int(thickness*transformation.scale) + 1, endpoint=False), 
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


