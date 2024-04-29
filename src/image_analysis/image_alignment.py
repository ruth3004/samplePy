import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage import feature

class ImageAlignment:
    def __init__(self):
        # Initialize any necessary properties here
        pass

    def find_plane_in_stack(self, plane, stack, plot_all_correlations=False, z_range=None):
        """
        Correlates a given plane with each slice in a 3D stack and identifies the slice with the highest correlation.

        Parameters:
            plane (numpy.ndarray): 2D array representing the plane to find in the stack.
            stack (numpy.ndarray): 3D stack of images to search.
            plot_all_correlations (bool): If True, plots a scatter plot of correlation values across slices.
            z_range (tuple or None): A tuple (start, end) defining the range of slices to consider. If None, all slices are considered.

        Returns:
            tuple: A tuple (max_corr, max_position) where 'max_corr' is the highest correlation value and 'max_position'
                   is the position (slice index, x, y) of the best match in the stack.
        """
        max_corr = -np.inf
        max_position = None
        correlations = []

        if z_range is None:
            z_range = (0, stack.shape[0])

        for i in range(*z_range):
            corr = feature.match_template(stack[i], plane, pad_input=True)
            corr_value = np.max(corr)
            if corr_value > max_corr:
                max_corr = corr_value
                max_position = (i,) + np.unravel_index(np.argmax(corr), corr.shape)
            correlations.append(corr_value)

        if plot_all_correlations:
            plt.scatter(range(*z_range), correlations)
            plt.xlabel('Slice Index')
            plt.ylabel('Correlation')
            plt.title('Correlation by Slice')
            plt.show()

        return max_corr, max_position

    def crop_stack_to_matched_plane(self, stack, plane, position, blank=100):
        """
        Crops a specific area from the stack around a given position, adjusted by a specified margin.

        Parameters:
            stack (numpy.ndarray): 3D stack of images.
            plane (numpy.ndarray): The 2D reference plane used for matching.
            position (tuple): The position (slice index, x, y) where the plane was matched in the stack.
            blank (int): Margin size to add around the cropped area for context.

        Returns:
            numpy.ndarray: Cropped area from the stack.
        """
        slice_idx, x_offset, y_offset = position
        slice_shape = stack.shape[1:]

        # Determine the crop boundaries, ensuring they stay within the stack bounds
        x_min = max(x_offset - plane.shape[1] // 2 - blank, 0)
        x_max = min(x_offset + plane.shape[1] // 2 + blank, slice_shape[1])
        y_min = max(y_offset - plane.shape[0] // 2 - blank, 0)
        y_max = min(y_offset + plane.shape[0] // 2 + blank, slice_shape[0])

        return stack[slice_idx, y_min:y_max, x_min:x_max]

    def plot_matched_plane_and_cropped_slice(self, stack, plane, position, match_hist=True):

        """
        Displays the matched plane and the corresponding cropped slice from the stack side-by-side.

        Parameters:
            stack (numpy.ndarray): 3D stack from which the slice is cropped.
            plane (numpy.ndarray): 2D plane that was matched in the stack.
            position (tuple): The position (slice index, x, y) where the plane was matched.

        Returns:
            None: Plots the images using matplotlib.
        """
        cropped_slice = crop_stack_to_matched_plane(stack, plane, position)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(plane, cmap='gray')
        axes[0].set_title('Reference Plane')
        axes[1].imshow(cropped_slice, cmap='gray')
        axes[1].set_title('Matched Cropped Slice')
        plt.show()

    def find_best_planes(self, tiles, stack, tiles_filter, z_range=None):
        """
        Identifies the best matching plane in the stack for each tile, considering only specified tiles.

        Parameters:
            tiles (numpy.ndarray): 4D array containing multiple 2D tiles (ny, nx, tile_height, tile_width).
            stack (numpy.ndarray): 3D stack of images to be searched.
            tiles_filter (numpy.ndarray): 2D boolean array (ny, nx) indicating which tiles to process.
            z_range (tuple or None): Tuple specifying the range of slices to search in the stack; if None, searches the entire stack.

        Returns:
            tuple: A tuple containing:
                   - best_plane_matrix (numpy.ndarray): 2D array with the position of the best match for each tile.
                   - all_correlations_matrix (numpy.ndarray): 3D array storing all correlation values for each tile.
        """
        ny, nx = tiles.shape[:2]
        best_plane_matrix = np.zeros((ny, nx, 3), dtype=int)
        all_correlations_matrix = np.zeros((ny, nx, stack.shape[0] if z_range is None else z_range[1] - z_range[0]))

        for i in range(ny):
            for j in range(nx):
                if tiles_filter[i, j]:
                    max_corr, max_position = find_plane_in_stack(tiles[i, j], stack, z_range=z_range)
                    best_plane_matrix[i, j] = max_position
                    all_correlations_matrix[i, j, :] = max_corr

        return best_plane_matrix, all_correlations_matrix
    def plot_image_correlation(self, tiles, stack, best_plane_matrix, all_correlations_matrix):"""
    Plots a comprehensive grid of the original tiles, the matched slices from the stack, and their correlation values.

    Parameters:
        tiles (numpy.ndarray): 4D array of tiles.
        stack (numpy.ndarray): 3D stack of images.
        best_plane_matrix (numpy.ndarray): Matrix containing the best matching slice indices for each tile.
        all_correlations_matrix (numpy.ndarray): Matrix of correlation values for each tile across the stack.

    Returns:
        None: Displays a matplotlib plot of the tiles, matched slices, and a scatter plot of correlation values.
    """
    num_rows, num_cols = tiles.shape[:2]
    fig = plt.figure(figsize=(15, 5 * num_rows))
    gs = GridSpec(num_rows, 3 * num_cols, fig)

    for i in range(num_rows):
        for j in range(num_cols):
            ax_tile = fig.add_subplot(gs[i, 3 * j])
            ax_tile.imshow(tiles[i, j], cmap='gray')
            ax_tile.set_title(f'Tile {i}-{j}')
            ax_tile.axis('off')

            ax_slice = fig.add_subplot(gs[i, 3 * j + 1])
            slice_idx = best_plane_matrix[i, j][0]
            ax_slice.imshow(stack[slice_idx], cmap='gray')
            ax_slice.set_title(f'Matched Slice {slice_idx}')
            ax_slice.axis('off')

            ax_corr = fig.add_subplot(gs[i, 3 * j + 2])
            ax_corr.plot(all_correlations_matrix[i, j])
            ax_corr.set_title(f'Correlation Plot {i}-{j}')
            ax_corr.set_xlabel('Slice Index')
            ax_corr.set_ylabel('Correlation')

    plt.tight_layout()
    plt.show()

# Additional helper functions or classes can also be defined here
