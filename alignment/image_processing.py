import numpy as np
import matplotlib.pyplot as plt

def bin_2x2_grayscale(image):
    ''' 
    Bin image in xy 2 by 2. 
    
    This function takes an image and bins it into 2x2 blocks, 
    averaging the pixel values within each block to reduce the resolution.
    
    Parameters:
    - image: 2D NumPy array representing the grayscale image to be processed.
    
    Returns:
    - A 2D NumPy array representing the binned image.
    '''
    # Ensure the height and width are divisible by 2
    h, w = image.shape
    h = h // 2 * 2
    w = w // 2 * 2
    image = image[:h, :w]

    # Reshape and calculate the mean along the new axes
    binned_image = image.reshape(h//2, 2, w//2, 2).mean(axis=(1, 3))
    return binned_image

def slice_into_uniform_tiles(image, nx, ny, plot=True):
    """
    Slice an image into uniformly sized tiles.
    
    Parameters:
    - image: 2D NumPy array representing the image to be sliced.
    - nx: Number of tiles along the x-axis (width).
    - ny: Number of tiles along the y-axis (height).
    - plot: Boolean if the tiles should be plotted.
    
    Returns:
    - A 4D NumPy array representing the sliced image tiles.
    - Tuple (M, N) representing the size of each tile.
    - Tuple (adjusted_height, adjusted_width) representing the adjusted size of the image for perfect slicing.
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
    tiles = np.array([adjusted_image[x:x+M, y:y+N] for x in range(0, adjusted_height, M) for y in range(0, adjusted_width, N)])
    
    print(f"Number of tiles: {len(tiles)}")

    # Reshape tiles into 4D array
    reshaped_tiles = tiles.reshape(ny, nx, tiles.shape[1], tiles.shape[2])

    if plot:
        # Plot the tiles
        fig, axs = plt.subplots(ncols= nx, nrows= ny)
    
        count = 0
        for i in range(ny):
            for j in range(nx):
                axs[i,j].imshow(reshaped_tiles[i,j])
                count += 1
        plt.show()
    
    return reshaped_tiles, (M, N), (adjusted_height, adjusted_width)

def extend_stack(stack, blank=100):
    """
    Extend the size of a stack by adding blank columns to its sides.
    
    Parameters:
    - stack: 3D NumPy array representing the stack of images.
    - blank: Number of blank columns to add to each side.
    
    Returns:
    - A 3D NumPy array representing the extended stack of images.
    """
    extended_stack = np.zeros(shape=(stack.shape[0], stack.shape[1]+2*blank, stack.shape[2]+2*blank)) 
    extended_stack[:, blank:-blank, blank:-blank] = stack
    return extended_stack
