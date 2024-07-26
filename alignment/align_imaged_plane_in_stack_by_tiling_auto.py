
# +
import numpy as np
import tifffile

#Image processing
from skimage import transform, measure, feature
from skimage.exposure import match_histograms, equalize_adapthist
from skimage.transform import SimilarityTransform, EuclideanTransform
from skimage.segmentation import expand_labels # expand EM warped labels
from skimage.filters import difference_of_gaussians

from warp_stack_to_plane import warp_stack_to_plane

#Visualization
import matplotlib.pyplot as plt
import napari
from scipy import ndimage
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec



# +
#Utils functions (image processing)

def bin_2x2_grayscale(image):
    ''' Bin image in xy 2 by 2'''
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
    tiles = np.array([adjusted_image[x:x+M, y:y+N] for x in range(0, adjusted_height, M) for y in range(0, adjusted_width, N)])
    
    print(f"Number of tiles: {len(tiles)}")

    reshaped_tiles = tiles.reshape(ny,nx, tiles.shape[1],tiles.shape[2])

    if plot:
        fig, axs = plt.subplots(ncols= nx, nrows= ny)
    
        count=0
        for i in range(ny):
            for j in range(nx):
                axs[i,j].imshow(reshaped_tiles[i,j])
                #axs[i,j].imshow(reshaped_tiles[i,j].astype(np.uint16))
                count+=1
        plt.show()
    
    return reshaped_tiles, (M,N), (adjusted_height, adjusted_width)

# Example usage:
# Assuming `plane` is your 2D numpy array representing the image
# plane = np.random.rand(100, 100) # Example initialization, replace with your actual image
# nx, ny = (3, 3)  # Desired number of tiles in each dimension
# tiles = slice_into_uniform_tiles(plane, nx, ny)
# This will print out the details and return the list of tiles.

# adding columns to stacks
def extend_stack(stack, blank=100):
    extended_stack = np.zeros(shape=(stack.shape[0],stack.shape[1]+2*blank,stack.shape[2]+2*blank)) 
    extended_stack[:,blank:-blank,blank:-blank] = stack#.astype(np.uint16)
    return extended_stack


# +
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
    
    slice_idx, x_offset, y_offset = position
    extended_stack = extend_stack(stack, blank)

    # Calculate the bounds for cropping the matched slice
    width, height = plane.shape
    y_start, y_end = y_offset - height//2+blank, y_offset + height//2+blank
    x_start, x_end = x_offset - width//2+blank , x_offset + width//2+blank

    cropped_slice = extended_stack[slice_idx, x_start:x_end, y_start:y_end]
    
    return cropped_slice

def plot_matched_plane_and_cropped_slice(stack, plane, position, match_hist =True):
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
                # Replace this with the actual function call and its return values
                max_corr, max_position, all_correlations = find_plane_in_stack(tiles[i, j], stack, z_range=z_range)
                
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


# +
## Loading data

#load lm stack image
lm_stack  = tifffile.imread(r"\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\CLEM_Analyses\CLEM_20220426_RM0008_130hpf_fP1_f3\pycpd\lm_stack.tif")

#load lm plane image
lm_plane = np.flip(tifffile.imread(r"\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\2P_RawData\2022-04-26\f3\results\anatomy\plane01\anatomy_binned_x1y1z1_20220426_RM0008_130hpf_fP1_f3_t1_o1Ala_001_.tif"),axis=1)
lm_plane= lm_plane.astype(np.uint16)[lm_plane.shape[0]//2:]

#load lm plane mask 
lm_plane_mask = np.flip(tifffile.imread(r"\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\2P_RawData\2022-04-26\f3\results\anatomy\plane01\anatomy_binned_x1y1z1_20220426_RM0008_130hpf_fP1_f3_t1_o1Ala_001__cp_masks.tif"),axis=1)
lm_plane_mask = lm_plane_mask[lm_plane_mask.shape[0]//2:] 

# Extract centroids of lm plane (can be done in a different script)
lm_plane_props = measure.regionprops(lm_plane_mask)
lm_plane_centroids = np.array([prop.centroid for prop in lm_plane_props])

# adding raw and transformed centroids from lm plane 
print(lm_plane_centroids.shape)
lm_plane_centroids_3d = np.hstack((np.zeros(shape=(lm_plane_centroids.shape[0],1)), lm_plane_centroids))
print(lm_plane_centroids_3d.shape)

# Binning images
binned_lm_stack = np.array([bin_2x2_grayscale(slice) for slice in lm_stack])
binned_lm_plane = bin_2x2_grayscale(lm_plane)

# +
#load em warped (FOV of LM) stack
em_warped_stack = tifffile.imread(r"\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\CLEM_Analyses\CLEM_20220426_RM0008_130hpf_fP1_f3\pycpd\em_stack_04_warped_fovLM.tif")

#load em warped mask
em_warped_mask = tifffile.imread(r"\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\CLEM_Analyses\CLEM_20220426_RM0008_130hpf_fP1_f3\pycpd\em_stack_04_warped_fovLM_cp_masks.tif")
expanded_em_labels = expand_labels(em_warped_mask, distance=3)

# Extract centroids of lm plane (can be done in a different script)
lm_stack_props = measure.regionprops(em_warped_mask)
lm_stack_centroids = np.array([prop.centroid for prop in lm_stack_props])
# -

all_transformation_matrices = [np.eye(4)]
all_transformation_matrices

equalized_lm_plane =equalize_adapthist(lm_plane, clip_limit=0.03)
equalized_lm_stack = np.array([equalize_adapthist(slice, clip_limit=0.03) for slice in lm_stack])
fig, axs = plt.subplots(ncols=6)
axs[0].imshow(lm_plane)
axs[1].imshow(equalized_lm_plane)
axs[2].imshow(lm_stack[20])
axs[3].imshow(equalized_lm_stack[20])
axs[4].imshow(lm_stack[120])
axs[5].imshow(equalized_lm_stack[120])

# +
gaussian_equalized_lm_plane = difference_of_gaussians(equalized_lm_plane, low_sigma=1, high_sigma =1000)
plt.imshow(gaussian_equalized_lm_plane)

#gaussian_equalized_lm_stack = np.array([difference_of_gaussians(slice, low_sigma=1, high_sigma =1000) for slice in equalized_lm_stack])

fig, axs = plt.subplots(ncols=2)

axs[0].imshow(equalized_lm_plane[50:150,50:150])
axs[1].imshow(gaussian_equalized_lm_plane[50:150,50:150])


# +
# %%time
# find coarse slice to plane  
binned_plane = bin_2x2_grayscale(gaussian_equalized_lm_plane)
binned_stack = np.array([bin_2x2_grayscale(slice) for slice in equalized_lm_stack])
max_corr_coarse, max_position_coarse, all_correlations_coarse = find_plane_in_stack(binned_plane, binned_stack, plot_all_correlations= True)


# -

stack = equalized_lm_stack
plane = gaussian_equalized_lm_plane

# +
print(max_position_coarse[0])
z_range = 50
half_z_range = z_range // 2
# Ensure min_z_range is not less than 0 and max_z_range does not exceed stack.shape[0]
min_z_range = max(0, max_position_coarse[0] - half_z_range)
max_z_range = min(stack.shape[0], max_position_coarse[0] + half_z_range)

z_crop = range(min_z_range, max_z_range)

print(z_crop[0], z_crop[-1])
# -

#Define stack and planes to match 
nx, ny = (4, 4)
tiles, tile_size, adj_image_size = slice_into_uniform_tiles(plane, nx, ny, plot=True)
tiles_filter = np.array([[1, 1, 1, 1],
                         [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]])

# %%time
best_plane_matrix, all_correlations_matrix = find_best_planes(tiles, stack, tiles_filter, z_range=[z_crop[0], z_crop[-1]])

# %%time
plot_image_correlation(tiles, stack, best_plane_matrix, all_correlations_matrix)

filter_matrix = np.zeros(shape=(nx, ny),dtype=int)
filter_matrix

# +
filter_matrix= np.array([[1, 1, 1, 1],
                      [0, 1, 1, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
points_filter = np.where(filter_matrix.flatten()==1)
print(points_filter)

# Create registration model  
# lm alignment points
lm_stack_points = np.array([best_plane_matrix[i,j] for i in range(nx) for j in range(ny) ])
print(lm_stack_points)


x = np.arange(tile_size[1]//2, adj_image_size[1], tile_size[1])
y = np.arange(tile_size[0]//2, adj_image_size[0], tile_size[0])
xv, yv = np.meshgrid(x, y)

lm_plane_points = np.array([(0, yv[i,j], xv[i,j])
                            for i in range(ny) for j in range(nx)])

source = lm_plane_points[points_filter]
target = lm_stack_points[points_filter]

# Calculate the transformation
tform = SimilarityTransform()
tform.estimate(source, target)

# Apply the transformation to a new set of points
transformed_source = tform(source)
print(f"transformed_source",transformed_source)
#print("Transformation matrix:")
#print(tform.params)

print(target-transformed_source)
plt.imshow(target-transformed_source, cmap="bwr", vmin = -5, vmax=5)
plt.colorbar()

# Apply transformation to lm plane centroids
transformed_lm_plane_centroids = tform(lm_plane_centroids_3d)

# Find overlay of em warped masks and transformed_lm_plane_centroids
labels_at_coords = ndimage.map_coordinates(
        expanded_em_labels, np.transpose(list(transformed_lm_plane_centroids)), order=0
        )
print(f"Found {np.unique(labels_at_coords[labels_at_coords>0]).shape[0]} from {transformed_lm_plane_centroids.shape[0]} centroids ")

# Filter mask of em warped
filtered_em_mask = np.isin(expanded_em_labels, list(np.unique(labels_at_coords[labels_at_coords>0])))

# -



# +
#unique_labels = np.unique(expanded_em_labels[filtered_em_mask])
#print(len(unique_labels))
#gt_unique_labels = np.unique(expanded_em_labels[gt_filtered_em_mask])
#print(len(gt_unique_labels))
#len(set(unique_labels).intersection(gt_unique_labels))

#from matplotlib_venn import venn2

#venn2(subsets = (len(unique_labels), len(gt_unique_labels), len(set(unique_labels).intersection(gt_unique_labels))), set_labels = ('filtered_em_mask', 'ground truth'))
#plt.show()

# +
# Visualize
viewer = napari.Viewer()

# adding plane with mask and centroids
viewer.add_image(plane)
viewer.add_labels(lm_plane_mask)
viewer.add_points(lm_plane_centroids)

#adding lm stack 
viewer.add_image(lm_stack, opacity=0.5)

# adding lm points
viewer.add_points(source, face_color='r')
viewer.add_points(target, face_color='g')

# adding transformed lm points
viewer.add_points(transformed_source, face_color='b')

viewer.add_points(transformed_lm_plane_centroids, face_color='g', size=5, name="SimilarityTransform dataset")
viewer.add_points(lm_plane_centroids_3d, face_color='m')


# adding em warped mask (extended) and its centroids
viewer.add_labels(expanded_em_labels, opacity=0.5, visible=False)
viewer.add_points(lm_stack_centroids)
# -

# adding filtered em mask against ground truth
#viewer.add_image(gt_filtered_em_mask, colormap = "red", opacity = 0.25 )
viewer.add_image(filtered_em_mask, colormap = "red", opacity = 0.25 )

tform

# +
# Save
name = "plane1_2half"
save_parent_folder = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\CLEM_Analyses\CLEM_20220426_RM0008_130hpf_fP1_f3\pycpd"
np.save(save_parent_folder+"\\tform_"+name+".npy", tform)
np.save(save_parent_folder+"\\labels_at_coords_"+name+".npy", labels_at_coords)
np.save(save_parent_folder+"\\transformed_source_"+name+".npy", transformed_source)



# -

# %%time
# Warping stack to plane an iterating
name = "plane1_2half"
save_parent_folder = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\CLEM_Analyses\CLEM_20220426_RM0008_130hpf_fP1_f3\pycpd"
tform = SimilarityTransform(np.load(save_parent_folder+"\\tform_"+name+".npy"))


# # Iterate and add transformation matrices

# +
all_transformation_matrices.append(tform.params)

all_transformation_matrices
# -

updated_tform = np.linalg.multi_dot(all_transformation_matrices[::-1])
updated_tform

# +
# %%time
thickness=20
interpolated_stack = warp_stack_to_plane(lm_stack, plane, SimilarityTransform(updated_tform), thickness)
equalized_interpolated_stack = np.array([equalize_adapthist(slice, clip_limit=0.03) for slice in interpolated_stack])

best_plane_matrix, all_correlations_matrix = find_best_planes(tiles, equalized_interpolated_stack, tiles_filter)

plot_image_correlation(tiles, equalized_interpolated_stack, best_plane_matrix, all_correlations_matrix)

filter_matrix = np.zeros(shape=(nx, ny))
filter_matrix

# +
#viewer = napari.Viewer()
#viewer.add_image(stack)

# 2 channel image with original plane in 2nd channel in the center z-plane
both = np.zeros((2,) + interpolated_stack.shape, dtype=interpolated_stack.dtype)

both[0] = interpolated_stack
both[1, interpolated_stack.shape[0]//2] = match_histograms(plane, interpolated_stack[-1])

viewer.add_image(both, channel_axis=0, contrast_limits = [0,5000], name= "Original in center")

# +
#Select which tiles to believe! 
filter_matrix = np.array([[1., 1., 1., 1.],
                           [0., 1., 1., 0.],
                           [0., 0., 0., 0.],
                           [0., 0., 0., 0.]])

points_filter = np.where(filter_matrix.flatten()==1)

lm_plane_points = np.array([(interpolated_stack.shape[0]//2, yv[i,j], xv[i,j])
                            for i in range(ny) for j in range(nx)])
lm_stack_points = np.array([best_plane_matrix[i,j] for i in range(nx) for j in range(ny) ])

source = lm_plane_points[points_filter]
target = lm_stack_points[points_filter]

print(f"source: ",source)
print(f"target: ",target)

# Calculate the transformation
tform = SimilarityTransform()
tform.estimate(source, target)

# Apply the transformation to a new set of points
transformed_source = tform(source)

print("Transformation matrix:")
print(tform.params)

print(target-transformed_source)
plt.imshow(target-transformed_source, cmap="bwr", vmin = -5, vmax=5)
plt.colorbar()


