# Imports
import tifffile
import numpy as np
from skimage.transform import SimilarityTransform
from skimage.exposure import match_histograms, equalize_adapthist, rescale_intensity
import os
import matplotlib.pyplot as plt

# Assuming these are custom modules. Make sure they're in the correct path.
from scripts.config_model import save_experiment_config, tree
from scripts.sample_db import SampleDB
from scripts.utils.image_utils import (
    load_tiff_as_hyperstack, save_array_as_hyperstack_tiff, bin_image,
    coarse_plane_detection_in_stack, slice_into_uniform_tiles,
    fine_plane_detection_in_stack_by_tiling, plot_matched_plane_and_cropped_slice,
    plot_image_correlation, warp_stack_to_plane
)

plt.set_cmap('binary')


# Step 1: Load the sample database
def load_sample_database():
    db_path = r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\sample_db.csv'
    sample_db = SampleDB()
    sample_db.load(db_path)
    print(sample_db)
    return sample_db


# Step 2: Load experiment configuration
def load_experiment_config(sample_db, sample_id):
    exp = sample_db.get_sample(sample_id)
    return exp


# Step 3: Set up experiment parameters
def setup_experiment_parameters(exp):
    sample = exp.sample
    root_path = exp.paths.root_path
    trials_path = exp.paths.trials_path
    anatomy_path = exp.paths.anatomy_path
    em_path = exp.paths.em_path
    n_planes = exp.params_lm.n_planes
    n_frames = exp.params_lm.n_frames
    n_slices = exp.params_lm.lm_stack_range
    doubling = 2 if exp.params_lm.doubling else 1

    n_frames_trial = n_frames // n_planes
    exp.params_lm["n_frames_trial"] = n_frames_trial

    raw_trial_paths = os.listdir(os.path.join(trials_path, "raw"))
    n_trials = len(raw_trial_paths)
    exp.params_lm["n_trials"] = n_trials

    ignore_until_frame = exp.params_lm.shutter_delay_frames

    processed_folder = os.path.join(trials_path, "processed")
    os.makedirs(processed_folder, exist_ok=True)

    ref_images_path = os.path.join(processed_folder, f"sum_raw_trials_{sample.id}.tif")

    return exp, ref_images_path


# Step 4: Load planes and stack
def load_images(ref_images_path, anatomy_path):
    lm_planes = tifffile.imread(ref_images_path)
    print(f"LM planes shape: {lm_planes.shape}")

    lm_stack = \
    load_tiff_as_hyperstack(os.path.join(anatomy_path, "raw", "20220426_RM0008_130hpf_fP1_f3_anatomyGFRF_001_.tif"),
                            n_channels=2)[0]
    print(f"LM stack shape: {lm_stack.shape}")

    return lm_planes, lm_stack


# Step 5: Plane detection function
def plane_detection_with_iterative_alignment(plane, stack, equalize=True, binning=True, plot=True, nx=2, ny=3,
                                             tiles_filter=None, thickness_values=None):


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
        thickness_values = [100, 50, 50, 30, 30, 20]

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

    last_matched_slice = \
    warp_stack_to_plane(stack, plane, SimilarityTransform(current_tform), thickness_values[-1])[
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


# Step 6: Process planes
def process_planes(lm_planes, lm_stack, trials_path):
    for ii, lm_plane in enumerate(lm_planes[:, -1]):
        print(f"Processing Plane {ii}")
        final_transform, all_transformation_matrices = plane_detection_with_iterative_alignment(
            lm_plane, lm_stack, equalize=True, binning=True, plot=True, nx=2, ny=3, tiles_filter=None,
            thickness_values=None
        )
        np.save(os.path.join(trials_path, f"fine_aligned_tform_plane_{ii}.npy"), final_transform)

    return all_transformation_matrices


# Main execution
if __name__ == "__main__":
    sample_db = load_sample_database()
    sample_id = '20220426_RM0008_130hpf_fP1_f3'
    exp = load_experiment_config(sample_db, sample_id)
    exp, ref_images_path = setup_experiment_parameters(exp)
    lm_planes, lm_stack = load_images(ref_images_path, exp.paths.anatomy_path)
    all_transformation_matrices = process_planes(lm_planes, lm_stack, exp.paths.trials_path)
    print(all_transformation_matrices)