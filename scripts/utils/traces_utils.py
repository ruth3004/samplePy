import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
import h5py
from tifffile import imread


# Function to process and save data
def extract_fluorescence_data(hdf5_file_path, sample_id, trial_names, processed_folder, masks_stack, n_planes,
                              doubling):
    with h5py.File(hdf5_file_path, 'w') as f:
        exp_grp = f.create_group(sample_id)

        all_traces, all_labels, all_planes, all_trials, all_odors, all_centroids = [], [], [], [], [], []

        for trial_idx, trial_path in enumerate(trial_names):
            movie_path = os.path.join(processed_folder, f"motion_corrected_{trial_path}")
            movie = imread(movie_path)
            print(f"Processing trial {trial_idx + 1}/{len(trial_names)}, shape: {movie.shape}")

            trial_info = trial_path.split('_')
            trial_num = trial_info[5][1:]
            odor_full = trial_info[6][2:]
            odor = odor_full[2:] if odor_full.startswith('o') else odor_full

            for plane in range(n_planes * doubling):
                plane_movie = movie[plane]
                mask = masks_stack[plane, trial_idx, :, :]

                for label in np.unique(mask):
                    if label != 0:
                        label_mask = mask == label
                        fluorescence_values = plane_movie[:, label_mask].mean(axis=1)

                        all_traces.append(fluorescence_values)
                        all_labels.append(label)
                        all_planes.append(plane)
                        all_trials.append(trial_num)
                        all_odors.append(odor)

                        y, x = np.where(label_mask)
                        centroid = (np.mean(y), np.mean(x))
                        all_centroids.append(centroid)

        # Save data in HDF5 file

        exp_grp.create_dataset('raw_traces', data=np.array(all_traces))
        exp_grp.create_dataset('lm_plane_labels', data=np.array(all_labels))
        exp_grp.create_dataset('plane_nr', data=np.array(all_planes))
        exp_grp.create_dataset('trial_nr', data=np.array(all_trials, dtype='S'))
        exp_grp.create_dataset('odor', data=np.array(all_odors, dtype='S'))
        exp_grp.create_dataset('lm_plane_centroids', data=np.array(all_centroids))

        # Create a mapping group
        mapping_grp = exp_grp.create_group('cell_mapping')
        mapping_grp.create_dataset('neuron_ids',
                                   data=np.array([f'n{i}' for i in range(1, len(all_labels) + 1)], dtype='S'))
        mapping_grp.create_dataset('lm_plane_labels', data=np.array(all_labels))
        mapping_grp.create_dataset('plane_nr', data=np.array(all_planes))

    print("Fluorescence intensities calculated and saved in HDF5 file.")

 # Plot extracted raw traces
def load_hdf5_data(hdf5_file_path, sample_id):
    with h5py.File(hdf5_file_path, 'r') as f:
        exp_grp = f[sample_id]
        return {key: exp_grp[key][()] for key in exp_grp.keys() if isinstance(exp_grp[key], h5py.Dataset)}

def calculate_dff(traces, baseline_frames=[50,100]):
    f0 = np.mean(traces[:, baseline_frames[0]:baseline_frames[1]], axis=1)
    dff = (traces - f0[:, np.newaxis]) / f0[:, np.newaxis]
    return dff

def plot_traces(traces, labels, title, ylabel, n_examples=3):
    plt.figure(figsize=(10, 6))
    for i in range(n_examples):
        plt.plot(traces[i], label=f'Label {labels[i]}')
    plt.title(title)
    plt.xlabel('Time (frames)')
    plt.ylabel(ylabel)
    plt.legend()

def plot_average_traces_by_group(traces, groups, title, ylabel):
    unique_groups = np.unique(groups)
    plt.figure(figsize=(10, 6))
    for group in unique_groups:
        plt.plot(traces[groups == group].mean(axis=0), label=group.decode('utf-8'))
    plt.legend()
    plt.title(title)
    plt.xlabel('Time (frames)')
    plt.ylabel(ylabel)
    plt.show()

def load_traces(traces_file_path):
    """
    Load fluorescence traces from a .npy file.

    Parameters:
    traces_file_path (str): Path to the .npy file containing fluorescence traces.
    as_df (bool): Return the traces as a DataFrame if True, otherwise as a dictionary.

    Returns:
    dict or pd.DataFrame: Fluorescence traces.
    """
    traces_dict = {id: trace for id, trace in np.load(traces_file_path, allow_pickle=True).item().items()}
    traces = pd.DataFrame.from_dict(traces_dict, orient='columns')
    return traces


def plot_traces_with_mean(traces):
    """
    Plot fluorescence traces with their mean.

    Parameters:
    traces (np.ndarray or pd.DataFrame): The fluorescence traces.
    """
    plt.figure(figsize=(10, 5))
    for trace in traces:
        plt.plot(trace, alpha=0.5)

    mean_traces = np.mean(traces, axis=0)
    plt.plot(mean_traces, color='black', linewidth=2, label='Mean Trace')
    plt.legend()
    plt.title('Fluorescence Traces')
    plt.xlabel('Frame')
    plt.ylabel('Fluorescence Intensity')
    plt.show()


def calculate_df_traces(raw_traces, baseline_frames, shutter_off_frames) -> pd.DataFrame:
    """
    Calculate ΔF/F traces.

    Parameters:
    raw_traces (pd.DataFrame): The raw fluorescence intensity DataFrame.
    baseline_frames (List[int]): The frames used to calculate the baseline.
    motor_frames (List[int]): The frames used to calculate the background noise.

    Returns:
    pd.DataFrame: The ΔF/F traces.
    """
    dig_error = raw_traces.iloc[shutter_off_frames].mean(axis=0)
    print(dig_error.shape)
    baseline = raw_traces.iloc[baseline_frames].mean(axis=0)

    print(baseline.shape)
    df_traces = (raw_traces - baseline) / (baseline - dig_error)
    return df_traces


def load_all_traces_for_plane(traces_folder: str, plane_str: str) -> List[pd.DataFrame]:
    """
    Load all traces for a specific plane from all trials.

    Parameters:
    traces_folder (str): Path to the folder containing fluorescence data.
    plane_str (str): String identifier for the plane.

    Returns:
    List[pd.DataFrame]: List of DataFrames containing fluorescence traces for each trial.
    """
    all_traces = []
    for file in os.listdir(traces_folder):
        if file.endswith(".npy") and f"traces_plane_{plane_str}" in file:
            traces_df = load_traces(os.path.join(traces_folder, file))
            all_traces.append(traces_df)
    return all_traces


def find_common_rois(all_traces: List[pd.DataFrame]) -> List[str]:
    """
    Find ROIs that are common across all trials.

    Parameters:
    all_traces (List[pd.DataFrame]): List of DataFrames containing fluorescence traces for each trial.

    Returns:
    List[str]: List of common ROI identifiers.
    """
    common_rois = set(all_traces[0].columns)
    for traces_df in all_traces[1:]:
        common_rois.intersection_update(traces_df.columns)
    return list(common_rois)
