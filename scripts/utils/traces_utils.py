import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

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