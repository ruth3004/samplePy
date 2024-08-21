# 13_analyze_correlation_lm_traces_batch.py

import os
import sys
import argparse
import logging
from datetime import datetime
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.sample_db import SampleDB
from scripts.utils.traces_utils import load_hdf5_data

# Define the step name as a variable
STEP_NAME = '13_analyze_correlation_lm_traces'


def setup_logging(script_name):
    log_folder = r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\2P_RawData\log'
    os.makedirs(log_folder, exist_ok=True)
    current_date = datetime.now().strftime('%Y%m%d')
    log_file = f"{current_date}_{script_name}.log"
    log_path = os.path.join(log_folder, log_file)
    logging.basicConfig(filename=log_path, level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def average_traces(traces, odors, trials, average_by='odor_trial'):
    unique_combinations = []
    averaged_traces = []
    averaged_labels = []

    if average_by == 'odor':
        unique_combinations = np.unique(odors)
    elif average_by == 'trial':
        unique_combinations = np.unique(trials)
    elif average_by == 'odor_trial':
        unique_combinations = np.unique(list(zip(odors, trials)), axis=0)
    else:
        raise ValueError("average_by must be 'odor', 'trial', or 'odor_trial'")

    for combo in unique_combinations:
        if average_by == 'odor':
            mask = odors == combo
            label = f"{combo}"
        elif average_by == 'trial':
            mask = trials == combo
            label = f"t{combo}"
        else:  # odor_trial
            odor, trial = combo
            mask = (odors == odor) & (trials == trial)
            label = f"{odor}_t{trial}"

        avg_trace = np.mean(traces[mask], axis=0)
        averaged_traces.append(avg_trace)
        averaged_labels.append(label)

    return np.array(averaged_traces), np.array(averaged_labels)


def compute_correlation_matrices(traces, window_size=4):
    num_traces, trace_length = traces.shape
    num_windows = trace_length - window_size + 1
    correlation_matrices = np.zeros((num_windows, num_traces, num_traces))

    for k in range(num_windows):
        window_traces = traces[:, k:k + window_size]
        for i in range(num_traces):
            for j in range(i, num_traces):
                corr, _ = pearsonr(window_traces[i], window_traces[j])
                correlation_matrices[k, i, j] = corr
                correlation_matrices[k, j, i] = corr

    return correlation_matrices


# Modify the plot_correlation_matrix function
def plot_correlation_matrix(correlation_matrix, labels, title, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, annot=False, cbar=True)

    # Add labels
    ax = plt.gca()
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, rotation=90, ha='right')
    ax.set_yticklabels(labels, rotation=0)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def process_sample(sample_id, db_path):
    try:
        # Load the sample database
        sample_db = SampleDB()
        sample_db.load(db_path)

        # Load experiment configuration
        exp = sample_db.get_sample(sample_id)

        # Check if this step has already been completed
        if sample_db.samples[sample_id].get(STEP_NAME) == "True":
            print(f"{STEP_NAME} already completed for sample {sample_id}. Skipping.")
            return

        # Load data
        traces_folder = os.path.join(exp.paths.trials_path, "traces")
        hdf5_file_path = os.path.join(traces_folder, f'{exp.sample.id}_fluorescence_data.h5')
        data = load_hdf5_data(hdf5_file_path, exp.sample.id)

        # Average traces
        averaged_traces, averaged_labels = average_traces(data['dff_traces'], data['odor'], data['trial_nr'])

        # Compute correlation matrices
        correlation_matrices = compute_correlation_matrices(averaged_traces)

        for i, matrix in enumerate([correlation_matrices[0], correlation_matrices[len(correlation_matrices) // 2],
                                    correlation_matrices[-1]]):
            plot_correlation_matrix(matrix, averaged_labels,
                                    f"Correlation Matrix - Window {i} - {exp.sample.id}",
                                    os.path.join(report_folder, f'{STEP_NAME}_window_{i}_{exp.sample.id}.png'))

        # Create report folder
        report_folder = os.path.join(exp.paths.root_path, "report")
        os.makedirs(report_folder, exist_ok=True)

        # Plot and save first, middle, and last correlation matrices
        plot_correlation_matrix(correlation_matrices[0], averaged_labels,
                                f"First Window Correlation Matrix - {exp.sample.id}",
                                os.path.join(report_folder, f'{STEP_NAME}_first_{exp.sample.id}.png'))

        mid_index = len(correlation_matrices) // 2
        plot_correlation_matrix(correlation_matrices[mid_index], averaged_labels,
                                f"Middle Window Correlation Matrix - {exp.sample.id}",
                                os.path.join(report_folder, f'{STEP_NAME}_middle_{exp.sample.id}.png'))

        plot_correlation_matrix(correlation_matrices[-1], averaged_labels,
                                f"Last Window Correlation Matrix - {exp.sample.id}",
                                os.path.join(report_folder, f'{STEP_NAME}_last_{exp.sample.id}.png'))

        # Save correlation matrices to HDF5 file
        with h5py.File(hdf5_file_path, 'r+') as f:
            exp_grp = f[sample_id]
            if 'correlation_matrices' in exp_grp:
                del exp_grp['correlation_matrices']
            exp_grp.create_dataset('correlation_matrices', data=correlation_matrices)
            exp_grp.create_dataset('averaged_labels', data=averaged_labels)

        print("Correlation matrices calculated and saved in HDF5 file.")

        # Update the sample database
        sample_db.update_sample_field(sample_id, STEP_NAME, True)
        sample_db.save(db_path)

        print(f"Processing completed for sample: {sample_id}")

    except Exception as e:
        logging.error(f"Error processing sample {sample_id}: {str(e)}")
        print(f"Error processing sample {sample_id}. See log for details.")


def process_samples_from_file(file_path, db_path):
    with open(file_path, 'r') as f:
        sample_ids = f.read().splitlines()
    for sample_id in sample_ids:
        try:
            process_sample(sample_id, db_path)
        except Exception as e:
            logging.error(f"Unhandled error for sample {sample_id}: {str(e)}")
            print(f"Unhandled error for sample {sample_id}. See log for details.")


def main():
    parser = argparse.ArgumentParser(description=f"Process samples for {STEP_NAME}")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--sample", help="Single sample ID to process")
    group.add_argument("-l", "--list", help="Path to text file containing sample IDs")
    parser.add_argument("--db_path", default=r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\sample_db.csv',
                        help="Path to the sample database CSV file")
    args = parser.parse_args()

    setup_logging(STEP_NAME)

    if args.sample:
        try:
            process_sample(args.sample, args.db_path)
        except Exception as e:
            logging.error(f"Unhandled error in main: {str(e)}")
            print(f"An error occurred. See log for details.")

    elif args.list:
        try:
            process_samples_from_file(args.list, args.db_path)
        except Exception as e:
            logging.error(f"Unhandled error in main: {str(e)}")
            print(e)


if __name__ == "__main__":
    main()