# 12_create_correlation_matrices_batch.py

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
STEP_NAME = '12_create_correlation_matrices'

def setup_logging(script_name):
    log_folder = r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\2P_RawData\log'
    os.makedirs(log_folder, exist_ok=True)
    current_date = datetime.now().strftime('%Y%m%d')
    log_file = f"{current_date}_{script_name}.log"
    log_path = os.path.join(log_folder, log_file)
    logging.basicConfig(filename=log_path, level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_correlation_matrix(traces):
    n_traces = traces.shape[0]
    corr_matrix = np.zeros((n_traces, n_traces))
    for i in range(n_traces):
        for j in range(i, n_traces):
            corr, _ = pearsonr(traces[i], traces[j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    return corr_matrix

def plot_correlation_matrix(corr_matrix, title, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0)
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

        # Making shortcuts of sample parameters/information
        trials_path = exp.paths.trials_path

        # Create folder for saving fluorescence data
        traces_folder = os.path.join(trials_path, "traces")
        hdf5_file_path = os.path.join(traces_folder, f'{exp.sample.id}_fluorescence_data.h5')

        # Load data
        data = load_hdf5_data(hdf5_file_path, exp.sample.id)
        dff_traces = data['dff_traces']

        # Create report folder
        report_folder = os.path.join(exp.paths.root_path, "report")
        os.makedirs(report_folder, exist_ok=True)

        # Calculate and plot overall correlation matrix
        overall_corr_matrix = calculate_correlation_matrix(dff_traces)
        plot_correlation_matrix(overall_corr_matrix,
                                f"Overall Correlation Matrix - {exp.sample.id}",
                                os.path.join(report_folder, f'{STEP_NAME}_overall_{exp.sample.id}.png'))

        # Calculate and plot correlation matrices for each odor
        odors = np.unique(data['odor'])
        for odor in odors:
            odor_traces = dff_traces[data['odor'] == odor]
            odor_corr_matrix = calculate_correlation_matrix(odor_traces)
            plot_correlation_matrix(odor_corr_matrix,
                                    f"Correlation Matrix for {odor.decode('utf-8')} - {exp.sample.id}",
                                    os.path.join(report_folder, f'{STEP_NAME}_{odor.decode("utf-8")}_{exp.sample.id}.png'))

        print(f"Correlation matrices saved in {report_folder}")

        # Save correlation matrices to HDF5 file
        with h5py.File(hdf5_file_path, 'r+') as f:
            exp_grp = f[sample_id]
            if 'correlation_matrices' in exp_grp:
                del exp_grp['correlation_matrices']
            corr_grp = exp_grp.create_group('correlation_matrices')
            corr_grp.create_dataset('overall', data=overall_corr_matrix)
            for odor in odors:
                odor_traces = dff_traces[data['odor'] == odor]
                odor_corr_matrix = calculate_correlation_matrix(odor_traces)
                corr_grp.create_dataset(odor.decode('utf-8'), data=odor_corr_matrix)

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
    parser = argparse.ArgumentParser(description=f"Create correlation matrices for {STEP_NAME}")
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
            print(f"An error occurred. See log for details.")

if __name__ == "__main__":
    main()