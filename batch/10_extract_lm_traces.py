# [Step Number]_[Step Name]_batch.py
import glob
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from datetime import datetime
from scripts.sample_db import SampleDB
from scripts.utils.traces_utils import extract_fluorescence_data, load_hdf5_data

import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread


def setup_logging(script_name):
    log_folder = r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\2P_RawData\log'
    os.makedirs(log_folder, exist_ok=True)
    current_date = datetime.now().strftime('%Y%m%d')
    log_file = f"{current_date}_{script_name}.log"
    log_path = os.path.join(log_folder, log_file)
    logging.basicConfig(filename=log_path, level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def process_sample(sample_id, db_path):
    try:
        # Load the sample database
        sample_db = SampleDB()
        sample_db.load(db_path)

        # Load experiment configuration
        exp = sample_db.get_sample(sample_id)

        # Check if this step has already been completed
        if sample_db.samples[sample_id].get('10_extract_lm_traces') == "True":
            print(f"Step 10_extract_lm_traces already completed for sample {sample_id}. Skipping.")
            return

        # Making shortcuts of sample parameters/information
        sample = exp.sample
        trials_path = exp.paths.trials_path
        n_planes = exp.params_lm.n_planes
        n_trials = exp.params_lm.n_trials
        doubling = 2 if exp.params_lm.doubling else 1

        # Getting paths of the trial acquisitions
        trial_names = os.listdir(os.path.join(trials_path, 'raw'))
        processed_folder = os.path.join(trials_path, 'processed')
        masks_folder = os.path.join(trials_path, "masks")

        # Load the masks
        masks_file = glob.glob(os.path.join(masks_folder, f'masks_{exp.sample.id}_*.tif'))[0]
        masks_stack = imread(masks_file)

        # Create folder for saving fluorescence data
        traces_folder = os.path.join(trials_path, "traces")
        os.makedirs(traces_folder, exist_ok=True)

        hdf5_file_path = os.path.join(traces_folder, f'{exp.sample.id}_fluorescence_data.h5')

        # Process and save data
        extract_fluorescence_data(hdf5_file_path, sample_id, trial_names, processed_folder, masks_stack, n_planes,
                                  doubling)

        # Load data
        data = load_hdf5_data(hdf5_file_path, sample_id)

        # Create report folder
        report_folder = os.path.join(exp.paths.root_path, "report")
        os.makedirs(report_folder, exist_ok=True)


        # Plot the first three traces
        plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(data['raw_traces'][i], label=f'Label {data["lm_plane_labels"][i]}')

        plt.title('Fluorescence Intensity Traces')
        plt.xlabel('Time (frames)')
        plt.ylabel('Fluorescence Intensity')
        plt.legend()

        # Save the first plot
        first_plot_path = os.path.join(report_folder, f'10_extract_lm_traces_first_raw_traces_{exp.sample.id}.png')
        plt.savefig(first_plot_path)
        plt.close()  # Close the plot to free up memory

        # Plot average traces for each odor
        odors_name = np.unique(data['odor'])
        plt.figure(figsize=(10, 6))
        for odor in odors_name:
            plt.plot(data['raw_traces'][data['odor'] == odor].mean(axis=0), label=odor.decode('utf-8'))
        plt.legend()
        plt.title('Average Fluorescence Intensity Traces by Odor')
        plt.xlabel('Time (frames)')
        plt.ylabel('Fluorescence Intensity')

        # Save the second plot
        second_plot_path = os.path.join(report_folder, f'10_extract_lm_traces_average_raw_traces_{exp.sample.id}.png')
        plt.savefig(second_plot_path)
        plt.close()  # Close the plot to free up memory

        print(f"Plots saved in {report_folder}")

        # Update the sample database
        sample_db.update_sample_field(sample_id, '10_extract_lm_traces', True)
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
            sample_db = SampleDB()
            sample_db.load(db_path)
            sample_db.update_sample_field(sample_id, '02_register_lm_trials_lm_stack', "Failed")
            sample_db.save(db_path)
def main():
    parser = argparse.ArgumentParser(description="Process samples for step 10_extract_lm_traces")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--sample", help="Single sample ID to process")
    group.add_argument("-l", "--list", help="Path to text file containing sample IDs")
    parser.add_argument("--db_path", default=r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\sample_db.csv',
                        help="Path to the sample database CSV file")
    args = parser.parse_args()

    setup_logging('10_extract_lm_traces')

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


