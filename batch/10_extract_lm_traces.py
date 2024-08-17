# [Step Number]_[Step Name]_batch.py
import glob
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from datetime import datetime
from scripts.sample_db import SampleDB

import numpy as np

import h5py
from tifffile import imread


def setup_logging(script_name):
    log_folder = r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\2P_RawData\log'
    os.makedirs(log_folder, exist_ok=True)
    current_date = datetime.now().strftime('%Y%m%d')
    log_file = f"{current_date}_{script_name}.log"
    log_path = os.path.join(log_folder, log_file)
    logging.basicConfig(filename=log_path, level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')


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

def process_sample(sample_id, db_path):
    try:
        # Load the sample database
        sample_db = SampleDB()
        sample_db.load(db_path)

        # Load experiment configuration
        exp = sample_db.get_sample(sample_id)

        # Check if this step has already been completed
        if sample_db.samples[sample_id].get('10_extract_lm_traces', False):
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

        extract_fluorescence_data(hdf5_file_path, sample_id, trial_names, processed_folder, masks_stack, n_planes,
                                  doubling)

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


