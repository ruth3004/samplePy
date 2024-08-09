import os
import glob
import numpy as np
import pandas as pd
from tifffile import imread
import h5py
import argparse
from scripts.sample_db import SampleDB


def load_sample_db(db_path):
    sample_db = SampleDB()
    sample_db.load(db_path)
    print(sample_db)
    return sample_db


def get_experiment_info(sample_db, sample_id):
    exp = sample_db.get_sample(sample_id)
    print(f"Processing sample: {exp.sample.id}")
    return exp


def load_masks(masks_folder, sample_id):
    masks_file = glob.glob(os.path.join(masks_folder, f'masks_{sample_id}_*.tif'))[0]
    return imread(masks_file)


def process_trial(trial_path, processed_folder, masks_stack, n_planes, doubling):
    movie_path = os.path.join(processed_folder, f"motion_corrected_{trial_path}")
    movie = imread(movie_path)
    print(f"Processing trial {trial_path}, shape: {movie.shape}")

    trial_info = trial_path.split('_')
    trial_num = trial_info[5][1:]
    odor_full = trial_info[6][2:]
    odor = odor_full[2:] if odor_full.startswith('o') else odor_full

    all_traces, all_labels, all_planes, all_trials, all_odors, all_centroids = [], [], [], [], [], []

    for plane in range(n_planes * doubling):
        plane_movie = movie[plane]
        mask = masks_stack[plane, :, :]

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

    return all_traces, all_labels, all_planes, all_trials, all_odors, all_centroids


def save_to_hdf5(hdf5_file_path, sample_id, data):
    with h5py.File(hdf5_file_path, 'w') as f:
        exp_grp = f.create_group(sample_id)

        exp_grp.create_dataset('raw_traces', data=np.array(data['traces']))
        exp_grp.create_dataset('lm_plane_labels', data=np.array(data['labels']))
        exp_grp.create_dataset('plane_nr', data=np.array(data['planes']))
        exp_grp.create_dataset('trial_nr', data=np.array(data['trials'], dtype='S'))
        exp_grp.create_dataset('odor', data=np.array(data['odors'], dtype='S'))
        exp_grp.create_dataset('lm_plane_centroids', data=np.array(data['centroids']))

        mapping_grp = exp_grp.create_group('cell_mapping')
        mapping_grp.create_dataset('neuron_ids',
                                   data=np.array([f'n{i}' for i in range(1, len(data['labels']) + 1)], dtype='S'))
        mapping_grp.create_dataset('lm_plane_labels', data=np.array(data['labels']))
        mapping_grp.create_dataset('plane_nr', data=np.array(data['planes']))

    print(f"Data saved to {hdf5_file_path}")


def main(db_path, sample_id):
    sample_db = load_sample_db(db_path)
    exp = get_experiment_info(sample_db, sample_id)

    trials_path = exp.paths.trials_path
    n_planes = exp.params_lm.n_planes
    doubling = 2 if exp.params_lm.doubling else 1

    trial_names = os.listdir(os.path.join(trials_path, 'raw'))
    processed_folder = os.path.join(trials_path, 'processed')
    masks_folder = os.path.join(trials_path, "masks")

    masks_stack = load_masks(masks_folder, sample_id)

    traces_folder = os.path.join(trials_path, "traces")
    os.makedirs(traces_folder, exist_ok=True)

    hdf5_file_path = os.path.join(traces_folder, f'{sample_id}_fluorescence_data.h5')

    all_data = {
        'traces': [],
        'labels': [],
        'planes': [],
        'trials': [],
        'odors': [],
        'centroids': []
    }

    for trial_path in trial_names:
        trial_data = process_trial(trial_path, processed_folder, masks_stack, n_planes, doubling)
        for key, value in zip(all_data.keys(), trial_data):
            all_data[key].extend(value)

    save_to_hdf5(hdf5_file_path, sample_id, all_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract fluorescence traces from imaging data.")
    parser.add_argument("--db_path", default=r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\sample_db.csv',
                        help="Path to the sample database CSV file")
    parser.add_argument("--sample_id", default='20220427_RM0008_126hpf_fP3_f3', help="Sample ID to process")
    args = parser.parse_args()

    main(args.db_path, args.sample_id)
