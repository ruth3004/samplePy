# 11_normalize_lm_traces_batch.py
import os
import sys
import argparse
import logging
from datetime import datetime
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.sample_db import SampleDB
from scripts.utils.traces_utils import load_hdf5_data, calculate_dff, plot_average_traces_by_group

STEP_NAME = '11_normalize_lm_traces'

def setup_logging(script_name):
    log_folder = r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\2P_RawData\log'
    os.makedirs(log_folder, exist_ok=True)
    current_date = datetime.now().strftime('%Y%m%d')
    log_file = f"{current_date}_{script_name}.log"
    log_path = os.path.join(log_folder, log_file)
    logging.basicConfig(filename=log_path, level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def detect_spikes(trace, threshold_factor=2):
    """Detect spikes in a trace based on a threshold."""
    threshold = np.mean(trace) + threshold_factor * np.std(trace)
    spikes = signal.find_peaks(trace, height=threshold)[0]
    return spikes

def align_traces(traces, odors, alignment_point):
    """Align traces based on a specific alignment point."""
    aligned_traces = []
    for trace in traces:
        aligned_trace = np.roll(trace, -alignment_point)
        aligned_traces.append(aligned_trace)
    return np.array(aligned_traces)

def process_sample(sample_id, db_path, update_all=False):
    try:
        # Load the sample database
        sample_db = SampleDB()
        sample_db.load(db_path)

        # Load experiment configuration
        exp = sample_db.get_sample(sample_id)

        # Check if this step has already been completed
        if not update_all and sample_db.samples[sample_id].get(STEP_NAME) == "True":
            print(f"{STEP_NAME} already completed for sample {sample_id}. Skipping.")
            return

        # Making shortcuts of sample parameters/information
        trials_path = exp.paths.trials_path

        # Create folder for saving fluorescence data
        traces_folder = os.path.join(trials_path, "traces")
        os.makedirs(traces_folder, exist_ok=True)

        hdf5_file_path = os.path.join(traces_folder, f'{exp.sample.id}_fluorescence_data.h5')

        # Load data
        data = load_hdf5_data(hdf5_file_path, exp.sample.id)

        # Calculate df/f traces
        dff_traces = calculate_dff(data['raw_traces'], baseline_frames=[50, 100])

        # Create report folder
        report_folder = os.path.join(exp.paths.root_path, "report")
        os.makedirs(report_folder, exist_ok=True)


        # Plot the first three df/f traces
        plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(dff_traces[i], label=f'Label {data["lm_plane_labels"][i]}')

        plt.title('df/f Traces')
        plt.xlabel('Time (frames)')
        plt.ylabel('df/f')
        plt.legend()

        # Save the first plot
        first_plot_path = os.path.join(report_folder, f'11_normalize_lm_traces_first_dff_traces_{exp.sample.id}.png')
        plt.savefig(first_plot_path)
        plt.close()  # Close the plot to free up memory

        # Detect spikes and align traces for each odor
        odors_name = np.unique(data['odor'])
        aligned_dff_traces = np.zeros_like(dff_traces)
        spike_times = {}

        plt.figure(figsize=(15, 10))
        for i, odor in enumerate(odors_name):
            odor_traces = dff_traces[data['odor'] == odor]
            mean_trace = np.mean(odor_traces, axis=0)

            # Detect spikes
            spikes = detect_spikes(mean_trace)
            spike_times[odor] = spikes

            # Align traces if spikes are detected (skip for SA)
            if len(spikes) > 0 and odor != b'SA':
                alignment_point = spikes[0]  # Align to the first spike
                aligned_odor_traces = align_traces(odor_traces, odor, alignment_point)
                aligned_dff_traces[data['odor'] == odor] = aligned_odor_traces
            else:
                aligned_dff_traces[data['odor'] == odor] = odor_traces

            # Plot
            plt.subplot(3, 3, i + 1)
            plt.plot(mean_trace, label='Mean')
            plt.plot(spikes, mean_trace[spikes], 'r*', label='Spikes')
            plt.title(f'Odor: {odor.decode("utf-8")}')
            plt.xlabel('Time (frames)')
            plt.ylabel('df/f')
            plt.legend()

        plt.tight_layout()
        spike_plot_path = os.path.join(report_folder, f'11_normalize_lm_traces_spike_detection_{exp.sample.id}.png')
        plt.savefig(spike_plot_path)
        plt.close()

        # Plot average aligned df/f traces for each odor
        plt.figure(figsize=(10, 6))
        for odor in odors_name:
            plt.plot(np.mean(aligned_dff_traces[data['odor'] == odor], axis=0), label=odor.decode('utf-8'))
        plt.legend()
        plt.title('Average Aligned df/f Traces by Odor')
        plt.xlabel('Time (frames)')
        plt.ylabel('df/f')

        aligned_plot_path = os.path.join(report_folder,
                                         f'11_normalize_lm_traces_aligned_average_dff_traces_{exp.sample.id}.png')
        plt.savefig(aligned_plot_path)
        plt.close()

        print(f"Plots saved in {report_folder}")

        # Save aligned df/f traces and spike times to HDF5 file
        with h5py.File(hdf5_file_path, 'r+') as f:
            exp_grp = f[sample_id]
            if 'aligned_dff_traces' in exp_grp:
                del exp_grp['aligned_dff_traces']
            exp_grp.create_dataset('aligned_dff_traces', data=aligned_dff_traces)

            if 'spike_times' in exp_grp:
                del exp_grp['spike_times']
            spike_times_group = exp_grp.create_group('spike_times')
            for odor, spikes in spike_times.items():
                spike_times_group.create_dataset(odor.decode('utf-8'), data=spikes)

        print("Aligned df/f traces and spike times calculated and saved in HDF5 file.")

        # Update the sample database
        sample_db = SampleDB()
        sample_db.load(db_path)
        sample_db.update_sample_field(sample_id, '11_normalize_lm_traces', True)
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
    parser = argparse.ArgumentParser(description="Normalize LM traces")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--sample", help="Single sample ID to process")
    group.add_argument("-l", "--list", help="Path to text file containing sample IDs")
    parser.add_argument("--db_path", default=r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\sample_db.csv',
                        help="Path to the sample database CSV file")
    parser.add_argument("--update_all", action='store_true', help="Ignore checks for already completed steps")
    args = parser.parse_args()

    setup_logging(STEP_NAME)

    if args.sample:
        try:
            process_sample(args.sample, args.db_path, args.update_all)
        except Exception as e:
            logging.error(f"Unhandled error in main: {str(e)}")
            print(f"An error occurred. See log for details.")

    elif args.list:
        try:
            process_samples_from_file(args.list, args.db_path, args.update_all)
        except Exception as e:
            logging.error(f"Unhandled error in main: {str(e)}")
            print(f"An error occurred. See log for details.")


if __name__ == "__main__":
    main()