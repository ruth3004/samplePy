import sys
import os
import argparse
from typing import List, Dict, Any, OrderedDict


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.sample_db import SampleDB
from scripts.config_model import Experiment, load_experiment_config, save_experiment_config, update_experiment_config

# Define processing steps
processing_steps: List[str] = [
    '00_load_experiment',
    '01_register_lm_trials',
    '02_register_lm_trials_lm_stack',
    '03_segment_lm_trials',
    '10_extract_lm_traces',
    '11_normalize_lm_traces',
    '12_deconvolve_lm_traces',
    '13_analyze_correlation_lm_traces',
    '20_preprocess_lm_stack',
    '21_register_lm_stack_to_ref_stack',
    '22_segment_lm_stack_from_em_warped_stack',
    '23_extract_marker_from_channel',
    '30_segment_em_stack',
    '31_segment_glomeruli',
    '32_find_landmarks_with_BigWarp',
    '33_register_em_stack_lm_stack_from_landmarks',
    '34_register_em_stack_lm_trials'
]

def process_config(config_file_path: str, db_file_path: str, update_all: bool) -> None:
    """
    Process a single configuration file and update the sample database.

    Args:
        config_file_path (str): Path to the configuration file.
        db_file_path (str): Path to the sample database file.
        update_all (bool): Whether to update all samples or not.
    """
    print(f"Processing config file: {config_file_path}")

    try:
        config = load_experiment_config(config_file_path)
        print("Loaded experiment configuration:")
        print(config.paths)

        root_path = os.path.dirname(config_file_path)
        trials_path = os.path.join(root_path, "trials")
        anatomy_path = os.path.join(root_path, "anatomy")

        # Add to config structure
        config.paths.root_path = root_path
        config.paths.config_path = config_file_path
        config.paths.anatomy_path = anatomy_path
        config.paths.trials_path = trials_path

        save_experiment_config(config)

        # Load or create a sample database
        sample_db = SampleDB()

        try:
            sample_db.load(db_file_path)
            print("Loaded existing sample database.")
        except FileNotFoundError:
            print("Sample database not found. Creating a new one.")
            sample_db.save(db_file_path)  # Create a new CSV file with headers

        sample = config.sample
        print(f"Sample ID: {sample.id}")
        existing_sample = sample_db.get_sample(sample.id)

        # Check if the sample is already in the database and ask if you want to update it
        update = update_all
        if existing_sample and not update_all:
            print(f"Sample with ID {sample.id} already exists.")
            update = input("Do you want to update this sample? (y/n): ").lower() == 'y'

        print(f"Update: {update}")

        # Ensure initial columns are present
        for column in ['sample_id', 'root_path', 'config_path', 'trials_path', 'anatomy_path', 'em_path',
                       'update'] + processing_steps:
            sample_db.add_column(column)

        # Add the sample data to the database
        sample_data: Dict[str, Any] = {
            "sample": sample,
            "root_path": root_path,
            "config_path": config_file_path,
            "trials_path": trials_path,
            "anatomy_path": anatomy_path,
            "em_path": "",
            "update": update
        }

        # Initialize all processing steps to False
        for step in processing_steps:
            sample_data[step] = False

        sample_db.samples[sample.id] = sample_data

        # Update the '00_load_experiment' step to True
        sample_db.update_sample_field(sample.id, '00_load_experiment', True)
        sample_db.save(db_file_path)
        print(sample_db.samples[sample.id])
        print(f"Sample database saved to {db_file_path}.")


    except Exception as e:
        print(f"Error processing config file {config_file_path}: {str(e)}")


def main(config_list_file: str, db_file_path: str, update_all: bool) -> None:
    """
    Process multiple configuration files listed in a file.

    Args:
        config_list_file (str): Path to the file containing list of config file paths.
        db_file_path (str): Path to the sample database file.
        update_all (bool): Whether to update all samples or not.
    """
    with open(config_list_file, 'r') as f:
        config_files = f.read().splitlines()

    for config_file in config_files:
        process_config(config_file, db_file_path, update_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process config files and update sample database.")
    parser.add_argument("config_list_file", help="File containing list of config file paths")
    parser.add_argument("--db_file_path", default=r"\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\sample_db.csv", help="Path to the sample database CSV file")
    parser.add_argument("--update_all", action="store_true", help="Whether to update all samples or not.")
    args = parser.parse_args()

    main(args.config_list_file, args.db_file_path, args.update_all)