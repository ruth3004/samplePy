# [Step Number]_[Step Name]_batch.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from datetime import datetime
from scripts.sample_db import SampleDB

# Define the step name as a variable
STEP_NAME = '[Step Number]_[Step Name]'

def setup_logging(script_name):
    log_folder = r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\2P_RawData\log'
    os.makedirs(log_folder, exist_ok=True)
    current_date = datetime.now().strftime('%Y%m%d')
    log_file = f"{current_date}_{script_name}.log"
    log_path = os.path.join(log_folder, log_file)
    logging.basicConfig(filename=log_path, level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def process_sample(sample_id, db_path, update_all=False):
    try:
        # Load the sample database
        sample_db = SampleDB()
        sample_db.load(db_path)

        # Load experiment configuration
        exp = sample_db.get_sample(sample_id)

        # Check if this step has already been completed and if we should skip it
        if not update_all and sample_db.samples[sample_id].get(STEP_NAME) == "True":
            print(f"{STEP_NAME} already completed for sample {sample_id}. Skipping.")
            return

        # Main processing steps
        # ... (Add your main processing steps here)

        # Update the sample database
        sample_db = SampleDB()
        sample_db.load(db_path)
        sample_db.update_sample_field(sample_id, STEP_NAME, True)
        sample_db.save(db_path)

        print(f"Processing completed for sample: {sample_id}")

    except Exception as e:
        logging.error(f"Error processing sample {sample_id}: {str(e)}")
        print(f"Error processing sample {sample_id}. See log for details.")

def process_samples_from_file(file_path, db_path, update_all=False):
    with open(file_path, 'r') as f:
        sample_ids = f.read().splitlines()
    for sample_id in sample_ids:
        try:
            process_sample(sample_id, db_path, update_all)
        except Exception as e:
            logging.error(f"Unhandled error for sample {sample_id}: {str(e)}")
            print(f"Unhandled error for sample {sample_id}. See log for details.")
            sample_db = SampleDB()
            sample_db.load(db_path)
            sample_db.update_sample_field(sample_id, STEP_NAME, "Failed")
            sample_db.save(db_path)

def main():
    parser = argparse.ArgumentParser(description=f"Process samples for {STEP_NAME}")
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