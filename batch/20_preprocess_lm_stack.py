# 20_preprocess_lm_stack_batch.py
import os
import sys
import glob
import argparse
import logging
from datetime import datetime
import numpy as np
from skimage import transform, exposure
from tifffile import imwrite

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.sample_db import SampleDB
from scripts.utils.image_utils import load_anatomy_stack

def setup_logging(script_name):
    log_folder = r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\2P_RawData\log'
    os.makedirs(log_folder, exist_ok=True)
    current_date = datetime.now().strftime('%Y%m%d')
    log_file = f"{current_date}_{script_name}.log"
    log_path = os.path.join(log_folder, log_file)
    logging.basicConfig(filename=log_path, level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def save_array_as_hyperstack_tiff(path, array):
    array_reshaped = array.transpose(1, 0, 2, 3).astype(np.float32)
    imwrite(path, array_reshaped, imagej=True)

def process_sample(sample_id, db_path):
    try:
        # Load the sample database
        sample_db = SampleDB()
        sample_db.load(db_path)

        # Load experiment configuration
        exp = sample_db.get_sample(sample_id)

        # Check if this step has already been completed
        if sample_db.samples[sample_id].get('20_preprocess_lm_stack') == "True":
            print(f"Step 20_preprocess_lm_stack already completed for sample {sample_id}. Skipping.")
            return

        print(f"Processing sample: {sample_id}")
        print("Anatomy path:", exp.paths.anatomy_path)

        # Find all anatomy files in the raw folder
        anatomy_files = glob.glob(os.path.join(exp.paths.anatomy_path, 'raw', '*anatomy*.tif'))

        for lm_stack_path in anatomy_files:
            try:
                lm_stack_name = os.path.basename(lm_stack_path)
                print(f"Processing: {lm_stack_name}")

                # Check for noise2void processed version
                n2v_stack_name = lm_stack_name.replace('.tif', '_n2v.tif')
                n2v_stack_path = os.path.join(exp.paths.anatomy_path, 'processed', n2v_stack_name)

                if os.path.exists(n2v_stack_path):
                    print("Using noise2void stack")
                    lm_stack = load_anatomy_stack(n2v_stack_path)
                else:
                    print("Using original stack")
                    lm_stack = load_anatomy_stack(lm_stack_path)

                print("LM stack shape:", lm_stack.shape)

                # Flip horizontally
                print("Flipping stack")
                lm_stack_flip = np.flip(lm_stack, axis=3)

                # Upsample the stack
                print("Upsampling stack")
                upsample_factor = (1, 2.5, 1, 1)
                lm_stack_upsampled = transform.rescale(lm_stack_flip, upsample_factor, order=1, mode='reflect',
                                                       anti_aliasing=True, preserve_range=False)

                # Apply CLAHE
                print("Applying CLAHE")
                lm_stack_clahe = lm_stack_upsampled.copy()
                for i in range(lm_stack_clahe.shape[0]):
                    lm_stack_clahe[i] = exposure.equalize_adapthist(lm_stack_upsampled[i])

                # Save the preprocessed stack
                output_folder = os.path.join(exp.paths.anatomy_path, 'processed')
                os.makedirs(output_folder, exist_ok=True)
                output_file = os.path.join(output_folder, f"flipped_upsampled_clahe_{lm_stack_name}")
                save_array_as_hyperstack_tiff(output_file, lm_stack_clahe)

                print(f"Preprocessed stack saved to: {output_file}")

            except Exception as e:
                logging.error(f"Error processing sample {sample_id}: {str(e)}")
                print(f"Error processing sample {sample_id}. See log for details.")

        # Update the '01_register_lm_trials' checkpoint
        sample_db.update_sample_field(sample_id, '20_preprocess_lm_stack', True)
        sample_db.save(db_path)

        print(f"Completed processing for sample: {sample_id}")

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
    parser = argparse.ArgumentParser(description="Preprocess LM stack")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--sample", help="Single sample ID to process")
    group.add_argument("-l", "--list", help="Path to text file containing sample IDs")
    parser.add_argument("--db_path", default=r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\sample_db.csv',
                        help="Path to the sample database CSV file")
    args = parser.parse_args()

    setup_logging('20_preprocess_lm_stack')

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