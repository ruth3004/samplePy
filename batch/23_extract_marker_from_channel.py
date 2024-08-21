import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from datetime import datetime
import numpy as np
import tifffile as tiff
from skimage.measure import manders_coloc_coeff, regionprops, label
from skimage.filters import threshold_otsu
from scipy.ndimage import zoom

from scripts.sample_db import SampleDB
from scripts.utils.image_utils import load_tiff_as_hyperstack, calculate_manders_coefficient_3d

# Define the step name as a variable
STEP_NAME = '23_extract_marker_from_channel'

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

        # Define paths
        anatomy_stack_path = os.path.join(exp.paths.anatomy_path, 'raw', f'{exp.sample.id}_anatomyGFRF_001_.tif')
        anatomy_mask_path = os.path.join(exp.paths.clem_path, 'pycpd', 'em_stack_channel 1_xfm_0_flfov_lmresolution_cp_masks.tif')

        # Load anatomy stack and mask
        anatomy_stack = load_tiff_as_hyperstack(anatomy_stack_path, n_channels=2)
        anatomy_mask = tiff.imread(anatomy_mask_path)

        # Flip and resample anatomy stack
        flipped_anatomy_stack = anatomy_stack[:, :, :, ::-1]
        zoom_factors = [2.5, 1, 1]
        resampled_shape = (flipped_anatomy_stack.shape[0], int(flipped_anatomy_stack.shape[1] * zoom_factors[0]), flipped_anatomy_stack.shape[2], flipped_anatomy_stack.shape[3])
        resampled_anatomy_stack = np.zeros(resampled_shape, dtype=flipped_anatomy_stack.dtype)

        for channel in range(flipped_anatomy_stack.shape[0]):
            resampled_anatomy_stack[channel] = zoom(flipped_anatomy_stack[channel], zoom_factors, order=3)

        # Save resampled stack
        resampled_stack_path = os.path.join(exp.paths.anatomy_path, 'processed', f'resampled_flipped_{exp.sample.id}_anatomyGFRF_001_.tif')
        tiff.imwrite(resampled_stack_path, resampled_anatomy_stack)

        # Calculate Manders' coefficients
        c0_anatomy_stack = resampled_anatomy_stack[0]
        c1_anatomy_stack = resampled_anatomy_stack[1]

        manders_results_c0, mask_colored_stack_c0 = calculate_manders_coefficient_3d(anatomy_mask, c0_anatomy_stack)
        manders_results_c1, mask_colored_stack_c1 = calculate_manders_coefficient_3d(anatomy_mask, c1_anatomy_stack)

        # Save Manders' coefficient based mask stacks
        tiff.imwrite(os.path.join(exp.paths.anatomy_path, 'masks', f'mask_manderscoeff_c0_{exp.sample.id}_anatomyGFRF_001_.tif'), mask_colored_stack_c0)
        tiff.imwrite(os.path.join(exp.paths.anatomy_path, 'masks', f'mask_manderscoeff_c1_{exp.sample.id}_anatomyGFRF_001_.tif'), mask_colored_stack_c1)

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
        parser.add_argument("--db_path",
                            default=r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\sample_db.csv',
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