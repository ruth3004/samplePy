import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set this at the beginning of your script
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
from tifffile import imread
from skimage.transform import SimilarityTransform
from scripts.sample_db import SampleDB
from scripts.utils.image_utils import load_tiff_as_hyperstack, save_array_as_hyperstack_tiff, \
    plane_detection_with_iterative_alignment, warp_stack_to_plane
import argparse
import logging
import datetime


def setup_logging(script_name):
    log_folder = r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\2P_RawData\log'
    os.makedirs(log_folder, exist_ok=True)
    current_date = datetime.now().strftime('%Y%m%d')
    log_file = f"{current_date}_{script_name}.log"
    log_path = os.path.join(log_folder, log_file)
    logging.basicConfig(filename=log_path, level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def process_sample(sample_id, db_path):
    # Load the sample database
    sample_db = SampleDB()
    sample_db.load(db_path)

    # Load experiment configuration
    exp = sample_db.get_sample(sample_id)

    # Check if this step has already been completed
    if sample_db.samples[sample_id].get('02_register_lm_trials_lm_stack')=="True":
        print(f"Step 02 already completed for sample {sample_id}. Skipping.")
        return

    # Making shortcuts of sample parameters/information
    sample = exp.sample
    root_path = exp.paths.root_path
    trials_path = exp.paths.trials_path
    anatomy_path = exp.paths.anatomy_path
    n_planes = exp.params_lm.n_planes

    # Define the path for the preprocessed folder
    processed_folder = os.path.join(trials_path, "processed")
    os.makedirs(processed_folder, exist_ok=True)

    # Create a report folder
    report_folder = os.path.join(root_path, "report")
    os.makedirs(report_folder, exist_ok=True)

    ref_images_path = os.path.join(processed_folder, f"sum_raw_trials_{sample.id}.tif")

    # Load planes and stack
    lm_planes = imread(ref_images_path)
    lm_stack = \
    load_tiff_as_hyperstack(os.path.join(anatomy_path, "raw", f"{sample.id}_anatomyGFRF_001_.tif"), n_channels=2)[0]

    # Find planes in stack
    for ii, lm_plane in enumerate(lm_planes[:, -1]):
        print(f"Processing Plane {ii}")
        final_transform, all_transformation_matrices = plane_detection_with_iterative_alignment(
            lm_plane, lm_stack, equalize=True, binning=True, plot=False, nx=2, ny=3, tiles_filter=None,
            thickness_values=None
        )

        # Save the transformation matrix
        np.save(os.path.join(processed_folder, f"registration_tform_lm_plane0{ii}_lm_stack.npy"), final_transform)

        # Generate and save the report image
        last_matched_slice = warp_stack_to_plane(lm_stack, lm_plane, SimilarityTransform(final_transform), 20)[10]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(lm_plane, cmap='gray')
        ax1.set_title('Original Plane')
        ax1.axis('off')
        ax2.imshow(last_matched_slice, cmap='gray')
        ax2.set_title('Matched Slice')
        ax2.axis('off')
        plt.tight_layout()

        report_image_path = os.path.join(report_folder,
                                         f"02_register_lm_trials_lm_stack_matched_slice_plane_{ii:02d}.png")
        plt.savefig(report_image_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Update the sample database
    sample_db = SampleDB()
    sample_db.load(db_path)
    sample_db.update_sample_field(sample_id, '02_register_lm_trials_lm_stack', True)
    sample_db.save(db_path)

    print(f"Processing completed for sample: {sample_id}")


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
    parser = argparse.ArgumentParser(description="Process samples for step 02")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--sample", help="Single sample ID to process")
    group.add_argument("-l", "--list", help="Path to text file containing sample IDs")
    parser.add_argument("--db_path", default=r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\sample_db.csv',
                        help="Path to the sample database CSV file")
    parser.add_argument("-o", "--overwrite", help="Overwrite existing processed files", action="store_true")
    args = parser.parse_args()

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

