import os
import shutil
import re
import argparse


def organize_files(source_dir):
    # Create destination folders if they don't exist
    trials_dir = os.path.join(source_dir, 'trials', 'raw')
    reference_dir = os.path.join(source_dir, 'reference','raw')
    anatomy_dir = os.path.join(source_dir, 'anatomy', 'raw')
    anatomy_aligned_dir = os.path.join(source_dir, 'anatomy', 'aligned_to_reference')

    for dir_path in [trials_dir, reference_dir, anatomy_dir, anatomy_aligned_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Regular expression for trial files
    trial_pattern = re.compile(r'.*_t\d+_o\d+.*\.tif$', re.IGNORECASE)

    # Iterate through files in the source directory (only first level)
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)

        # Skip if it's a directory
        if os.path.isdir(file_path):
            continue

        if filename.lower().endswith('.tif'):
            if trial_pattern.match(filename):
                # Move trial files
                dest_path = os.path.join(trials_dir, filename)
            elif 'alignment' in filename.lower() or 'reference' in filename.lower():
                # Move reference files
                dest_path = os.path.join(reference_dir, filename)
            elif 'anatomy' in filename.lower():
                # Move anatomy files
                dest_path = os.path.join(anatomy_dir, filename)
            else:
                # Skip files that don't match any category
                print(f"Skipping file: {filename}")
                continue

            # Move the file
            shutil.move(file_path, dest_path)
            print(f"Moved {filename} to {os.path.relpath(dest_path, source_dir)}")

        elif filename.lower().endswith('.nrrd') and 'anatomy' in filename.lower():
            # Move .nrrd anatomy files to the aligned_to_reference folder
            dest_path = os.path.join(anatomy_aligned_dir, filename)
            shutil.move(file_path, dest_path)
            print(f"Moved {filename} to {os.path.relpath(dest_path, source_dir)}")


def process_config_list(config_list_file):
    with open(config_list_file, 'r') as f:
        config_paths = f.read().splitlines()

    for config_path in config_paths:
        source_dir = os.path.dirname(config_path)
        print(f"Processing directory: {source_dir}")
        organize_files(source_dir)
        print(f"Finished processing {source_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize files based on a list of config file paths.")
    parser.add_argument("config_list_file", help="Path to the text file containing config file paths")
    args = parser.parse_args()

    process_config_list(args.config_list_file)
