import os

def find_config_files(root_path):
    config_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.startswith('config_') and file.endswith('.json'):
                config_files.append(os.path.join(root, file))
    return config_files


def main(root_path, output_file):
    config_files = find_config_files(root_path)

    if not config_files:
        print("No config_*.json files found.")
        return

    with open(output_file, 'w') as f:
        for file_path in config_files:
            f.write(f"{file_path}\n")

    print(f"List of config files saved to: {output_file}")
    print(f"Total config files found: {len(config_files)}")


if __name__ == "__main__":
    root_path = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\2P_RawData"
    output_file = root_path + "\config_files_list.txt"
    main(root_path, output_file)