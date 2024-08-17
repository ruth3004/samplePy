import os
import json
from datetime import datetime


def find_config_files(root_path):
    config_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('exp_config.json'):
                config_files.append(os.path.join(root, file))
    return config_files


def convert_config(old_config, config_path):
    new_config = {
        "experiment": {
            "sample": {
                "id": old_config["info"]["ID"],
                "parents_id": old_config["info"]["pair"],
                "genotype": old_config["info"]["genotype"],
                "phenotype": old_config["info"]["phenotype"],
                "dof": old_config["info"]["dof"],
                "hpf": old_config["info"]["hpf"],
                "body_length_mm": old_config["info"]["bodyLength"]
            },
            "params_odor": {
                "odor_list": old_config["experiment"]["paramsOdor"]["odorList"],
                "odor_sequence": old_config["experiment"]["paramsOdor"]["odorSequence"],
                "odor_concentration_uM": [
                    {"name": odor, "concentration_mM": 100.0}  # Default concentration, adjust as needed
                    for odor in old_config["experiment"]["paramsOdor"]["odorList"]
                ],
                "n_trials": max(old_config["experiment"]["paramsOdor"]["nTrials"].values()),
                "pulse_delay_s": old_config["experiment"]["paramsOdor"].get("pulseDelay", 15),
                "pulse_duration_s": old_config["experiment"]["paramsOdor"]["pulseDuration"],
                "trial_interval_s": old_config["experiment"]["paramsOdor"].get("trialInterval", 60),
                "missed_trials": old_config["experiment"]["paramsOdor"]["missedTrials"],
                "events": old_config["experiment"]["paramsOdor"]["bubbles"]
            },
            "params_lm": {
                "start_time": f"{old_config['experiment']['params2P']['date']}T{old_config['experiment']['params2P']['startTime']}:00",
                "end_time": f"{old_config['experiment']['params2P']['date']}T{old_config['experiment']['params2P']['endTime']}:00",
                "date": f"{old_config['experiment']['params2P']['date']}T00:00:00",
                "zoom_x": old_config["experiment"]["params2P"].get("zoom", 4.7),
                "power_percentage": old_config["experiment"]["params2P"].get("power", 30.0),
                "shutter_delay_frames": old_config["experiment"]["params2P"].get("shutterDelay", 40),
                "sampling_hz": old_config["experiment"]["params2P"]["sampFreq"],
                "n_frames": 1500,  # Assuming this value, adjust if needed
                "n_planes": old_config["experiment"]["params2P"]["nPlanes"],
                "doubling": old_config["experiment"]["params2P"]["doubling"],
                "lm_stack_range": old_config["experiment"]["params2P"]["anatoRange"],
                "ref_plane": old_config["experiment"]["params2P"]["refPlane"],
                "ref_frames_ignored": old_config["experiment"]["params2P"]["refFramesIgnored"],
                "ref_n_frames": old_config["experiment"]["params2P"].get("refNFrames", 300),
                "ref_n_slices": old_config["experiment"]["params2P"].get("refNPlanes", 9),
                "ref_slice_interval_um": old_config["experiment"]["params2P"]["refSliceInterval"],
                "n_frames_trial": 375,  # Assuming this value, adjust if needed
                "n_trials": len(old_config["experiment"]["paramsOdor"]["odorSequence"])
            },
            "params_em": {
                "fixation_protocol": old_config["experiment"]["paramsEM"]["fixation"].get("protocol", "fBROPA"),
                "embedding_protocol": "anterior-up, silver",  # Assuming this value, adjust if needed
                "acquisition_completed": True,  # Assuming this value, adjust if needed
                "acquisition_resolution_zyx": [25, 9, 9]  # Assuming these values, adjust if needed
            },
            "paths": {
                "root_path": os.path.dirname(config_path),
                "config_path": config_path,
                "trials_path": os.path.join(os.path.dirname(config_path), "trials"),
                "anatomy_path": os.path.join(os.path.dirname(config_path), "anatomy"),
                "em_path": "."
            }
        }
    }
    return new_config


def main(root_path):
    config_files = find_config_files(root_path)

    if not config_files:
        print("No config_*.json files found.")
        return

    for config_path in config_files:
        with open(config_path, 'r') as f:
            old_config = json.load(f)

        print(config_path)
        new_config = convert_config(old_config, config_path)

        sample_id = new_config["experiment"]["sample"]["id"]
        new_config_path = os.path.join(os.path.dirname(config_path), f"config_{sample_id}.json")

        with open(new_config_path, 'w') as f:
            json.dump(new_config, f, indent=4)

        print(f"New config file saved as: {new_config_path}")


if __name__ == "__main__":
    root_path = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\2P_RawData"
    main(root_path)