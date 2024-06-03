import json
from scripts.config_model import ExperimentConfig

def load_experiment_config(json_file_path: str) -> ExperimentConfig:
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    config = ExperimentConfig(**data)
    return config

# Example usage
json_file_path = 'data/calcium/2022-04-26/f3/sampleConfig_20220426_RM0008_130hpf_fP1_f3.json'
config = load_experiment_config(json_file_path)

print(config)
