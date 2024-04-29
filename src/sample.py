import json
from types import SimpleNamespace
import os
import sys
import pickle
from pathlib import Path


from .data_management.data_getter import DataGetter
from .data_management.data_loader import DataLoader
from .data_management.data_processor import DataProcessor
from .data_management.data_plotter import DataPlotter

def recursive_namespace(data):
    """ Recursively convert dictionaries to SimpleNamespace """
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = recursive_namespace(value)
        return SimpleNamespace(**data)
    elif isinstance(data, list):
        return [recursive_namespace(item) for item in data]
    return data


class Sample(DataGetter, DataPlotter, DataProcessor, DataLoader):
    def __init__(self, sample_json):
        self.load_configuration(sample_json)
        self.setup_paths(sample_json)

    def load_configuration(self, sample_json):
        """ Load and parse the sample configuration from a JSON file using SimpleNamespace for attribute access. """
        with open(sample_json, 'r') as f:
            data = json.load(f)
        self.config = recursive_namespace(data)
        self.info = self.config.info
        self.exp = self.config.experiment

    def setup_paths(self, sample_json):
        """ Setup necessary paths from configuration using the location of the JSON file. """
        config_path = Path(sample_json).resolve()
        self.base_path = config_path.parent  # Use the directory of the JSON file as the base path
        self.results_path = self.base_path / 'results'  # Define where results should be stored relative to the JSON file
        if not self.results_path.exists():
            self.results_path.mkdir(parents=True, exist_ok=True)  # Create the results directory if it doesn't exist

    def save_sample(self, path=None):
        """ Save sample instance to a pickle file. """
        pickle_filename = f'{self.info.ID}_pickle.pickle'
        save_path = Path(path) if path else self.base_path
        with open(save_path / pickle_filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def to_dict(self):
        """Convert the Sample object to a dictionary suitable for JSON serialization."""
        # This will involve recursively converting Namespace objects to dictionaries
        def serialize(obj):
            if isinstance(obj, SimpleNamespace):
                return {k: serialize(v) for k, v in vars(obj).items()}
            elif isinstance(obj, list):
                return [serialize(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            else:
                return obj

        return serialize(vars(self))

    def save_to_json(self, filename):
        """Save the sample object to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def __repr__(self):
        return f"Sample('{self.info.ID}')"

    def __str__(self):
        return f'Sample ID: {self.info.ID}'

