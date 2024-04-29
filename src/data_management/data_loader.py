import json
from pathlib import Path
import pickle
class DataLoader:
    def __init__(self, base_directory):
        self.base_directory = Path(base_directory)

    def load_sample(sample_ID):
        """ Load instance"""  # TODO: to read from a file maybe in a database
        with open(f'{sample_ID}.pickle', 'rb') as handle:
            sample = pickle.load(handle)
            return sample

    def load_data(self, data_file):
        """Generic data loader that reads data based on the file type."""
        file_path = self.base_directory / data_file
        if file_path.suffix == '.json':
            return self.load_json(file_path)
        # Add more conditions here for different data types

    def load_json(self, file_path):
        """Loads a JSON file."""
        with open(file_path, 'r') as file:
            return json.load(file)

    def load_calcium_data(self, file_name):
        """Specifically load calcium data."""
        return self.load_data(file_name)



    def load_em_data(self, file_name):
        """Specifically load EM data."""
        return self.load_data(file_name)
