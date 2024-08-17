import csv
import json
from typing import Dict, Optional, Any
from pydantic import BaseModel
from scripts.config_model import Sample, Experiment

class SampleDB:
    def __init__(self):
        self.samples: Dict[str, Dict[str, Any]] = {}
        self.columns = ['sample_id', 'root_path', 'config_path', 'trials_path', 'anatomy_path', 'em_path', 'update']

    def get_sample(self, sample_id: str) -> Optional[Experiment]:
        sample_data = self.samples.get(sample_id)
        if sample_data:
            config_path = sample_data["config_path"]
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                experiment = Experiment(**config_data['experiment'])
            return experiment
        return None

    def save(self, file_path: str):
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = set(self.columns)
            for sample_data in self.samples.values():
                fieldnames.update(sample_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=list(fieldnames))
            writer.writeheader()
            for sample_id, sample_data in self.samples.items():
                row = sample_data.copy()
                row['sample_id'] = sample_id
                writer.writerow(row)

    def load(self, file_path: str):
        try:
            with open(file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                self.columns = reader.fieldnames
                for row in reader:
                    sample_id = row['sample_id']
                    self.samples[sample_id] = row
        except FileNotFoundError:
            print(f"File {file_path} not found. Initializing empty database.")
            self.save(file_path)  # Create a new CSV file with headers
        except Exception as e:
            print(f"Error loading sample database: {e}")

    def add_column(self, column_name: str, default_value: Any = None):
        if column_name not in self.columns:
            self.columns.append(column_name)
            for sample_id in self.samples:
                self.samples[sample_id][column_name] = default_value

    def update_sample_field(self, sample_id: str, field: str, value: Any):
        if sample_id in self.samples:
            self.samples[sample_id][field] = value
            if field not in self.columns:
                self.columns.append(field)
        else:
            print(f"Sample {sample_id} not found in the database.")

    def update_column(self, column_name: str, update_function):
        if column_name in self.columns:
            for sample_id in self.samples:
                self.samples[sample_id][column_name] = update_function(self.samples[sample_id])

    def delete_column(self, column_name: str):
        if column_name in self.columns:
            self.columns.remove(column_name)
            for sample_id in self.samples:
                if column_name in self.samples[sample_id]:
                    del self.samples[sample_id][column_name]

    def get_sample_json(self, sample_id: str) -> Optional[str]:
        sample_data = self.samples.get(sample_id)
        if sample_data:
            sample = sample_data["sample"]
            segment_paths = {
                "root_path": sample_data["root_path"],
                "config_path": sample_data["config_path"],
                "trials_path": sample_data["trials_path"],
                "em_path": sample_data["em_path"]
            }
            config_dict = sample.dict()
            config_dict['segment_paths'] = segment_paths
            return json.dumps({"experiment": config_dict}, indent=4)
        return None


    def __repr__(self):
        sample_ids = list(self.samples.keys())
        return f"SampleDB(sample_ids={sample_ids})"
