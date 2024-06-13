import csv
import json
from typing import Dict, Optional, Any
from pydantic import BaseModel
from scripts.config_model import Sample, Experiment

class SampleDB:
    def __init__(self):
        self.samples: Dict[str, Dict[str, Optional[str]]] = {}
        # Define the initial set of columns
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
            writer = csv.DictWriter(csvfile, fieldnames=self.columns)
            writer.writeheader()
            for sample_id, sample_data in self.samples.items():
                row = {
                    'sample_id': sample_data["sample"].id,
                    'root_path': sample_data["root_path"],
                    'config_path': sample_data["config_path"],
                    'trials_path': sample_data["trials_path"],
                    'anatomy_path': sample_data["anatomy_path"],
                    'em_path': sample_data["em_path"],
                    'update': sample_data["update"]
                }
                # Add any extra columns
                for col in self.columns:
                    if col not in row:
                        row[col] = sample_data.get(col, None)
                writer.writerow(row)

    def load(self, file_path: str):
        try:
            with open(file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    sample = Sample(
                        id=row.get('sample_id'),
                        parents_id=row.get('parents_id'),
                        genotype=row.get('genotype'),
                        phenotype=row.get('phenotype'),
                        dof=row.get('dof', ''),
                        hpf=int(row.get('hpf', 0)),
                        body_length_mm=int(row.get('body_length_mm', 0))
                    )
                    self.samples[sample.id] = {
                        "sample": sample,
                        "root_path": row.get('root_path', ''),
                        "config_path": row.get('config_path', ''),
                        "trials_path": row.get('trials_path', ''),
                        "anatomy_path": row.get('anatomy_path', ''),
                        "em_path": row.get('em_path', None),
                        "update": row.get('update', '')
                    }
                    # Load any extra columns
                    for col in row:
                        if col not in self.samples[sample.id]:
                            self.samples[sample.id][col] = row[col]
                    # Update columns list
                    if set(self.columns) != set(row.keys()):
                        self.columns = list(row.keys())
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
                "anatomy_path": sample_data["anatomy_path"],
                "em_path": sample_data["em_path"]
            }
            config_dict = sample.dict()
            config_dict['segment_paths'] = segment_paths
            return json.dumps({"experiment": config_dict}, indent=4)
        return None

    def __repr__(self):
        sample_ids = list(self.samples.keys())
        return f"SampleDB(sample_ids={sample_ids})"
