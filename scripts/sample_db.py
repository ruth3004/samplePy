import csv
import json
from typing import Dict, Optional
from pydantic import BaseModel
from scripts.config_model import Sample, Experiment

class SampleDB:
    def __init__(self):
        self.samples: Dict[str, Dict[str, Optional[str]]] = {}

    def add_or_update_sample(self, sample: Sample, raw_path: str, anatomy_path: str, config_path: str, em_path: Optional[str] = None, *, update: bool = False):
        if sample.id in self.samples:
            if update:
                print(f"Updating existing sample with ID {sample.id}")
        else:
            print(f"Adding new sample with ID {sample.id}")

        self.samples[sample.id] = {
            "sample": sample,
            "raw_path": raw_path,
            "anatomy_path": anatomy_path,
            "em_path": em_path,
            "config_path": config_path
        }

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
            fieldnames = ['sample_id', 'parents_id', 'genotype', 'phenotype', 'dof', 'hpf', 'body_length_mm',
                          'raw_path', 'anatomy_path', 'em_path', 'config_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for sample_id, sample_data in self.samples.items():
                row = {
                    'sample_id': sample_data["sample"].id,
                    'parents_id': sample_data["sample"].parents_id,
                    'genotype': sample_data["sample"].genotype,
                    'phenotype': sample_data["sample"].phenotype,
                    'dof': sample_data["sample"].dof,
                    'hpf': sample_data["sample"].hpf,
                    'body_length_mm': sample_data["sample"].body_length_mm,
                    'raw_path': sample_data["raw_path"],
                    'anatomy_path': sample_data["anatomy_path"],
                    'em_path': sample_data["em_path"],
                    'config_path': sample_data["config_path"]
                }
                writer.writerow(row)

    def load(self, file_path: str):
        try:
            with open(file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    sample = Sample(
                        id=row['sample_id'],
                        parents_id=row['parents_id'],
                        genotype=row['genotype'],
                        phenotype=row['phenotype'],
                        dof=row['dof'],
                        hpf=int(row['hpf']),
                        body_length_mm=int(row['body_length_mm'])
                    )
                    self.samples[sample.id] = {
                        "sample": sample,
                        "raw_path": row['raw_path'],
                        "anatomy_path": row['anatomy_path'],
                        "em_path": row['em_path'] if row['em_path'] else None,
                        "config_path": row['config_path']
                    }
        except FileNotFoundError:
            print(f"File {file_path} not found. Initializing empty database.")
            self.save(file_path)  # Create a new CSV file with headers
        except Exception as e:
            print(f"Error loading sample database: {e}")

    def get_sample_json(self, sample_id: str) -> Optional[str]:
        sample_data = self.samples.get(sample_id)
        if sample_data:
            sample = sample_data["sample"]
            segment_paths = {
                "raw_path": sample_data["raw_path"],
                "anatomy_path": sample_data["anatomy_path"],
                "em_path": sample_data["em_path"],
                "config_path": sample_data["config_path"]
            }
            config_dict = sample.dict()
            config_dict['segment_paths'] = segment_paths
            return json.dumps({"experiment": config_dict}, indent=4)
        return None

    def __repr__(self):
        sample_ids = list(self.samples.keys())
        return f"SampleDB(sample_ids={sample_ids})"
