import json
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, date

class Sample(BaseModel):
    id: str
    parents_id: Optional[str]
    genotype: Optional[str]
    phenotype: Optional[str]
    dof: str  # date of fertilization: Including both date and time in the format "dd.mm.yyyy-hh:mm"
    hpf: int  # Hours post fertilization
    body_length_mm: Optional[int]

class OdorConcentration(BaseModel):
    name: str
    concentration_mM: float

class ParamsOdor(BaseModel):
    odor_list: List[str]
    odor_sequence: List[str]
    odor_concentration_uM: List[OdorConcentration]
    n_trials: int
    pulse_delay_s: int
    pulse_duration_s: int
    trial_interval_s: int
    missed_trials: List = []
    events: List[Tuple[str, datetime]] = []

class ParamsLM(BaseModel):
    start_time: datetime  # Including both date and time in the format "yyyy-mm-ddTHH:MM:SS"
    end_time: datetime    # Including both date and time in the format "yyyy-mm-ddTHH:MM:SS"
    date: Optional[datetime]
    zoom_x: Optional[float]
    power_percentage: Optional[float]
    shutter_delay_frames: Optional[int]
    sampling_hz: int
    n_frames: int
    n_planes: int
    doubling: bool
    lm_stack_range: int
    ref_plane: int
    ref_frames_ignored: int
    ref_n_frames: Optional[int]
    ref_n_slices: Optional[int]
    ref_slice_interval_um: float

class ParamsEM(BaseModel):
    fixation_protocol: str
    embedding_protocol: str
    acquisition_completed: bool
    acquisition_resolution_zyx: Tuple[int, int, int]

class Experiment(BaseModel):
    sample: Sample
    params_odor: Optional[ParamsOdor]
    params_lm: Optional[ParamsLM]
    params_em: Optional[ParamsEM]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        extra = 'allow'

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)

def load_experiment_config(json_file_path: str) -> Experiment:
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return Experiment(**data['experiment'])

# Function to update any part of the experiment configuration
def update_experiment_config(config: Experiment, updates: Dict[str, Any]) -> Experiment:
    config_dict = config.dict()
    for key, value in updates.items():
        # Update nested dictionaries if needed
        if isinstance(value, dict) and key in config_dict:
            config_dict[key].update(value)
        else:
            config_dict[key] = value
    updated_config = Experiment(**config_dict)
    return updated_config


# Function to save the updated configuration back to the JSON file
def save_experiment_config(config: Experiment, json_file_path: str):
    with open(json_file_path, 'w') as f:
        json.dump({"experiment": config.dict()}, f, indent=4, cls=DateTimeEncoder)

