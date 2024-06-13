import json
from pydantic import BaseModel, Field, DirectoryPath, FilePath
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from pathlib import Path


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

    class Config:
        extra = 'allow'
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class ParamsLM(BaseModel):
    start_time: datetime  # Including both date and time in the format "yyyy-mm-ddTHH:MM:SS"
    end_time: datetime  # Including both date and time in the format "yyyy-mm-ddTHH:MM:SS"
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

    class Config:
        extra = 'allow'
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class ParamsEM(BaseModel):
    fixation_protocol: str
    embedding_protocol: str
    acquisition_completed: bool
    acquisition_resolution_zyx: Tuple[int, int, int]

    class Config:
        extra = 'allow'
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class Paths(BaseModel):
    root_path: DirectoryPath = Field(default_factory=DirectoryPath)
    config_path: FilePath = Field(default_factory=FilePath)
    trials_path: DirectoryPath = Field(default_factory=DirectoryPath)
    anatomy_path: DirectoryPath = Field(default_factory=DirectoryPath)
    em_path: DirectoryPath = Field(default_factory=DirectoryPath)

    class Config:
        extra = 'allow'
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class Experiment(BaseModel):
    sample: Sample
    params_odor: Optional[ParamsOdor]
    params_lm: Optional[ParamsLM]
    params_em: Optional[ParamsEM]
    paths: Paths = Field(default_factory=Paths)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        extra = 'allow'


def save_experiment_config(config: Experiment, json_file_path: str = ""):
    config_dict = config.dict()
    # Convert all Path objects to strings
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, Path):
                    value[sub_key] = str(sub_value)

    if json_file_path == "":
        json_file_path = str(config.paths.config_path)
        print(json_file_path)

    with open(json_file_path, 'w') as f:
        json.dump({"experiment": config_dict}, f, indent=4, cls=DateTimeEncoder)


def load_experiment_config(json_file_path: str) -> Experiment:
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        if 'experiment' in data:
            experiment_data = data['experiment']
        else:
            experiment_data = data

        # Convert string paths back to Path objects
        if 'paths' in experiment_data:
            for key, value in experiment_data['paths'].items():
                experiment_data['paths'][key] = Path(value)

        return Experiment(**experiment_data)


def update_experiment_config(config: Experiment, changes: Dict[str, Any]) -> Experiment:
    config_dict = config.dict()
    for key, value in changes.items():
        if isinstance(value, dict) and key in config_dict:
            config_dict[key].update(value)
        else:
            config_dict[key] = value
    return Experiment(**config_dict)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)


# Function to print tree of Pydantic model
def tree(model: Any, indent: int = 0):
    for field_name, field_type in model.__annotations__.items():
        print(' ' * indent + f'{field_name}: {field_type}')
        field_value = getattr(model, field_name, None)
        if isinstance(field_value, BaseModel):
            tree(field_value, indent + 4)
        elif isinstance(field_value, list) and len(field_value) > 0 and isinstance(field_value[0], BaseModel):
            tree(field_value[0], indent + 4)

