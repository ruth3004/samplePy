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
    pulse_delay_s: Optional[int] = None
    trial_interval_s: Optional[int] = None
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
    shutter_delay_frames: Optional[int] = 40
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
    fixation_protocol: Optional[str] = None
    embedding_protocol: Optional[str] = None
    acquisition_completed: Optional[bool] = None
    acquisition_resolution_zyx: Optional[Tuple[int, int, int]] = None

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
    # TODO: consider using confi.model_dump_json()
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
    """
    Recursively print all fields of a Pydantic model, including non-annotated fields.
    """
    if isinstance(model, BaseModel):
        # Get all fields including those added dynamically
        for field_name, field_value in model:
            print(' ' * indent + f'{field_name}: {type(field_value).__name__}')
            if isinstance(field_value, BaseModel):
                tree(field_value, indent + 4)
            elif isinstance(field_value, list) and len(field_value) > 0:
                if isinstance(field_value[0], BaseModel):
                    for item in field_value:
                        tree(item, indent + 4)
                else:
                    print(' ' * (indent + 4) + f'{field_value}')
            elif isinstance(field_value, dict):
                for key, value in field_value.items():
                    print(' ' * (indent + 4) + f'{key}: {type(value).__name__}')
            else:
                print(' ' * (indent + 4) + f'{field_value}')
    else:
        print(' ' * indent + f'{model}: {type(model).__name__}')


