from pydantic import BaseModel
from typing import List, Optional, Tuple
from datetime import datetime

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
    n_frames: int
    pulse_delay_s: int
    pulse_duration_s: int
    trial_interval_s: int
    missed_trials: List = []
    events: List[Tuple[str, datetime]] = []

class ParamsLM(BaseModel):
    start_time: datetime  # Including both date and time in the format "yyyy-mm-dd-hh:mm"
    end_time: datetime    # Including both date and time in the format "yyyy-mm-dd-hh:mm"
    zoom_x: Optional[float]
    power_percentage: Optional[float]
    shutter_delay_frames: Optional[int]
    sampling_hz: int
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
