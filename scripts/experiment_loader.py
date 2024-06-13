## TODO: OBSOLETE DELETE

import json

from scripts.config_model import Experiment
import xarray as xr
import numpy as np




def load_experiment_config(json_file_path: str) -> Experiment:
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return Experiment(**data['experiment'])

def save_experiment_config(config: Experiment, json_file_path: str):
    with open(json_file_path, 'w') as f:
        json.dump({"experiment": config.dict()}, f, indent=4, cls=DateTimeEncoder)


def create_fluorescence_dataset(roi_id: int, intensities: np.ndarray, frame_numbers: np.ndarray, time_seconds: np.ndarray, stimulus: np.ndarray, metadata: dict) -> xr.Dataset:
    data_vars = {
        'fluorescence': (('frame',), intensities)
    }
    coords = {
        'frame': frame_numbers,
        'time_s': time_seconds,
        'stimulus': ('frame', stimulus)
    }
    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs.update(metadata)
    ds.attrs['roi_id'] = roi_id
    return ds


def create_experiment_dataset(experiment: Experiment, roi_id: int, intensities: np.ndarray) -> xr.Dataset:
    frame_rate = experiment.params_lm.sampling_hz
    n_frames = experiment.params_lm.n_frames
    frame_numbers = np.arange(n_frames)
    time_seconds = frame_numbers / frame_rate

    # Create the stimulus binary array (stimulus starts at 10s and lasts for 5s)
    stim_start_frame = int(10 * frame_rate)
    stim_end_frame = int(15 * frame_rate)
    stim_array = np.zeros(n_frames, dtype=int)
    stim_array[stim_start_frame:stim_end_frame] = 1

    # Generate metadata dictionary from Experiment
    metadata = {
        'sample_id': experiment.sample.id,
    }

    return create_fluorescence_dataset(roi_id, intensities, frame_numbers, time_seconds, stim_array, metadata)
