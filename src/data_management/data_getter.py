import os
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional


class DataGetter:
    def get_param(self, arg: str) -> Any:
        """ Search parameter across Sample structure and return its value with recursion for nested SimpleNamespace. """

        def recursive_search(obj: SimpleNamespace) -> Optional[Any]:
            # If the current object has the target attribute directly
            if hasattr(obj, arg):
                return getattr(obj, arg)
            # If not, iterate over all attributes that might be SimpleNamespace or dicts
            for attr in vars(obj).values():
                if isinstance(attr, SimpleNamespace) or isinstance(attr, dict):
                    result = recursive_search(attr)
                    if result is not None:
                        return result

        # Start recursive search from self, assuming the entire object is a SimpleNamespace
        return recursive_search(self)

    def get_baseline(self) -> range:
        """Returns time window of baseline in frames dynamically."""
        baseline_start = self.get_param("baseline_start") or 50  # Fetch from data or use default
        baseline_end = self.get_param("baseline_end") or 120
        return range(baseline_start, baseline_end + 1)

    def get_trial_index(self, odors: Optional[List[int]] = None, trials: Optional[List[int]] = None) -> List[int]:
        """ Returns the trial index given the trials (list) for specific odors (list) """
        nOdors = len(self.get_param("odorList"))
        odors = odors if odors is not None else list(range(nOdors))
        trials = trials if trials is not None else list(range(self.get_param("nTrials")))

        return [od + (trial * nOdors) for trial in trials for od in odors]

    def get_traces(self, n_ID: Optional[List[int]] = None, time_window: Optional[range] = None,
                   odors: Optional[List[int]] = None, trials: Optional[List[int]] = None,
                   trace: str = 'raw', aligned: bool = True) -> np.ndarray:
        """ Returns numpy array of traces at defined time window (in frames), for neurons (ID), odors,
            trials and type of traces (raw,df,...) """
        n_ID = n_ID or self.get_default("n_ID")
        time_window = time_window or self.get_default("time_window")
        odors = odors or self.get_default("odors")
        trials = trials or self.get_default("trials")

        selected_trials = self.get_trial_index(odors, trials)
        all_traces = np.array(self.neuron[trace + '_traces'].tolist())

        if aligned and hasattr(self, 'paramsOdor') and 'odorStart' in self.paramsOdor:
            shifts = self.paramsOdor['odorStart'] - np.max(self.paramsOdor['odorStart'])
            for t in selected_trials:
                all_traces[:, :, t] = np.roll(all_traces[:, :, t], -int(shifts[0, t]), axis=1)

        return all_traces[n_ID, :, :][:, time_window[0]: time_window[-1] + 1, :][:, :, selected_trials]

    def get_odor_name(self, odor: int) -> str:
        """Returns the odor name as string from index"""
        odor_list = self.get_param("odorList")
        return odor_list[odor % len(odor_list)]

    def get_default(self, param: str) -> Any:
        """Provides default values for various parameters"""
        defaults = {
            "time_window": range(0, (self.get_param("nFrames") // self.get_param("nPlanes"))),
            "odors": list(range(len(self.get_param("odorList")))),
            "trials": list(range(self.get_param("nTrials"))),
            "n_ID": list(range(len(self.neuron)))
        }
        return defaults.get(param)

    def get_parent_path(self, user: str = 'montruth') -> str:
        """GET_PARENT_PATH returns the parent path until the username folder. Python compatible path."""
        current_path = os.getcwd()
        parts = current_path.split(os.sep)
        parts_to_join = parts[:parts.index(user) + 1]
        parent_folder = '/'.join(parts_to_join)
        return parent_folder

    def get_raw_acquisition_filename(self, odor: int, trial: int) -> str:
        """
        Retrieves the filename of the raw acquisition file for a given odor and trial.

        Parameters:
            odor (int): The index of the odor.
            trial (int): The index of the trial.

        Returns:
            str: The filename of the raw acquisition data file.

        Raises:
            FileNotFoundError: If no matching files are found.
        """
        # Using pathlib for path operations
        home_path = Path(self.get_param("home"))
        relative_path = Path(self.get_param("relative"))
        exp_path = home_path / relative_path

        # Get directory listing using pathlib
        list_dir_all = [x.name for x in exp_path.iterdir() if x.is_file()]

        # Filters only trial files
        filename_filter = f"{self.get_param('ID')}_t"
        filtered_files = [filename for filename in list_dir_all if filename_filter in filename]

        # Gets only selected by odor and trial
        id = self.get_trial_index(odors=[odor], trials=[trial])
        if id[0] < len(filtered_files):
            return filtered_files[id[0]]
        else:
            raise FileNotFoundError("The requested file does not exist.")

    def get_anatomy_filename(self, plane: int, trial: int, odor: int) -> str:
        """
        Retrieves the filename of the anatomy file for a given plane, trial, and odor.

        Parameters:
            plane (int): The plane number.
            trial (int): The trial number.
            odor (int): The odor number.

        Returns:
            str: The filename of the anatomy file.

        Raises:
            FileNotFoundError: If the directory does not exist or no files match the criteria.
        """
        # Build path using pathlib
        results_path = Path(self.get_param("results"))
        anatomy_path = results_path / "anatomy" / f"plane0{plane}"

        # Ensure the directory exists
        if not anatomy_path.exists() or not anatomy_path.is_dir():
            raise FileNotFoundError(f"No directory found for anatomy data at {anatomy_path}")

        # List directory content
        list_dir = [x.name for x in anatomy_path.iterdir() if x.is_file()]
        id = self.get_trial_index(odors=[odor], trials=[trial])
        if id[0] < len(list_dir):
            return list_dir[id[0]]
        else:
            raise FileNotFoundError("The requested anatomy file does not exist.")

    def get_raw_image_acquisition(self, odor, trial):
        # getting acquisition info
        nPlanes = self.get_param("nPlanes")
        doubling = 2 if self.get_param("doubling") == 1 else 1
        nFrames = self.get_param("nFrames")

        raw_image_acquisition_path = self.get_param("home") / self.get_param(
            "relative") / self.get_raw_acquisition_filename(odor, trial)
        raw_stack = skio.imread(raw_image_acquisition_path, plugin="tifffile")

        # Converting raw stack to hyperstack and separating planes if acquisition was doubled
        im_frames, im_height, im_width = np.shape(raw_stack)
        hyperstack = np.zeros((nPlanes * doubling, im_frames // nPlanes, im_height // doubling, im_width), dtype='uint16')
        for plane in range(8):
            if plane % 2 == 0 or doubling == 1:
                half = slice(0, im_height // doubling)
            else:
                half = slice(im_height // doubling, im_width)
            hyperstack[plane, :, :, :] = raw_stack[slice(ceil(plane // doubling), im_frames, nPlanes), half, :]

        return hyperstack
