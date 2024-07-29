import os
import numpy as np
# --------------------------------------#
""" GET FUNCTIONS FOR STRUCT"""
# --------------------------------------#
class DataGetter:
    def get_param(self, arg):
        """ Searchs parameter across Sample structure and return its value"""
        # TODO: Iterate over attributes of Sample
        # TODO: Refer to same function to go through all nested dictionary
        for id, value in self.exp.items():
            if id == arg:
                return value
            if hasattr(value, '__iter__'):
                for k in value:
                    if k == arg:
                        return value[k]

        for id, value in self.info.items():
            if id == arg:
                return value
            if hasattr(value, '__iter__'):
                for k in value:
                    if k == arg:
                        return value[k]


    def get_baseline(self):
        """Returns time window of baseline in frames"""

        return range(50, 121)  # TODO: HARDCODED

    def get_trial_index(self, odors="all", trials="all"):
        """ Returns the trial index given the trials (list) for specific odors (list) """
        nOdors = len(self.get_param("odorList"))
        if odors == "all":
            odors = [*range(0, nOdors)]
        if trials == "all":
            trials = [*range(0, self.get_param("nTrials"))]

        return [od + (trial * nOdors) for trial in trials for od in odors]


    def get_traces(self, n_ID="all", time_window='all', odors='all', trials='all', trace='raw', aligned=True):
        """ Returns numpy array of traces at defines time_window window (in frames), for neurons (ID), odors,
        trials and type of traces (raw,df,...)"""  # FIXME: for several n_IDs (maybe loop through ID list)

        # get default values
        if n_ID == "all":
            n_ID = self.get_default("n_ID")
        if time_window == "all":
            time_window = self.get_default("time_window")
        if odors == "all":
            odors = self.get_default("odors")
        if trials == "all":
            trials = self.get_default("trials")

        # get trial indices
        selected_trials = self.get_trial_index(odors=odors, trials=trials)

        # get traces
        all_traces = np.array(self.neuron[trace + '_traces'].tolist())
        if aligned == True and hasattr(self.exp.paramsOdor, "odorStart"):  # TODO: and self.exp.paramsOdor.odorStart
            shifts = self.exp.paramsOdor.odorStart - np.max(self.exp.paramsOdor.odorStart)
            for t in selected_trials:
                all_traces[:, :, t] = np.roll(all_traces[:, :, t], -int(shifts[0, t]), axis=1)

        return all_traces[n_ID, :, :][:, time_window[0]: time_window[-1] + 1, :][:, :, selected_trials]



    def get_odor_name(self, odor):
        """Returns the odor name as string from index"""
        odor_list = self.get_param("odorList")
        return odor_list[odor % len(odor_list)]


    def get_neuron_id(self, n_ID):  # TODO
        """Returns from unique ID, the index in DataFrame"""
        pass


    def get_raw_acquisition_filename(self, odor, trial):  # TODO: get image file names elsewhere (from neuROI)
        """Reads raw experiment names by using a string filter"""

        # Extract files from experiment path
        exp_path = self.get_param("home") / self.get_param("relative")
        list_dir_all = [x for x in os.listdir(exp_path)]

        # Filters only trial files
        filename_filter = "_".join([self.get_param("ID"), "t"])
        list_dir = [x for x in list_dir_all if filename_filter in x]  # Getting experiment names

        # Gets only selected by odor and trial
        id = self.get_trial_index(odors=[odor], trials=[trial])

        return list_dir[id[0]]


    def get_anatomy_filename(self, plane, trial, odor):
        anatomy_path = self.get_param("results") / "anatomy" / "".join(["plane0", str(plane)])
        id = self.get_trial_index(odors=[odor], trials=[trial])
        # id = id[0] -1
        list_dir = [x for x in os.listdir(anatomy_path)]
        return list_dir[id[0]]


    def get_default(self, param):
        return {
            "time_window": [*range(0, (self.get_param("nFrames") // self.get_param("nPlanes")))],
            "odors": [*range(0, len(self.get_param("odorList")))],
            "trials": [*range(0, self.get_param("nTrials"))],
            "n_ID": [*range(0, len(self.neuron))]
        }.get(param)

    def get_parent_path(user='montruth'):
        """
        GET_PARENT_PATH returns the parent path until the username folder. Python compatible path.
        OPTIONAL: user: str username
        """
        current_path = os.getcwd()
        parts = current_path.split('\\')
        parts_to_join = parts[:parts.index(user)+1]
        parent_folder = '/'.join(parts_to_join)
        return parent_folder


