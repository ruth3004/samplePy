### Contains functions or classes for data processing tasks (calculating df/F, baseline, etc.).
###
# --------------------------------------#
""" CALCULATE FUNCTIONS"""
# --------------------------------------#

from data.getter import *

class DataProcessor:

    # Calculate functions
    def calculate_dFoverF_traces(self, baseline_window):

        # Get values
        raw_traces = self.get_traces()
        raw_bl_mean = self.get_traces(time_window=baseline_window).mean(axis=1).tolist()
        dig_err = np.array(self.neuron["dig_err"].tolist())

        # Calculate df/f
        df_traces = (np.moveaxis(raw_traces, 1, 0) - raw_bl_mean) / (raw_bl_mean - dig_err)
        df_traces = np.moveaxis(df_traces, 0, 1)

        # Add to struct
        self.neuron["df_traces"] = df_traces.tolist()

    def calculate_shutter_off_background(self, trials="all", time_window=[0, 6]):

        if trials == "all":
            trials = self.get_default("trials")

        # Calculate mean at shutter off time. Selecting only those trials that shutter was off.
        dig_err = self.get_traces(trials=trials, time_window=time_window, aligned=False).mean(axis=1)

        # if only one trial was selected
        if len(trials) == 1:
            dig_err = np.tile(dig_err, (1, 3))

        # to add each row of 2d numpy array to each cell of a column
        self.neuron['dig_err'] = dig_err.tolist()
        self.neuron["dig_err"]

    def characterize_response(self):
        pass