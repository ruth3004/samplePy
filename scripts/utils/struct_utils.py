import numpy as np
import pickle
import re


# TODO: extract folder structure from experiment name
def get_path_from_experiment_name(experiment_name):
    pass


# Useful tools
def df2arr(df):
    # convert Series or DataFrame into numpy array
    return np.array(df.tolist())