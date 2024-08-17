# parameter_testing.py

import os
import glob
import numpy as np
import napari
from tifffile import imread
from cellpose import models
from itertools import product
from scripts.sample_db import SampleDB

def test_parameters(sample_id, db_path, model_path, parameter_ranges):
    # Load the sample database
    sample_db = SampleDB()
    sample_db.load(db_path)

    # Load experiment
    exp = sample_db.get_sample(sample_id)
    print(f"Testing parameters for sample: {exp.sample.id}")

    # Load model
    model = models.CellposeModel(model_type=model_path, gpu=True)

    # Load image stack
    processed_folder = os.path.join(exp.paths.trials_path, 'processed')
    images_path = glob.glob(os.path.join(processed_folder, 'sum_elastic_*.tif'))[0]
    images_stack = imread(images_path)

    # Generate all combinations of parameters
    parameter_combinations = list(product(*parameter_ranges.values()))
    print(f"Number of combinations to test: {len(parameter_combinations)}")

    # Create Napari viewer
    viewer = napari.Viewer()
    viewer.add_image(images_stack, name='Original')

    # Test plane (you can modify this)
    test_plane = 3
    images = images_stack[test_plane]

    for idx, params in enumerate(parameter_combinations):
        param_dict = dict(zip(parameter_ranges.keys(), params))
        params_text = '-'.join([f"{k}_{v}" for k, v in param_dict.items()])
        print(f"Testing combination {idx + 1}/{len(parameter_combinations)}: {params_text}")

        # Segment the images using Cellpose with current parameter combination
        masks, _, _ = model.eval(images,
                                 channels=[0, 0],
                                 **param_dict)

        # Add the masks to Napari viewer
        viewer.add_labels(masks, name=params_text)

    napari.run()

if __name__ == "__main__":
    sample_id = '20220118_RM0012_124hpf_fP8_f2'  # Replace with your sample ID
    db_path = r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\sample_db.csv'
    model_path = r'D:\montruth\cellpose\models\CP_20230803_101131'

    parameter_ranges = {
        'cellprob_threshold': [-3, -2, -1],
        'flow_threshold': [0, 0.4, 0.8],
        'resample': [True, False],
        'augment': [False, True],
        'stitch_threshold': [0.01, 0.05, 0.1]
    }

    test_parameters(sample_id, db_path, model_path, parameter_ranges)