# segmentation_proofreading.py

import os
import napari
import numpy as np
from tifffile import imread, imwrite
from scripts.sample_db import SampleDB

def proofread_segmentation(sample_id, db_path):
    # Load the sample database
    sample_db = SampleDB()
    sample_db.load(db_path)

    # Load experiment
    exp = sample_db.get_sample(sample_id)
    print(f"Proofreading segmentation for sample: {exp.sample.id}")

    # Load masks
    masks_folder = os.path.join(exp.paths.trials_path, "masks")
    masks_file = [f for f in os.listdir(masks_folder) if f.startswith(f'masks_{exp.sample.id}')][0]
    masks_path = os.path.join(masks_folder, masks_file)
    masks = imread(masks_path)

    # Load original images
    processed_folder = os.path.join(exp.paths.trials_path, 'processed')
    images_path = [f for f in os.listdir(processed_folder) if f.startswith('sum_elastic_')][0]
    images_path = os.path.join(processed_folder, images_path)
    images = imread(images_path)

    # Create Napari viewer
    viewer = napari.Viewer()
    viewer.add_image(images, name='Original')
    labels_layer = viewer.add_labels(masks, name='Masks')

    # Define proofreading functions
    @viewer.bind_key('d')
    def delete_selected_label(viewer):
        selected_label = labels_layer.selected_label
        if selected_label != 0:
            labels_layer.data[labels_layer.data == selected_label] = 0
            labels_layer.refresh()

    @viewer.bind_key('m')
    def merge_selected_labels(viewer):
        selected_labels = list(labels_layer.selected_label)
        if len(selected_labels) > 1:
            for label in selected_labels[1:]:
                labels_layer.data[labels_layer.data == label] = selected_labels[0]
            labels_layer.refresh()

    @viewer.bind_key('s')
    def split_selected_label(viewer):
        selected_label = labels_layer.selected_label
        if selected_label != 0:
            new_label = labels_layer.data.max() + 1
            mask = labels_layer.data == selected_label
            split_mask = mask & (np.random.rand(*mask.shape) > 0.5)
            labels_layer.data[split_mask] = new_label
            labels_layer.refresh()

    @viewer.bind_key('c')
    def create_new_label(viewer):
        new_label = labels_layer.data.max() + 1
        labels_layer.mode = 'paint'
        labels_layer.selected_label = new_label

    @viewer.bind_key('l')
    def save_proofreading(viewer):
        save_path = os.path.join(masks_folder, f'proofread_masks_{exp.sample.id}.tif')
        imwrite(save_path, labels_layer.data)
        print(f"Saved proofread masks to: {save_path}")

    napari.run()

if __name__ == "__main__":
    sample_id = '20220118_RM0012_124hpf_fP8_f2'  # Replace with your sample ID
    db_path = r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\sample_db.csv'
    proofread_segmentation(sample_id, db_path)