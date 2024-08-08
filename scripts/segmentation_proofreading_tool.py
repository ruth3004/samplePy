# Proofreading segmentation

# --- Imports ---
# Standard libraries
import os
import re
from datetime import datetime
# Image processing and data handling
import numpy as np
import tifffile
import imageio.v2 as imageio
# Visualization and GUI
import napari
from napari.utils.colormaps import Colormap

# --- Data Loading ---
# Load image data
image_path = mask_plane_path
img = tifffile.imread(image_path)

# Set path to save proofreading
dir_to_save = os.path.join(trials_path, "masks")
print(f"Predictions will be saved in {dir_to_save}")

# --- Napari Viewer Setup ---
viewer = napari.Viewer()
scale = (1, 1, 1)
img_layer = viewer.add_image(img, blending="additive", contrast_limits=[0, 8000], name="img", scale=scale)

load_proofread = False
if load_proofread == True:
    proofread_path = r"C:\Users\montruth\fishPy\tests\proofreading\20231006_162735_proofreading.tif"
    proofread_data = tifffile.imread(proofread_path)
else:
    # Proofread layer for showing proofreading status
    proofread_data = np.zeros_like(img.data)

proofread_layer = viewer.add_labels(proofread_data, name='proofread', opacity=2, visible=True)

# Add relabelled stack as a label layer
label_layer = viewer.add_labels(img, opacity=0.3, name="labels", scale=scale)
label_layer.contour = 2

# Start at the beginning of the stack
viewer.dims.current_step = [0, 0, 0]


# --- Keybindings ---
@viewer.bind_key('x')
def toggle_label_visibility(viewer):
    """Toggle visibility of the label layer."""
    if "labels" in viewer.layers:
        viewer.layers["labels"].visible = not viewer.layers["labels"].visible


@viewer.bind_key('a')
def activate_erase_mode(viewer):
    """Activate erase mode for label layers."""
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Labels):
            layer.mode = 'erase'


@viewer.bind_key('s')
def activate_paint_mode(viewer):
    """Activate paint mode for label layers."""
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Labels):
            layer.mode = 'paint'


@viewer.bind_key('d')
def activate_fill_mode(viewer):
    """Activate fill mode for label layers."""
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Labels):
            layer.mode = 'fill'


@viewer.bind_key('f')
def activate_pick_and_mark_mode(viewer):
    """Activate pick mode for label layers and mark the spot with a point."""
    global last_added_point_z

    layer = viewer.layers["labels"]
    viewer.layers["labels"].visible = True
    viewer.layers.selection.active = viewer.layers['labels']
    viewer.layers["labels"].mode = 'pick'

    def on_click(layer, event):
        """Handle mouse click event to mark the spot with a point."""
        global last_added_point_z
        if event.type == 'mouse_press':
            # Get click coordinates
            coord = viewer.cursor.position
            last_added_point_z = coord[0]

            # If a points layer named 'marks' doesn't exist, create it
            if 'marks' not in viewer.layers:
                viewer.add_points(coord, name='marks', face_color='green', edge_color='white', symbol='cross', size=4,
                                  opacity=0.5)
            else:
                viewer.layers['marks'].data = np.array([coord])  # Update the existing points layer

            total_z_slices = int(viewer.dims.range[0][1])

            # Add the point at the clicked position on the 'marks' layer for each z-slice
            min_mark = int(max(0, last_added_point_z - 10))
            max_mark = int(min(total_z_slices, last_added_point_z + 10))

            for z in range(min_mark, max_mark):
                if z != int(coord[0]):
                    viewer.layers['marks'].add([z, coord[1], coord[2]])

            viewer.layers.selection.active = viewer.layers['labels']
            viewer.layers.selection.selected = [viewer.layers['img'], viewer.layers['labels'], viewer.layers['marks']]
            viewer.layers['labels'].mode = 'fill'

            # Disconnect the callback to prevent further marking until 'f' is pressed again
            layer.mouse_drag_callbacks.remove(on_click)

    # Connect the callback
    layer.mouse_drag_callbacks.append(on_click)


@viewer.bind_key('Control-f')
def activate_pick_mode(viewer):
    """Activate pick mode without marking a point."""
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Labels):
            layer.mode = 'pick'


@viewer.bind_key('r')
def toggle_proofread_visibility(viewer):
    """Toggle visibility of the proofread layer."""
    if "proofread" in viewer.layers:
        viewer.layers["proofread"].visible = not viewer.layers["proofread"].visible


@viewer.bind_key('Space', overwrite=True)
def transfer_proofread_label_and_remove_points(viewer):
    """Transfer the proofread label and remove points."""
    global proofread_data
    # Get the currently selected label
    selected_label = label_layer.selected_label
    print(f"selected label: {selected_label}")

    # Set the currently selected label in the copied_data
    proofread_data[label_layer.data == selected_label] = selected_label

    # 3. Refresh the new_label_layer to reflect the changes
    proofread_layer.data = proofread_data

    proofread_layer.refresh()
    print(f"ROIs proofread: {len(np.unique(viewer.layers['proofread'].data))}")


@viewer.bind_key('q')
def delete_selected_label(viewer):
    """Delete the selected label in the active layer."""
    layer_active = viewer.layers.selection.active
    print(layer_active)
    selected_label = layer_active.selected_label
    if selected_label != 0:  # Ensure you're not deleting the background
        layer_active.data[layer_active.data == selected_label] = 0
        layer_active.refresh()


@viewer.bind_key("l")
def save_proofread_labels(viewer):
    """Save proofread labels to disk."""
    # Construct the filename
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    proofreading_filename = f"{current_time_str}_proofreading.tif"
    profreading_save_path = os.path.join(dir_to_save, proofreading_filename)
    labels_filename = f"{current_time_str}_labels.tif"
    labels_save_path = os.path.join(dir_to_save, labels_filename)
    # Save
    tifffile.imwrite(profreading_save_path, proofread_layer.data)
    tifffile.imwrite(labels_save_path, label_layer.data)
    print(f"Saved file {labels_save_path}")
    print(f"Saved file {profreading_save_path}")


@viewer.bind_key('c')
def connect_labels(viewer):
    """Connect two selected labels in the active layer."""
    active_layer = viewer.layers.selection.active
    if isinstance(active_layer, napari.layers.Labels):
        labels = list(active_layer.selected_label)
        if len(labels) == 2:
            active_layer.data[active_layer.data == labels[1]] = labels[0]
            active_layer.refresh()


@viewer.bind_key('Ctrl-s')
def split_labels(viewer):
    """Split the selected label into two new labels in the active layer."""
    active_layer = viewer.layers.selection.active
    if isinstance(active_layer, napari.layers.Labels):
        selected_label = active_layer.selected_label
        if selected_label != 0:
            # Generate a new label ID
            new_label = active_layer.data.max() + 1
            # Logic to split the label can be customized; for simplicity, we'll use a thresholding approach here.
            mask = active_layer.data == selected_label
            split_mask = mask & (np.random.rand(*mask.shape) > 0.5)
            active_layer.data[split_mask] = new_label
            active_layer.refresh()


# Handling shifts in z
@viewer.bind_key('Ctrl-z')
def handle_shifts_in_z(viewer):
    """Handle shifts in z and align the labels accordingly."""
    active_layer = viewer.layers.selection.active
    if isinstance(active_layer, napari.layers.Labels):
        data = active_layer.data
        for i in range(1, data.shape[0]):
            previous_slice = data[i - 1]
            current_slice = data[i]
            unique_labels = np.unique(previous_slice)
            for label in unique_labels:
                if label != 0:
                    shift = np.mean(
                        np.argwhere(current_slice == label)[:, 0] - np.argwhere(previous_slice == label)[:, 0])
                    if np.abs(shift) > 1:  # Apply shift threshold to adjust labels
                        data[i] = np.roll(data[i], int(-shift), axis=0)
        active_layer.data = data
        active_layer.refresh()


# --- Start the Napari Viewer ---
viewer.show()