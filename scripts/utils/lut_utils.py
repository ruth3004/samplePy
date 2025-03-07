import os
import time
from tqdm import tqdm

# Third-party imports
import numpy as np
import h5py
import tifffile
from google.auth import exceptions

# Custom imports
from brainmaps_api_fcn.equivalence_requests import EquivalenceRequests
from brainmaps_api_fcn.subvolume_requests import SubvolumeRequest


def get_agglo_group_with_retry(sa, volume_id, stack_change, centroid_xyz, max_retries=5):
    """Get agglomeration group with retry mechanism."""
    for attempt in range(max_retries):
        try:
            sr = SubvolumeRequest(sa, volume_id)
            vol = sr.get_subvolume(centroid_xyz, size=[1, 1, 1], change_stack_id=stack_change)
            agglo_id = int(np.unique(vol[vol > 0])[0])

            er = EquivalenceRequests(sa, volume_id, stack_change)
            return er.get_groups(agglo_id)
        except exceptions.RefreshError:
            if attempt == max_retries - 1:
                raise
            print(f"Try {attempt + 1}/{max_retries}")
            time.sleep(attempt)


def get_neuron_segments(lut_path, neuron_id):
    """Get all segment IDs for a given neuron"""
    with h5py.File(lut_path, 'r') as f:
        neuron_path = f'neurons/neuron_{neuron_id}'
        if neuron_path not in f:
            return []

        # Get all segment IDs from the segments group
        segments = f[neuron_path]['agglo_segments'][:]
        # Convert segment IDs from strings to integers

        return segments


def get_segments_by_agglo_id(lut_path, agglo_id):
    """Get all segment IDs for a given agglomeration ID"""
    segments = []
    with h5py.File(lut_path, 'r') as f:
        # Iterate through all neurons to find matching agglo_id
        for neuron_name, neuron_group in f['neurons'].items():
            if 'agglo_id' in neuron_group.attrs:
                if neuron_group.attrs['agglo_id'] == agglo_id:
                    # Get segments from this neuron
                    if 'agglo_segments' in neuron_group:
                        segments = neuron_group['agglo_segments'][:]
                        break
    return segments


def find_neurons_in_mask(lut_path, mask_path):
    """Find neurons with centroids inside a 3d mask"""
    # Open the Paintera zarr mask
    mask = tifffile.imread(mask_path, mode='r')
    neurons_inside = []

    with h5py.File(lut_path, 'r') as f:
        for neuron_name, neuron_group in f['neurons'].items():
            # Get the neuroglancer coordinates of the centroid
            # We use ng coordinates since Paintera mask is in the same space
            centroid = neuron_group['em_centroid_ng'][:] // 16

            # print(centroid)

            # Round coordinates to integers for indexing
            x, y, z = np.round(centroid).astype(int)

            # Check if the centroid is within mask bounds
            if (0 <= x < mask.shape[2] and
                    0 <= y < mask.shape[1] and
                    0 <= z < mask.shape[0]):

                # Check if the point is inside the mask (non-zero value)
                if mask[z, y, x] > 0:
                    neuron_id = int(neuron_name.split('_')[1])
                    agglo_id = neuron_group.attrs.get('agglo_id', None)
                    neurons_inside.append({
                        'neuron_id': neuron_id,
                        'agglo_id': agglo_id,
                        'centroid_zyx': centroid[::-1]
                    })

    return neurons_inside


def get_neurons_with_attribute(lut_path, attribute_name, attribute_value, operator="=="):
    """
    Get neurons where attribute matches the comparison with threshold

    Args:
        lut_path: Path to HDF5 file
        attribute_name: Name of attribute to check
        operator: String specifying comparison ('>', '<', '>=', '<=', '==')
        threshold: Value to compare against
    """
    neuron_with_attribute = []
    operators = {
        '>': lambda x, y: x > y,
        '<': lambda x, y: x < y,
        '>=': lambda x, y: x >= y,
        '<=': lambda x, y: x <= y,
        '==': lambda x, y: x == y
    }

    if operator not in operators:
        raise ValueError(f"Operator must be one of {list(operators.keys())}")

    with h5py.File(lut_path, 'r') as f:
        for neuron, neuron_group in f['neurons'].items():
            if attribute_name in neuron_group.attrs:
                stored_value = neuron_group.attrs[attribute_name]
                if operators[operator](stored_value, attribute_value) and "agglo_id" in neuron_group.attrs:
                    # print(stored_value)
                    neuron_with_attribute.append(neuron_group.attrs["agglo_id"])
                    # print(f"Neuron {neuron}: {attribute_name} = {stored_value}")

    return neuron_with_attribute


def extract_dataset(h5_path, group_path):
    """Extract datasets and attributes from HDF5 group into a dictionary"""
    extracted_data = {}

    with h5py.File(h5_path, 'r') as f:
        group = f[group_path]

        # Get datasets
        for key, val in group.items():
            extracted_data[key] = val[()]

        # Get attributes
        for key, val in group.attrs.items():
            extracted_data[key] = val

    return extracted_data


def apply_transformation(points, transform):
    # Convert to homogeneous coordinates
    homogeneous_points = np.column_stack((points, np.ones(len(points))))
    # Apply transformation
    transformed_points = homogeneous_points @ transform.T
    # Return to 3D coordinates
    return transformed_points[:, :3]


# Function to check if a point is within the 3D mask
def is_within_mask(point, mask):
    z, y, x = np.round(point).astype(int)
    if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
        return mask[z, y, x]
    return False



