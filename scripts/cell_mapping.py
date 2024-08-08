# Function to add EM labels (to be used after EM segmentation)
import numpy as np
import h5py


# Function to update cell mapping
def update_cell_mapping(cell_mapping, neuron_id, modality, label, plane=None):
    if neuron_id not in cell_mapping:
        cell_mapping[neuron_id] = {
            'lm_plane_labels': {},
            'lm_stack_label': None,
            'em_label': None,
            'in_lm_planes': [],
            'in_lm_stack': False,
            'in_em': False
        }

    if modality == 'lm_plane':
        cell_mapping[neuron_id]['lm_plane_labels'][plane] = label
        if plane not in cell_mapping[neuron_id]['in_lm_planes']:
            cell_mapping[neuron_id]['in_lm_planes'].append(plane)
    elif modality == 'lm_stack':
        cell_mapping[neuron_id]['lm_stack_label'] = label
        cell_mapping[neuron_id]['in_lm_stack'] = True
    elif modality == 'em':
        cell_mapping[neuron_id]['em_label'] = label
        cell_mapping[neuron_id]['in_em'] = True

# Function to add EM labels (to be used after EM segmentation)
def add_em_labels(hdf5_file_path, sample_id, em_labels):
    with h5py.File(hdf5_file_path, 'r+') as f:
        mapping_grp = f[sample_id]['cell_mapping']
        mapping_grp.create_dataset('em_labels', data=np.array(em_labels))

# Functions to query the mapping
def get_neuron_info(hdf5_file_path, sample_id, neuron_id):
    with h5py.File(hdf5_file_path, 'r') as f:
        mapping_grp = f[sample_id]['cell_mapping']
        index = np.where(mapping_grp['neuron_ids'][:].astype(str) == neuron_id)[0][0]
        return {
            'lm_label': mapping_grp['lm_labels'][index],
            'plane': mapping_grp['planes'][index],
            'em_label': mapping_grp['em_labels'][index] if 'em_labels' in mapping_grp else None
        }

def get_neurons_by_plane(hdf5_file_path, sample_id, plane):
    with h5py.File(hdf5_file_path, 'r') as f:
        mapping_grp = f[sample_id]['cell_mapping']
        indices = np.where(mapping_grp['planes'][:] == plane)[0]
        return mapping_grp['neuron_ids'][indices].astype(str)