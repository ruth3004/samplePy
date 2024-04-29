# Solution to import nested MATLAB structs into Python
# from https://devpress.csdn.net/python/630452e07e66823466199d04.html

import scipy.io as spio

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


# --------------------------------------#
""" ADD FUNCTIONS e.g. raw traces from NeuROI"""


# --------------------------------------#

def add_raw_traces(self):
    """ Reads timetrace files of each plane folder and adds them as individual neurons"""
    for plane_folder in range(1, self.get_param("nPlanes") + 1):  # Going through folders
        print(plane_folder)
        if "traces_plane_stack" in locals():
            del traces_plane_stack
        for path in (
                self.info.path.relative / "results" / "time_trace" / f'plane0{plane_folder}').iterdir():  # Finding timetraces files

            if "traceResult" in path.parts[-1]:
                traces = loadmat(path)["traceResult"]["timeTraceMat"]
                # Getting timetraces for all ROIs of current odor. shape: roi x time
                if "traces_plane_stack" in locals():
                    traces_plane_stack = np.concatenate((traces_plane_stack, traces[None, :200]), axis=0)
                    # traces_plane_stack = np.concatenate((traces_plane_stack, traces[None,:200]), axis=0)
                else:
                    traces_plane_stack = traces[None, :200]  #####!!!! HARDCODED
                    # traces_plane_stack = traces[None,:]
        # Concatenating all rois with their traces of all planes. shape: trial x roi x time
        if "timetraces_stack" in locals():
            timetraces_stack = np.concatenate((timetraces_stack, traces_plane_stack), axis=1)
        else:
            timetraces_stack = traces_plane_stack

    # Creating neuron field if not present
    # Changing shape of timetraces_stack to roi x time x trial e.g. (1071,375,24)
    timetraces_stack = np.moveaxis(timetraces_stack, 0, -1)
    if hasattr(self, 'neuron') == False:
        df = pd.DataFrame()
    # Adding traces to dataframe column
    timetraces_series = [timetraces_stack[x, :, :] for x in range(
        timetraces_stack.shape[0])]  # Converting to one dimensional so it can be added as a Series
    self.neuron = df.assign(raw_traces=pd.Series(timetraces_series))


def add_rois_from_LM_experiment(self):
    """ Reads imaged rois files of each plane folder and adds them as individual neurons"""
    doubling = 2 if self.get_param("doubling") else 1
    for plane_folder in range(1, self.get_param("nPlanes") + 1):
        for path in (self.info.path.results / "roi" / f'plane0{plane_folder}').iterdir():  # Finding timetraces files
            if "RoiCell" in path.parts[-1]:
                extracted_rois = io.loadmat(path)["RoiCell"]
                # Adding ROIs to field
                # Stacking all tags and positions of all planes
                extracted_tags = [int(extracted_rois[0, :][x]) for x in range(extracted_rois.shape[1])]
                extracted_positions = [extracted_rois[1, :][x] for x in range(extracted_rois.shape[1])]
                planes = list()
                for pos in extracted_positions:
                    half = pos[:, 1] < 256  # TODO: Hardcoded! Make flexible for no doubling images
                    plane = (plane_folder * doubling - 1) if half.any() else (plane_folder * doubling)
                    planes.append(plane)

                if "tags" in locals():
                    tags = tags + extracted_tags
                    positions = positions + extracted_positions
                    plane_acquired = plane_acquired + planes
                else:
                    tags = extracted_tags
                    positions = extracted_positions
                    plane_acquired = planes

    # Creating new field if not present and adding tags and positions to respective ROIs
    if hasattr(self, 'neuron') == False:
        self.neuron = pd.DataFrame()
    self.neuron = self.neuron.assign(roi_exp_tag=pd.Series(tags))
    self.neuron = self.neuron.assign(roi_exp_position=pd.Series(positions))
    self.neuron = self.neuron.assign(roi_exp_plane=pd.Series(plane_acquired))
