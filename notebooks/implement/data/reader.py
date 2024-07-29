### Contains functions or classes to read different types of data (calcium imaging, anatomy stacks, ROIs, electron microscopy, etc. also within the sample struct)."""

''' IMAGE READER/loader '''

from PIL import Image
import tifffile
import glob

import os
from pathlib import Path

import numpy as np

from scipy import io
import pandas as pd

import skimage.io as skio #for tiff files processing
from utils.mat_loader import *
from math import ceil
class DataReader:
    def load_anatomy_stack(anatomy_stack_path, n_channels, channel_num):
        """
        Load anatomy stack from a specific channel.

        Parameters:
        - anatomy_stack_path (str): Path to the anatomy stack image.
        - n_channels (int): Total number of channels in the stack.
        - channel_num (int): Channel number to be loaded (0-based).

        Returns:
        - numpy.ndarray: 3D array containing the anatomy stack for the specified channel in (slice,width,height) or(z,y,x) shape
        """
        anatomy_stack = Image.open(anatomy_stack_path)
        n_frames = anatomy_stack.n_frames
        channel_stack = np.zeros((n_frames // n_channels, anatomy_stack.width, anatomy_stack.height))

        # Extract frames belonging to the specified channel
        for k in range(0, n_frames, n_channels):
            anatomy_stack.seek(k + channel_num)
            frame = np.array(anatomy_stack)
            channel_stack[k // n_channels, :, :] = frame
        return channel_stack


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

    def load_planes_from_anatomy_folder(self,anatomy_dir, n_planes, doubling=True):
        """
        Load planes from a directory, considering the doubling factor.

        Parameters:
        - anatomy_dir (str): Directory containing plane images.
        - n_planes (int): Total number of planes.
        - doubling (bool): Whether to use a doubling factor when reading planes.

        Returns:
        - numpy.ndarray: 3D array containing the loaded planes in (slice,width,height) or(z,y,x) shape.
        """
        doubling = 2 if doubling else 1

        for plane in range(n_planes):
            plane_path = os.path.join(anatomy_dir, "plane0" + str(plane + 1))
            list_img = os.listdir(plane_path)
            plane_img = np.array(Image.open(os.path.join(plane_path, list_img[2])))

            # Initialize planes_stack based on the size of the first plane image
            if plane == 0:
                planes_stack = np.zeros((n_planes * doubling, plane_img.shape[0] // doubling, plane_img.shape[1]))

                # Split and assign the plane image based on the doubling factor
            planes_stack[2 * plane, :, :] = plane_img[:plane_img.shape[0] // 2, :]
            planes_stack[2 * plane + 1, :, :] = plane_img[plane_img.shape[0] // 2:, :]

        return planes_stack


    def load_planes_from_tif(tif_path, n_planes, doubling, ignore_frames=0):
        """
        Load planes from a (hyper)stack TIF file and reshape based on input parameters.

        Parameters:
        - tif_path (str): Path to the TIF file.
        - n_planes (int): Total number of planes.
        - doubling (int): Doubling factor.
        - ignore_frames (int, optional): Number of frames to be ignored at the start.

        Returns:
        - numpy.ndarray: 3D array containing the reshaped planes in (slice,width,height) or(z,y,x) shape.
        """
        with tifffile.TiffFile(tif_path) as tif:
            img = tif.asarray()

        doubling = 2 if doubling else 1
        hyperstack_img = img.reshape(len(img) // n_planes, n_planes, img.shape[1], img.shape[2])[ignore_frames:, :, :,
                         :].sum(axis=0)
        planes_stack = np.zeros((n_planes * doubling, hyperstack_img.shape[1] // doubling, hyperstack_img.shape[2]))

        # Split and assign the planes based on the doubling factor
        for plane in range(n_planes):
            planes_stack[2 * plane, :, :] = hyperstack_img[plane, :hyperstack_img.shape[1] // doubling, :]
            planes_stack[2 * plane + 1, :, :] = hyperstack_img[plane, hyperstack_img.shape[1] // doubling:, :]

        return planes_stack


    def load_raw_acquisition_from_tif(tif_path, n_planes, doubling):
        """
        Load raw acquisition from a (hyper)stack TIF file and reshape based on input parameters.

        Parameters:
        - tif_path (str): Path to the TIF file.
        - n_planes (int): Total number of planes.
        - doubling (boolean): Was the acquisition doubled?.

        Returns:
        - numpy.ndarray: 3D array containing the reshaped planes in (slice,width,height) or(z,y,x) shape.
        """
        with tifffile.TiffFile(tif_path) as tif:
            img = tif.asarray()

        doubling = 2 if doubling else 1

        # Reshaping stack
        hyperstack_img = img.reshape(len(img) // n_planes, n_planes, img.shape[1], img.shape[2])

        # Initiating stack (relevant if doubling)
        acquisition_stack = np.zeros(
            (len(img) // n_planes, n_planes * doubling, hyperstack_img.shape[2] // doubling, hyperstack_img.shape[3]))

        # Split and assign the planes based on the doubling factor
        for plane in range(n_planes):
            acquisition_stack[:, doubling * plane, :, :] = hyperstack_img[:, plane, :hyperstack_img.shape[2] // doubling, :]
            acquisition_stack[:, doubling * plane + 1, :, :] = hyperstack_img[:, plane,
                                                               hyperstack_img.shape[2] // doubling:, :]

        return acquisition_stack


    def split_anatomy_stack(anatomy_stack_path, n_channels=2):
        anatomy_stack = Image.open(anatomy_stack_path)
        n_frames = anatomy_stack.n_frames

        # Initialize anatomy stack of channel 1: normally green channel
        ch1_anatomy_stack = np.zeros((anatomy_stack.width, anatomy_stack.height, n_frames // n_channels))
        # Initialize anatomy stack of channel 2: normally red channel
        ch2_anatomy_stack = np.zeros_like(ch1_anatomy_stack)

        # Create anatomy stack
        for k in range(n_frames):
            anatomy_stack.seek(k)
            frame = np.array(anatomy_stack)
            if k % 2 == 0:
                ch1_anatomy_stack[:, :, k // 2] = frame
            else:
                ch2_anatomy_stack[:, :, k // 2] = frame
        return ch1_anatomy_stack, ch2_anatomy_stack


    def read_images_convert_to_stack(path, ending="*.tif"):  # Fixed to
        image_stack = []
        file_list = glob.glob(os.path.join(path, ending))

        # Extract the number from the filename, convert to int, and sort.
        file_list.sort(key=lambda f: int(os.path.basename(f).split('_')[1]))

        for file in file_list:
            img = Image.open(file)
            image_stack.append(np.array(img))

        image_stack = np.stack(image_stack)
        print(f"{len(image_stack)} files found and stacked")
        return image_stack


    def get_anatomy_image(self, plane, trial, odor):
        anatomy_filename = self.get_anatomy_filename(plane=plane, trial=trial,
                                                     odor=odor)  # self.get_param("results") / "anatomy" / "".join(["plane0",str(plane)]) / file_name
        anatomy_path = self.get_param("results") / "anatomy" / "".join(["plane0", str(plane)]) / anatomy_filename
        anatomy_image = skio.imread(anatomy_path)
        return anatomy_image



    #--------------------------------------#
    """ ADD FUNCTIONS e.g. raw traces from NeuROI"""
    # --------------------------------------#

    def add_raw_traces(self):
        """ Reads timetrace files of each plane folder and adds them as individual neurons"""
        for plane_folder in range(1, self.get_param("nPlanes")+1): # Going through folders
            print(plane_folder)
            if "traces_plane_stack" in locals():
                del traces_plane_stack
            for path in (self.info.path.relative / "results" / "time_trace" / f'plane0{plane_folder}').iterdir(): # Finding timetraces files

                if "traceResult" in path.parts[-1]:
                    traces = loadmat(path)["traceResult"]["timeTraceMat"]
                    # Getting timetraces for all ROIs of current odor. shape: roi x time
                    if "traces_plane_stack" in locals():
                        traces_plane_stack = np.concatenate((traces_plane_stack, traces[None,:200]), axis=0)
                        #traces_plane_stack = np.concatenate((traces_plane_stack, traces[None,:200]), axis=0)
                    else:
                        traces_plane_stack = traces[None,:200] #####!!!! HARDCODED
                        #traces_plane_stack = traces[None,:]
            # Concatenating all rois with their traces of all planes. shape: trial x roi x time
            if "timetraces_stack" in locals():
                timetraces_stack = np.concatenate((timetraces_stack,traces_plane_stack), axis=1)
            else:
                timetraces_stack = traces_plane_stack

        # Creating neuron field if not present
        # Changing shape of timetraces_stack to roi x time x trial e.g. (1071,375,24)
        timetraces_stack = np.moveaxis(timetraces_stack,0,-1)
        if hasattr(self, 'neuron') == False:
            df = pd.DataFrame()
        # Adding traces to dataframe column
        timetraces_series = [timetraces_stack[x, :, :] for x in range(timetraces_stack.shape[0])] #Converting to one dimensional so it can be added as a Series
        self.neuron = df.assign(raw_traces = pd.Series(timetraces_series))


    def add_rois_from_LM_experiment(self):
        """ Reads imaged rois files of each plane folder and adds them as individual neurons"""
        doubling = 2 if self.get_param("doubling") else 1
        for plane_folder in range(1,self.get_param("nPlanes")+1):
            for path in (self.info.path.results / "roi" / f'plane0{plane_folder}').iterdir(): # Finding timetraces files
                if "RoiCell" in path.parts[-1]:
                    extracted_rois = io.loadmat(path)["RoiCell"]
                    # Adding ROIs to field
                    #Stacking all tags and positions of all planes
                    extracted_tags = [int(extracted_rois[0, :][x]) for x in range(extracted_rois.shape[1])]
                    extracted_positions = [extracted_rois[1, :][x] for x in range(extracted_rois.shape[1])]
                    planes = list()
                    for pos in extracted_positions:
                        half = pos[:,1]<256 #TODO: Hardcoded! Make flexible for no doubling images
                        plane = (plane_folder*doubling-1) if half.any() else (plane_folder*doubling)
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
        self.neuron = self.neuron.assign(roi_exp_tag =  pd.Series(tags))
        self.neuron = self.neuron.assign(roi_exp_position = pd.Series(positions))
        self.neuron = self.neuron.assign(roi_exp_plane = pd.Series(plane_acquired))

    def add_rois_from_LM_anatomy(self):
        """ Reads LM rois of anatomy image and maps them to imaged rois """
        pass

    def add_rois_from_EM_anatomy(self):
        pass



    # --------------------------------------#
    """ UPDATE FUNCTIONS"""

    # --------------------------------------#
    def update_results_path(self):
        for path in Path(self.info.path.results).iterdir():
            if path.is_dir():
                print('Folders updated:')
                folder_name = path.parts[-1]
                print(folder_name)
                # self.info.path.info["folder_name"] =
                # TODO: think again how to add the foldering


    def update_roi_from_neuroi(self):
        # TODO
        pass
