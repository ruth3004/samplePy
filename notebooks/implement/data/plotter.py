#--------------------------------------#
""" PLOT FUNCTIONS""" # TODO: make plot functions to scripts
# --------------------------------------#

import matplotlib.pyplot as plt
import numpy as np

import skimage.util
import skimage

import seaborn as sns
import seaborn_image as isns

class DataPlotter:


    def plot_traces(self, time_window = 'all', n_ID = 'all', odors = 'all', trials = 'all', trace='raw'):
        """ Plot traces (raw, dF, spikes) at defined time, roi, odor and trial
        Useful for ploting few selected traces """
        if time_window == "all":
            time_window = self.get_default("time_window")
        else:
            time_window = np.arange(time_window[0],time_window[1])
        if n_ID == "all":
            n_ID = self.get_default("n_ID")

        if odors == "all":
            odors = self.get_default("odors")

        if trials == "all":
            trials = self.get_default("trials")

        data = self.get_traces(time_window =time_window, n_ID = n_ID, odors =odors,trials =trials,trace=trace)
        selected_trials = self.get_trial_index(odors= odors, trials=trials)
        # Plot
        fig, ax = plt.subplots()
        for id, n in enumerate(n_ID):
            for idt, trial in enumerate(selected_trials):
                plt.plot(time_window, data[id, :,idt], label= "".join(["nID: "+str(n) + "_od: "+self.get_odor_name(trial)+"_t:_"+str(trial)]))
            plt.legend()
        return fig, ax


    def plot_heatmap(self, time_window = 'all', n_ID = 'all', odors = 'all', trials = 'all', trace='raw'):
        """ PLots heatmap of reponse at specified time_window [start, end], of neurons (list)
        for defined odors (list), trials (list) and type of data ("raw", "df", "spikes") """
        #Get traces data
        data = self.get_traces(time_window= time_window, n_ID= n_ID, odors= odors, trials=trials, trace=trace)

        #Get default values
        if time_window == "all":
            time_window = self.get_default("time_window")
        else:
            time_window = np.arange(time_window[0],time_window[-1])
        if n_ID == "all":
            n_ID = self.get_default("n_ID")
        if odors == "all":
            odors = self.get_default("odors")
        if trials == "all":
            trials = self.get_default("trials")
        #Plot heatmap on subplots
        fig, axes = plt.subplots(nrows=len(trials), ncols=len(odors)+1, sharex=True, sharey=True)

        ax_first =  axes[0] if len(trials) == 1 | len(odors) == 1 else axes[0, 0]
        ax_last  =  axes[-1] if len(trials) == 1 | len(odors) == 1 else axes[-1, -1]
        for oo, odor in enumerate(odors):
            for tt, trial in enumerate(trials):
                trial_data = data[:, :, trial].squeeze()
                ax =  axes[oo] if len(trials) == 1  else axes[tt] if len(odors) == 1 else axes[tt, oo]

                sns.heatmap(ax= ax, data= trial_data, cbar= False)
        for row in range(axes.shape[0]):
            axes[row, 0].set(ylabel="ROI nr.")

        for col in range(axes.shape[1]):
            axes[-1, col].set(xlabel="Frames")

        # cb = fig.add_subplot(len(trials), len(odors)+1, len(trials)*(len(odors)+1))

        return fig, axes

    def get_response_map(self, time_window, odor, trial):
        """ Plots response map at a defined time_window window (list: [start, end] in frames)
        of all the planes from sample"""

        nPlanes = self.get_param("nPlanes")
        time = slice(time_window[0],time_window[1])  # in frames
        hyperstack = self.get_raw_image_acquisition(odor, trial)
        response_map = skimage.util.montage(
            [hyperstack[x, time, :, :].sum(axis=0) for x in list(range(np.shape(hyperstack)[0]))],
            grid_shape=(nPlanes, 2))

        return response_map


    def plot_ROI_on_anatomy_map(self, plane, roi_exp_tag):
        anatomy_plane = self.get_anatomy_image(1, 1, 1) #TODO: Hardcoded
        roi_selected = self.neuron.roi_exp_position[self.neuron.roi_exp_plane == plane]

        ax = isns.imgplot(anatomy_plane, cmap="magma")
        for roi in roi_selected:
            x, y = zip(*roi)
            ax.plot(x, y)

        return ax

    def plot_neuron_response_on_anatomy_map(self):
        pass

    # Plot functions
    def plot_mean_traces(self, time_window ='all', n_ID ='all', odors ='all', trials ='all', trace='raw', aligned = True, pool_trials = True, save_to_path=False):
        if time_window == "all":
            time_window = self.get_default("time_window")
        else:
            time_window = np.arange(time_window[0],time_window[1])
        if n_ID == "all":
            n_ID = self.get_default("n_ID")

        if odors == "all":
            odors = self.get_default("odors")

        if trials == "all":
            trials = self.get_default("trials")

        fig, axes = plt.subplots()
        for od in odors:
            if pool_trials == True:
                data = self.get_traces(time_window = [time_window[0], time_window[-1]], n_ID = n_ID, odors=[od], trials=trials, trace=trace, aligned=aligned)
                axes.plot(time_window,data.mean(axis=2).mean(axis=0), label=f"{self.get_odor_name(od)} - mean")
            else:
                for trial in trials:
                    data = self.get_traces(time_window = [time_window[0], time_window[-1]], n_ID = n_ID, odors=[od], trials=[trial], trace=trace, aligned=aligned)
                    axes.plot(time_window,data.mean(axis=2).mean(axis=0), label=f"{self.get_odor_name(od)} - mean trial {trial+1}")

        axes.set(xlabel="Frame", ylabel="Pixel value")
        fig.suptitle(f"{trace} traces = {self.get_param('ID')}")
        plt.legend(ncol=2)
        if save_to_path:
            plt.savefig(f'{self.get_param("results")}/ {f"{trace} traces"}.png', dpi=300)

        plt.show()


    # @staticmethod
    # def clut2b_map():
    #     """ Generates clut2b colormap used also in Neuroi e.g.
    #         # clut2b_map = clut2b_map()
    #         # plt.register_cmap(cmap=clut2b)
    #         # plt.set_cmap(clut2b)
    #         # plt.show()"""
    #     mat_data = loadmat("utils/clut2b.mat")
    #     clut2b_data = mat_data["clut2b"]
    #     clut2b = matplotlib.colors.LinearSegmentedColormap.from_list('clut2b', clut2b_data)
    #     return clut2b