import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn_image as isns

class DataPlotter:
    def __init__(self, data_getter):
        self.data_getter = data_getter

    def plot_traces(self, time_window='all', n_ID='all', odors='all', trials='all', trace='raw'):
        """
        Plots neural activity traces for selected neurons and conditions.

        Parameters:
            time_window (list | str): Start and end frames to plot, or 'all' for full range.
            n_ID (list | str): Neuron IDs to include in the plot, or 'all' for all neurons.
            odors (list | str): Odor conditions to include, or 'all' for all conditions.
            trials (list | str): Trial numbers to include, or 'all' for all trials.
            trace (str): Type of trace data to plot (e.g., 'raw', 'dF').

        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The figure and axes of the plot.
        """
        try:
            data = self.data_getter.get_traces(time_window=time_window, n_ID=n_ID, odors=odors, trials=trials, trace=trace)
            selected_trials = self.data_getter.get_trial_index(odors=odors, trials=trials)

            fig, ax = plt.subplots()
            for id, n in enumerate(n_ID):
                for idt, trial in enumerate(selected_trials):
                    ax.plot(data[id, :, idt], label=f"nID: {n} - {self.data_getter.get_odor_name(trial)} - t:{trial}")
            ax.legend()
            return fig, ax
        except Exception as e:
            print(f"An error occurred while plotting traces: {e}")
            raise

    def plot_heatmap(self, time_window='all', n_ID='all', odors='all', trials='all', trace='raw'):
        """
        Plots a heatmap of responses for specified neurons and conditions over time.

        Parameters:
            time_window (list | str): Specific time window or 'all' for the full range.
            n_ID (list | str): Neuron IDs to plot or 'all' for all.
            odors (list | str): Odor conditions or 'all'.
            trials (list | str): Trials or 'all'.
            trace (str): Trace type such as 'raw', 'df', or 'spikes'.

        Returns:
            Tuple[matplotlib.figure.Figure, ndarray[matplotlib.axes.Axes]]: The figure and array of axes with the heatmap.
        """
        try:
            data = self.data_getter.get_traces(time_window=time_window, n_ID=n_ID, odors=odors, trials=trials, trace=trace)
            if time_window == "all":
                time_window = self.data_getter.get_default("time_window")
            else:
                time_window = np.arange(time_window[0], time_window[-1])

            fig, axes = plt.subplots(nrows=len(trials), ncols=len(odors), sharex=True, sharey=True, figsize=(15, 10))
            for row in range(len(trials)):
                for col in range(len(odors)):
                    sns.heatmap(data[:, :, row * len(odors) + col], ax=axes[row][col], cbar=True)
                    axes[row][col].set_title(f'Odor {odors[col]} Trial {trials[row]}')

            for ax, col in zip(axes[0], odors):
                ax.set_title(f'Odor {col}')
            for ax, row in zip(axes[:,0], trials):
                ax.set_ylabel(f'Trial {row}', rotation=90, size='large')

            plt.tight_layout()
            return fig, axes
        except Exception as e:
            print(f"An error occurred while plotting heatmap: {e}")
            raise


    def plot_mean_traces(self, time_window='all', n_ID='all', odors='all', trials='all', trace='raw', aligned=True,
                         pool_trials=True):
        """
        Plots mean traces for selected conditions, potentially pooling across trials.

        Parameters:
            time_window (list | str): Start and end frames to plot, or 'all' for full range.
            n_ID (list | str): Neuron IDs to include in the plot, or 'all' for all neurons.
            odors (list | str): Odor conditions to include, or 'all' for all conditions.
            trials (list | str): Trial numbers to include, or 'all' for all trials.
            trace (str): Type of trace data to plot (e.g., 'raw', 'dF').
            aligned (bool): Whether to align traces based on some event or marker.
            pool_trials (bool): Whether to average traces across trials.

        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The figure and axes of the plot.
        """
        try:
            data = self.data_getter.get_traces(time_window=time_window, n_ID=n_ID, odors=odors, trials=trials,
                                               trace=trace)
            fig, ax = plt.subplots()

            for od in odors:
                if pool_trials:
                    mean_data = data.mean(axis=2)  # Averaging across trials
                else:
                    mean_data = data

                ax.plot(mean_data, label=f"{self.data_getter.get_odor_name(od)}")

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Signal")
            ax.legend(title="Odor")
            return fig, ax
        except Exception as e:
            print(f"An error occurred while plotting mean traces: {e}")
            raise

    def plot_ROI_on_anatomy_map(self, plane, roi_exp_tag):
        """
        Plots ROIs on an anatomical map of a given plane.

        Parameters:
            plane (int): The specific plane of the sample to visualize.
            roi_exp_tag (list): Tags or identifiers for ROIs to be plotted.

        Returns:
            matplotlib.axes.Axes: The axes object with the ROI plot.
        """
        try:
            anatomy_image = self.data_getter.get_anatomy_image(plane)
            ax = isns.imgplot(anatomy_image, cmap="magma")

            for roi in roi_exp_tag:
                x, y = zip(*roi)  # Assuming roi is a list of tuples (x, y)
                ax.plot(x, y, marker='o', color='red', linestyle='None')

            return ax
        except Exception as e:
            print(f"An error occurred while plotting ROIs on anatomy map: {e}")
            raise
