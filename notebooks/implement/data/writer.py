import matplotlib.pyplot as plt
class DataWriter:
    # --------------------------------------#
    """ SAVE FUNCTIONS"""
    ###Contains functions or classes to save processed data in a standardized format."""

    # --------------------------------------#
    def save_to_eps(self, path, name):
        # matplotlib.rcParams['pdf.fonttype'] = 42
        # matplotlib.rcParams['ps.fonttype'] = 42
        plt.savefig(f'{path}/{name}.eps', format='eps')
