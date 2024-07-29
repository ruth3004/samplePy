import matplotlib.pyplot as plt
# --------------------------------------#
""" REPORT AND SAVE FUNCTIONS"""

# --------------------------------------#


class DataReporter:

    def save_to_eps(self, path, name):
        # matplotlib.rcParams['pdf.fonttype'] = 42
        # matplotlib.rcParams['ps.fonttype'] = 42
        plt.savefig(f'{path}/{name}.eps', format='eps')


    #--------------------------------------#
    """ REPORT FUNCTIONS"""
    # --------------------------------------#
    def report(self, *args):
        #TODO: Create report
        pass