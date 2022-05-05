import matplotlib.pyplot as plt
import scipy
import pylab
from matplotlib import pyplot


# create a class for preprocessing plots
class PreprocessPlots:

    def __init__(self, df_comp):
        """
        plots
        :param df_comp:
        """
        df_comp.Healthcare.plot(figsize=(20, 5), title="Healthcare")
        plt.savefig("../output/" + "healthcare.png")

        df_comp.Telecom.plot(figsize=(20, 5), title="Telecom")
        plt.savefig("../output/" + "telecome.png")

        df_comp.Banking.plot(figsize=(20, 5), title="Banking")
        plt.savefig("../output/" + "banking.png")

        df_comp.Technology.plot(figsize=(20, 5), title="Technology")
        plt.savefig("../output/" + "technology.png")

        df_comp.Insurance.plot(figsize=(20, 5), title="Insurance")
        plt.savefig("../output/" + "Insurance.png")

        df_comp["#noofchannels"].plot(figsize=(20, 5), title="#noofchannels")
        plt.savefig("../output/" + "noofchannels.png")

        df_comp["#ofphonelines"].plot(figsize=(20, 5), title="#ofphonelines")
        plt.savefig("../output/" + "noofphoneline.png")

        # Density Plots
        df_comp["Banking"].plot(kind='kde', figsize=(20, 10))
        pyplot.savefig("../output/" + "density_banking.png")

        # Density Plots
        df_comp["#noofchannels"].plot(kind='kde', figsize=(20, 10))
        pyplot.savefig("../output/" + "density_noofchannels.png")

        # Density Plots
        df_comp["#ofphonelines"].plot(kind='kde', figsize=(20, 10))
        pyplot.savefig("../output/" + "density_noofphonelines.png")

        # The QQ plot
        scipy.stats.probplot(df_comp["Banking"], plot=pylab)
        plt.title("QQ plot for Banking")
        pylab.savefig("../output/" + "banking_qq.png")

        # The QQ plot
        scipy.stats.probplot(df_comp["#ofphonelines"], plot=pylab)
        plt.title("QQ plot for ofphonelines")
        pylab.savefig("../output/" + "noofchanel_qq.png")
