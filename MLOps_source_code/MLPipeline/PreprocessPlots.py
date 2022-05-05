import matplotlib.pyplot as plt
import scipy
import pylab
from matplotlib import pyplot



class PreprocessPlots:

    def __init__(self, df_comp):
        """
        plots
        :param df_comp:
        """
        df_comp.Healthcare.plot(figsize=(20, 5), title="Healthcare")
        plt.savefig("./Output/" + "healthcare.png")

        df_comp.Telecom.plot(figsize=(20, 5), title="Telecom")
        plt.savefig("./Output/" + "telecome.png")

        df_comp.Banking.plot(figsize=(20, 5), title="Banking")
        plt.savefig("./Output/" + "banking.png")

        df_comp.Technology.plot(figsize=(20, 5), title="Technology")
        plt.savefig("./Output/" + "technology.png")

        df_comp.Insurance.plot(figsize=(20, 5), title="Insurance")
        plt.savefig("./Output/" + "Insurance.png")

        df_comp["#noofchannels"].plot(figsize=(20, 5), title="#noofchannels")
        plt.savefig("./Output/" + "noofchannels.png")

        df_comp["#ofphonelines"].plot(figsize=(20, 5), title="#ofphonelines")
        plt.savefig("./Output/" + "noofphoneline.png")

        # Density Plots
        df_comp["Banking"].plot(kind='kde', figsize=(20, 10))
        pyplot.savefig("./Output/" + "density_banking.png")

        # Density Plots
        df_comp["#noofchannels"].plot(kind='kde', figsize=(20, 10))
        pyplot.savefig("./Output/" + "density_noofchannels.png")

        # Density Plots
        df_comp["#ofphonelines"].plot(kind='kde', figsize=(20, 10))
        pyplot.savefig("./Output/" + "density_noofphonelines.png")

        # The QQ plot
        scipy.stats.probplot(df_comp["Banking"], plot=pylab)
        plt.title("QQ plot for Banking")
        pylab.savefig("./Output/" + "banking_qq.png")

        # The QQ plot
        scipy.stats.probplot(df_comp["#ofphonelines"], plot=pylab)
        plt.title("QQ plot for ofphonelines")
        pylab.savefig("./Output/" + "noofchanel_qq.png")
