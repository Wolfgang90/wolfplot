import matplotlib as mpl


# Style for client 1

def set_style():

        dict_style = {
                "patch.linewidth": 0.3, 
                "patch.linewidth": 0.3,   
                "patch.facecolor": "003781",
                "patch.force_edgecolor": True,   
                "boxplot.patchartist" : True,
                "boxplot.flierprops.color":"000000",
                "boxplot.flierprops.markerfacecolor":"003781",
                "boxplot.boxprops.color": "000000",
                "boxplot.medianprops.color": "ffffff",
                "font.family": "arial",
                "font.size": 10.0,
                "axes.linewidth": 0.5,
                "axes.titlesize": "small",
                "axes.titleweight": "bold",
                "axes.titlepad" : 15.0,
                "axes.labelpad": 6.0,
                "axes.labelweight": "bold",
                "axes.axisbelow": True,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.prop_cycle"    : "cycler('color', ['003781', '96dcfa', 'ccdd61', 'fdd25c', 'dad0e1', 'ff934f', 'b1dadd', '000000', 'd4cdcd', '706060'])",
                "xtick.major.size"     : 4,
                "xtick.minor.size"     : 2,
                "xtick.major.width"    : 0.3,
                "xtick.minor.width"    : 0.3,
                "xtick.labelsize"      : "x-small",
                "ytick.major.width"    : 0.7,
                "ytick.labelsize":"x-small",
                "grid.color":"d4cdcd",
                "grid.linestyle":"-",
                "grid.linewidth":0.3,
                "grid.alpha":1,
                "legend.framealpha":1,
                "legend.fontsize":"xx-small",
                "figure.titlesize":"small",
                "figure.dpi": 500,
                "savefig.dpi": 500
                }

        mpl.rcParams.update(dict_style)

