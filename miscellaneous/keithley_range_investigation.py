"""  This script will plot traces acquired from the molT-master EM algorithm that are color by the current sensing range
for each data point (which is recorded in the data file). I want to have a sense of what kind of current range issues we
expect. It also encapsulates a model for fitting a voltage and current offset for keithley, assuming that the trace is
measuring a true fixed resistance.
"""

__author__ = 'Sonya'
__email__ = "sdsawtelle@gmail.com"
__version__ = "3.5"

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import pickle
import os
import scipy
from scipy.optimize import least_squares

import snips as snp

# Boilerplate settings
matplotlib.interactive(True)
snp.prettyplot(matplotlib)
plt.close("all")

path = "C:/Users/Sonya/Documents/My Box Files/molT Project/161018_SDS20_Chip_7-2-b_TestingRanges/good"
os.chdir(path)


############ PLOTTING TRACES COLORED BY RANGE ############
fnames = snp.txtfilenames(path)
for fname in fnames:
    # Prepare Data Frame
    res = fname.strip(".txt").split("_")[0]
    df = pd.read_csv(fname, header=2)
    df.columns = ["voltage","current", "resistance", "vrange", "irange"]
    df["res"] = df["voltage"]/df["current"]

    df_pt1mA = df[df["irange"] == 0.0001]
    df_1mA = df[df["irange"] == 0.001]
    df_10mA = df[df["irange"] == 0.01]
    df_100mA = df[df["irange"] == 0.1]
    fig, ax = snp.newfig()
    snp.labs("Voltage (V)", "Resistance ($\Omega$))", "V and I Ranges - " + res)
    ax.plot(df_pt1mA["voltage"], df_pt1mA["res"], label="Current Range 100 uA", linestyle="")
    ax.plot(df_1mA["voltage"], df_1mA["res"], label="Current Range 1 mA", linestyle="")
    ax.plot(df_10mA["voltage"], df_10mA["res"], label="Current Range 10 mA", linestyle="")
    ax.plot(df_100mA["voltage"], df_100mA["res"], label="Current Range 100 mA", linestyle="")
    ax.vlines(0.20, *ax.get_ylim(), linestyles="dashed")
    # ax.vlines(2.05, *ax.get_ylim(), linestyles="dashed")
    ax.legend(loc="lower center")
    plt.savefig("EM_Variable_Ranges_Res_" + res + ".png")
    plt.close(fig)

    fig, ax = snp.newfig()
    snp.labs("Voltage (V)", "Current (A))", "V and I Ranges - " + res)
    ax.plot(df_pt1mA["voltage"], df_pt1mA["current"], label="Current Range 100 uA", linestyle="")
    ax.plot(df_1mA["voltage"], df_1mA["current"], label="Current Range 1 mA", linestyle="")
    ax.plot(df_10mA["voltage"], df_10mA["current"], label="Current Range 10 mA", linestyle="")
    ax.plot(df_100mA["voltage"], df_100mA["current"], label="Current Range 100 mA", linestyle="")
    ax.vlines(0.20, *ax.get_ylim(), linestyles="dashed")
    # ax.vlines(2.05, *ax.get_ylim(), linestyles="dashed")
    ax.legend(loc="lower center")
    plt.savefig("EM_Variable_Ranges_Curr_" + res + ".png")
    plt.close(fig)


############ ACTUAL RESISTANCE TRACES ############
resists = np.array(["84ohm", "176ohm", "605ohm", "1490ohm"])
algos = ["21V_100mA", "2V_10mA", "21V_10mA"]
colordict = snp.styledict(resists, type="color")
markerdict = snp.styledict(algos, type="marker")

fig1, ax1 = snp.newfig("ll")
# background = fig1.add_subplot(111, frameon=False)  # Invisible except use him to create centered title and legends
# background.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')  # Hide ticks & tick labels
fig1.suptitle("Measured IV For Different R")
fig1.text(0.5, 0.04, 'Voltage (V)', ha='center')
fig1.text(0.04, 0.5, 'Resistance ($\Omega$)', va='center', rotation='vertical')

axlist = {}
for indx, res in enumerate(resists):
    axlist[res] = plt.subplot(2, 2, indx+1)

names = snp.txtfilenames(path)
df_all = pd.DataFrame(index=names, columns=["volt", "curr", "res"])
for name in names:
    parsedname = name.split("_")
    res = parsedname[0]
    ax = axlist[res]
    alg = parsedname[1] + "_" + parsedname[2]
    df = pd.read_csv(name, header=1)
    df_all.loc[name, "volt"] = df.loc[:, " voltage"].values
    df_all.loc[name, "curr"] = df.loc[:, "current"].values
    # df_all.loc[name, "res"] = df.loc[:, "resistance "].values
    df_all.loc[name, "res"] = df_all.loc[name, "volt"]/ df_all.loc[name, "curr"]
    ax.plot(df_all.loc[name, "volt"], df_all.loc[name, "res"], marker=markerdict[alg], color=colordict[res],
             markersize=6, label=name)
    plt.text(0.8, 0.15, str(res), ha='center', va='center', transform=ax.transAxes, size=10)
    # snp.labs("Voltage (V)", "Resistance ($\Omega$)", "Trace - " + res)

# with open("Res_Oscillation_Offset.pickle", 'wb') as myfile:
#     pickle.dump(fig1, myfile)


################ FITTING A VOLTAGE AND CURRENT OFFSET MODEL ##############################
def predict_data(theta, volt):
    '''Compute a predicted current given fixed resistance and model parameters of voltage and current offset.'''
    rconst = theta[0]
    voffset = theta[1]
    ioffset = theta[2]
    predicted_curr = (volt + voffset)/rconst - ioffset
    return predicted_curr

def residuals(theta, volt, curr):
    """Cmpute the residuals between the real data and the predicted current"""
    predicted_curr = predict_data(theta, volt)
    # Note, without .astype() the returned array would actually have data type = "object" and would store pointers
    # to the float values rather than storing the values themselves. This would throw an error in the LSQ.
    return (curr - predicted_curr).astype(np.float64)

def do_fit(volt, curr):
    # Fit the model using LSQ loss (alternately, specify loss to get more outlier-robust fitting)
    theta0 = [86.55, 0, 0]  # first param is critical power, second param is initial Rj
    constraints = ([70, -0.05, -10e-5], [100, 0.05, 10e-5])
    lsq = least_squares(residuals, theta0, args=(volt, curr), bounds=constraints)
    return lsq

name = "84ohm_2V_10mA_test_0102.txt"
curr = df_all.loc[name, "curr"].copy()[0:700]
volt = df_all.loc[name, "volt"].copy()[0:700]

fit = do_fit(volt, curr)
predict_curr = predict_data(fit.x, volt)

# Plot the fit and the data
fig2, ax2 = snp.newfig("ul")
snp.labs("Voltage (V)", "Resistance ($\Omega$)", "Keithley Voltage / Current Offset - R = 84 $\Omega$")
ax2.plot(volt, volt/curr, label="data - 84 $\Omega$")
ax2.plot(volt, (volt-fit.x[1])/(curr-fit.x[2]), label="LSQ fit - 84 $\Omega$")
ax2.legend()
ax2.text(x=1.2, y=0.2, s="R = %.2f\n Voffset = %.2f mV\n Ioffset = %.2f $\mu$A" %
                         (fit.x[0], fit.x[1]*1000, fit.x[2]*10**6), ha='center', va='center',
         transform=ax.transAxes, size=14)
