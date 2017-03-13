"""  This script will make plots of I vs. V and R vs. V from standard EM traces that were acquired with C++ molT master
program.
"""

__author__ = 'Sonya'
__email__ = "sdsawtelle@gmail.com"
__version__ = "3.5"

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import least_squares
import pickle
import math
import os
import sys

import snips as snp

# Boilerplate settings
matplotlib.interactive = True
snp.prettyplot(matplotlib)
plt.close("all")


# Move to the place where the data is stored
path = "C:/Users/Sonya/Documents/My Box Files/MolT Project/170225_SDS20_Chip_10-5/50x400_93"
os.chdir(path)
save = True

################### PLOT IV ##########################
for fname in snp.txtfilenames(path):
    df = pd.read_csv(fname, header=1)
    df.columns = [col.strip(" ") for col in df.columns]  # Some headers have extra spaces
    fig, ax = snp.newfig()
    snp.labs("Voltage (V)", "Current (I)", "EM Trace - %s" %(fname.strip(".txt"),))
    # ax.plot(df["voltage"], df["current"], linestyle="None")
    ax.plot(df["current"], df["resistance"], linestyle="None")

    if(save):
        plt.savefig("EM_IV_Trace_%s.png" %(fname.strip(".txt"),))
plt.close("all")


################### PLOT RV ##########################
for fname in snp.txtfilenames(path):
    df = pd.read_csv(fname, header=1)
    df.columns = [col.strip(" ") for col in df.columns]  # Some headers have extra spaces
    fig, ax = snp.newfig()
    snp.labs("Voltage (V)", "Resistance ($\Omega$)", "EM Trace - %s" %(fname.strip(".txt"),))
    ax.plot(df["voltage"], df["resistance"], linestyle="None")
    if(save):
        plt.savefig("EM_RV_Trace_%s.png" %(fname.strip(".txt"),))
    plt.close("all")


################### PLOT RV FOR PARTIAL/FINAL BREAKING TOGETHER ##########################
names = snp.txtfilenames(path)
ids = ["_".join(name.split("_")[-2:]).strip(".txt") for name in names]
ids = list(set(ids))
for id in ids:
    fig, ax = snp.newfig()
    snp.labs("Voltage (V)", "Resistance ($\Omega$)", "EM Trace - %s" % (id,))
    for fname in [name for name in names if id in name]:
        df = pd.read_csv(fname, header=1)
        df.columns = [col.strip(" ") for col in df.columns]  # Some headers have extra spaces
        ax.plot(df["voltage"], df["resistance"], linestyle="None")
    if(save):
        # plt.savefig("EM_RV_Trace_%s.png" %(id,))
        pass
    # plt.close("all")


################### PLOT FOR CONST-VOLTAGE EM TRACES ##########################
for fname in snp.txtfilenames(path):
    with open(fname) as f:
        data = f.read()
    temp = data.split("\n0, ")
    partialem = temp[0]
    fname_partial = fname.strip(".txt") + "_PARTIAL.txt"
    with open(fname_partial, "w") as fout:
        fout.write(partialem)

    if len(temp) != 1:
        constv = "time,voltage,current,resistance,volt_range,curr_range\n0, " + temp[1]
        fname_constv = fname.strip(".txt") + "_CONSTV.txt"
        with open(fname_constv, "w") as fout:
            fout.write(constv)

        df = pd.read_csv(fname_constv, header=0)
        df.columns = [col.strip(" ") for col in df.columns]  # Some headers have extra spaces
        fig, ax = snp.newfig()
        snp.labs("Voltage (V)", "Resistance ($\Omega$)", "EM Trace - %s" %(fname_constv.strip(".txt"),))
        ax.plot(df["time"], df["resistance"], linestyle="None")
        # ax.set_ylim([df["resistance"].min(), df["resistance"].min()+200])
        if(save):
            plt.savefig("EM_CONSTV_Trace_%s.png" %(fname.strip(".txt"),))
plt.close("all")


################### COMPARE IVs AT DIFFERENT TEMPERATURES ##########################
# Move to the place where this data is stored
path = "C:/Users/Sonya/Documents/My Box Files/MolT Project/170112_SDS20_Chip_7-11/IV_125K"

os.chdir(path)
save = True
fnames = snp.txtfilenames(path)
ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
ports = list(set(ports))
colors = len(ports)

fig, ax = snp.newfig()
snp.labs("Voltage (V)", "Resistance ($\Omega$)", "Comparing IVs Before and After Heating")

ratios = {}
for port in ports:
    color = next(colors)
    traces = [fname for fname in fnames if port in fname]
    mtimes = [os.path.getmtime(fname) for fname in traces]
    traces = [trace for (mtime, trace) in sorted(zip(mtimes, traces))]
    df = pd.read_csv(traces[0], header=0)
    df.columns = ["voltage", "current"]  # Some headers have extra spaces
    ax.plot(df["voltage"], df["voltage"]/df["current"], color=color)
    df1 = pd.read_csv(traces[1], header=0)
    df1.columns = ["voltage", "current"]  # Some headers have extra spaces
    ax.plot(df1["voltage"], df1["voltage"]/df1["current"], color=color, linestyle="--", label=port)

    res1 = df1["voltage"]/df1["current"]
    res0 = df["voltage"]/df["current"]
    ratios[port] = (res1[-10:] / res0[-10:]).mean()

for item in ratios.items():
    print(item)
print("Mean Ratio is %.2f +/- %.2f" % (np.mean(list(ratios.values())), np.std(list(ratios.values()))))
ax.legend(loc="lower right")
plt.savefig("Compare_Temperature_IVs.png")
plt.close("all")



