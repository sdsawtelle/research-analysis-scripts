"""  This script will take the standard yield measurement files in a directory and make a plot of resistance values.
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


################### PLOT YIELD RESISTANCES FOR SINGLE CHIP ##########################
# Move to the place where this data is stored
path = "C:/Users/Sonya/Documents/My Box Files/MolT Project/170113_SDS20_Chip_9-11/YIELDS/yield-monitor"
os.chdir(path)
save = True

res = []
for fname in snp.txtfilenames(path):
    df = pd.read_csv(fname, header=1)
    df.index = df.iloc[:, 0].values
    df.drop("Device Port Specs", axis=1, inplace=True)
    df.columns=["Resistance"]
    res += list(df["Resistance"].values)

res = np.array(res)
res = np.log10(res)
geom_map_30x10 = np.array([True]*11 + [False]*10 + [True]*11 + [False]*10)
geom_map_50x400 = ~ geom_map_30x10

res_30x10 = res[geom_map_30x10]
res_50x400 = res[geom_map_50x400]

# catlabs = ["30x10"]*len(res_30x10) + ["50x400"]*len(res_50x400)
# fig, ax = snp.newfig()
# snp.labs("device geometry", "LOG10 device resistances", "Initial Yields at 100 mV")
# snp.catplot(catlabs, np.concatenate([res_30x10, res_50x400]), ax, s=20)

fig, ax = snp.newfig()
snp.labs("device geometry", "LOG10 device resistances", "Initial Yields at 100 mV")
ax.plot(np.arange(len(res_30x10)), res_30x10, label="30x10", linestyle="None", markersize=8)
ax.plot(np.arange(len(res_50x400)), res_50x400, label="50x400", linestyle="None", markersize=8)
ax.legend()


################### PLOT YIELD RESISTANCES FOR MULTIPLE CHIP ##########################
paths = ["C:/Users/Sonya/Documents/My Box Files/MolT Project/170113_SDS20_Chip_9-11/YIELDS/yield-monitor",
        "C:/Users/Sonya/Documents/My Box Files/MolT Project/170113_SDS20_Chip_9-11/YIELDS_6-9",
        "C:/Users/Sonya/Documents/My Box Files/MolT Project/170113_SDS20_Chip_9-11/YIELDS_11-10",
         "C:/Users/Sonya/Documents/My Box Files/MolT Project/170113_SDS20_Chip_9-11/YIELDS_7-11",
         "C:/Users/Sonya/Documents/My Box Files/MolT Project/170113_SDS20_Chip_9-11/YIELDS_5-8"]
fig1, ax1 = snp.newfig()
snp.labs("device geometry", "LOG10 device resistances", "30x10 Initial Yields at 100 mV")
ax1.axhline(y=2.6989, xmin=0, xmax=42, linewidth=2, color="black")

fig2, ax2 = snp.newfig()
snp.labs("device geometry", "LOG10 device resistances", "50x400 Initial Yields at 100 mV")
ax2.axhline(y=2.6989, xmin=0, xmax=42, linewidth=2, color="black")

chiplabs = iter(["Chip 9-11", "Chip 6-9", "Chip 11-10", "Chip 7-11", "Chip 5-8"])

for path in paths:
    chiplab = next(chiplabs)
    # Move to the place where this data is stored
    os.chdir(path)
    save = True
    res = []
    for fname in snp.txtfilenames(path):
        df = pd.read_csv(fname, header=1)
        df.index = df.iloc[:, 0].values
        df.drop("Device Port Specs", axis=1, inplace=True)
        df.columns=["Resistance"]
        res += list(df["Resistance"].values)

    res = np.array(res)
    res = np.log10(res)
    geom_map_30x10 = np.array([True]*11 + [False]*10 + [True]*11 + [False]*10)
    geom_map_50x400 = ~ geom_map_30x10

    res_30x10 = res[geom_map_30x10]
    res_50x400 = res[geom_map_50x400]

    catlabs = ["30x10"]*len(res_30x10) + ["50x400"]*len(res_50x400)

    ax1.plot(np.arange(len(res_30x10)), res_30x10, label=chiplab, linestyle="None", markersize=8)
    ax2.plot(np.arange(len(res_50x400)), res_50x400, label=chiplab, linestyle="None", markersize=8)
ax1.legend(loc="upper right")
ax2.legend(loc="upper right")

fig1.savefig("YieldMonitoring_30x10.png")
fig2.savefig("YieldMonitoring_50x400.png")

