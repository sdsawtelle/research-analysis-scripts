"""  This script generates the calibration file for the Cernox resistor on the 84-pin PGA probe. This file gives
resistance of the sensor as a function of temperature and is fed to the Lakeshore Temperature Controller. Note that the
TC will only use the first 200 points of any such file and discard the rest (without a freaking warning!).
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


# As of 2/11/2017 I can't find the original data files that Zak took which showed the Cernox resistance as a function of
# the reading on the AlGas sensor (calibrated by Janis) during a passive cooldown. Instead I just have the calibration
# file I generated from the data. The passive cooldown data goes from 292 to 92 K, but now that we are filling the LHe
# Reservoir with LN2 we need to extend the calibration file down to 73 K. Note that TCB is presently just used as the
# controller for the heater, not as a faithful reading of the sample temperature, so it's acceptable to have large error
# in the calibration file.

os.chdir("C:/Users/Sonya/Documents/My Box Files/Projects/python-research-data-scripts/cernox_calibration")
df = pd.read_csv("calcurve.34A", skiprows=5, header=None)
df.columns = ["log(R/T)", "Temp"]
df["log(R/T)"] = df["log(R/T)"].str.split(": ").apply(lambda x: x[1]).astype(np.float64)

fig, ax = plt.subplots()
snp.labs("Temperature (K)", "log10 (Resistance / Temp)", "Cernox Calibration Curve")
ax.plot(df["Temp"], df["log(R/T)"], markersize=0, label="previous calibration curve")

# calculate polynomial
pfit = np.polyfit(df["Temp"], df["log(R/T)"], 5)
pfitfunc = np.poly1d(pfit)

# calculate new x's and y's
temps = np.arange(91.61, 70, -0.01)
predict = pfitfunc(temps)
ax.plot(temps, predict, color="red", markersize=0, label="extrapolated region")
ax.legend()
plt.savefig("Cernox_CalCurve.png")

all_temps = np.concatenate([df["Temp"].values, temps])
all_vals = np.concatenate([df["log(R/T)"].values, predict])

space = np.floor(len(all_vals)/200)
indxs = space*np.arange(0, 199, 1)
indxs = indxs.astype(int)

T = all_temps[indxs]
logR = all_vals[indxs]

newdf = pd.DataFrame({"T": T, "logR": logR})
newdf["idx"] = newdf.index

# Pandas apply passes each row as a numpy array
def stringify(x):
    return "Point %i: %.6f" % (x[2], x[1])
newdf["col1"] = newdf.apply(stringify, axis=1)

HEADER = "Name: CX-1050-CU-HT\nSerial number: X104902\nFormat: 4             ; Log Ohms/Kelvin\nLimit: 292.5\nCoefficient: 1        ; Negative"
newdf.to_csv(columns=["col1", "T"], path_or_buf="cernox_calcurve.34A", index=False, header=None)