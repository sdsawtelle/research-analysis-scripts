"""This module provides functionality to work with the results of fitting the constant power model EM traces. The
preparation and fitting of traces is housed in em_pc_prep_and_fit.py. The aforementioned module generates pickled
dictionary and dataframe objects that contain, for each trace, some demographics about the device, the prepared
dataframe, and the fitting results. This module is concerned with analyzing and visualizing the results stored in the
dicts and dataframes. Here I provide a series of code snippets that can be copy/pasted and run to achieve various
visualizations."""

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


######################################################
# HELPER PLOTTING FUNCTION #####
######################################################
def plot_ramps(df_ramps, falsecol=False, rjmax=None, trace=""):
    """Plot ramp cycle DF for ramps below rjmax, optionally with false ramps in magenta color."""
    fig1, ax1 = snp.newfig("ul")
    snp.labs("Voltage (V)", "Resistance ($\Omega$)", "EM Trace - " + trace)
    # Plot the full resistance trace before we drop rows for outliers etc.
    if not rjmax:
        for indx, row in df_ramps.iterrows():
            if row["falseramp"] and falsecol == True:
                ax1.plot(row["volt"], row["res"], marker='s', color="magenta", markersize=5, label=indx)
            else:
                ax1.plot(row["volt"], row["res"], markerfacecolor="None", label=indx)
    else:
        start_r = df_ramps.loc[0, "first_r"]
        rmax = start_r + rjmax
        for indx, row in df_ramps.iterrows():
            if row["last_v"] / row["last_i"] <= rmax:
                if row["falseramp"] and falsecol == True:
                    ax1.plot(row["volt"], row["res"], marker='s', color="magenta", markersize=5, label=indx)
                else:
                    ax1.plot(row["volt"], row["res"], markerfacecolor="None", label=indx)
            else:
                break
    return fig1, ax1



######################################################
# USEFUL SNIPPETS #####
######################################################
########################
# Set params and path
path = "C:/Users/Sonya/Documents/My Box Files/MolT Project/160927_SDS20_Chip_4-10/EM/EM_50x400/"
os.chdir(path)
rjmax = 200
rjmin = 0
dfpickle = "fit_results_df_Rjmax_" + str(rjmax) + "_Rjmin_" + str(rjmin) + ".pickle"


########################
# Open up the fit-info dataframe for playing with :)
with open(dfpickle, "rb") as myfile:
    df = pickle.load(myfile)


########################
# Open up the fit-info dataframe and truncate to only pc-conforming devices
with open(dfpickle, "rb") as myfile:
    df = pickle.load(myfile)
    df = df[df["pc_conforming"] == 1]


########################
# Open up the fit-info dict for playing with :)
with open("prepped_traces_dict.pickle", "rb") as myfile:
    traces = pickle.load(myfile)


########################
# Plot a fit-prepped DF (to see what traces were excluded etc.)
trace = "EM22_FT2_2526"
with open("prepped_traces_dict.pickle", "rb") as myfile:
    prepped_traces_dict = pickle.load(myfile)
df = prepped_traces_dict[trace]["df"]
plot_ramps(df, falsecol=True)


########################
# Set all devices to "conform"
with open(dfpickle, "rb") as myfile:
    df = pickle.load(myfile)
df["pc_conforming"] = 1
with open(dfpickle, "wb") as myfile:
    pickle.dump(df, myfile)


########################
# Copy conforming info from one fit_results_df to another
fit1 = "fit_results_df_Rjmax_200_Rjmin_0.pickle"
fit2 = "fit_results_df_Rjmax_80_Rjmin_0.pickle"
with open(fit1, "rb") as myfile:
    df = pickle.load(myfile)
    conforms = df["pc_conforming"]
with open(fit2, "rb") as myfile:
    df = pickle.load(myfile)
    df["pc_conforming"] = conforms
with open(fit2, "wb") as myfile:
    pickle.dump(df, myfile)




######################################################
# PLOT FITTED PARAMETERS ####
######################################################
# Move to the place where this data is stored
path = "C:/Users/Sonya/Documents/My Box Files/MolT Project/160927_SDS20_Chip_4-10/EM/"
os.chdir(path)
with open(dfpickle, "rb") as myfile:
    df = pickle.load(myfile)
    df = df[df["pc_conforming"] == 1]

fig, ax = snp.newfig()
snp.labs("Initial Total R ($\Omega$)", "Fitted Power (mW)", "Discard Devices with Large Initial R")
ax.plot(df["res_tot_0"], df["powj"]*1000, linestyle="", markersize=9)
plt.savefig("Powj_vs_InitialR_Rjmax_" + str(rjmax) + "_Rjmin_" + str(rjmin) + ".png")

fig, ax = snp.newfig()
snp.labs("Fitted $R_0^J$ ($\Omega$)", "Fitted Power (mW)", "Fit Parameters Excluding Non-Pc-Conforming")
ax.plot(df["rj0"], df["powj"]*1000, linestyle="", markersize=9, label="Chip 4-10")
plt.savefig("Rj0_vs_Powj_Rjmax_" + str(rjmax) + "_Rjmin_" + str(rjmin) + ".png")
ax.legend(loc="upper left")





######################################################
# PLOT COMBINED RESULTS FROM ALL SUBDIRECTORIES ####
######################################################
df_combo = pd.DataFrame()  # Don't run this if you are combining df's from mulitple chips

# Move to the place where this data is stored
path = "C:/Users/Sonya/Documents/My Box Files/MolT Project/160927_SDS20_Chip_4-10/EM/"
os.chdir(path)
rjmax = 200
rjmin = 0
dfpickle = "fit_results_df_Rjmax_" + str(rjmax) + "_Rjmin_" + str(rjmin) + ".pickle"

for (path, dirs, files) in os.walk(path):
    if (dfpickle) in files:
        fpath = path + "/" + dfpickle
        with open(fpath, "rb") as myfile:
            print(path)
            df = pickle.load(myfile)
            df = df[df["pc_conforming"] == 1]
            df_combo = pd.concat([df, df_combo])

combo_grp = df_combo["group"]
combo_pow = df_combo["powj"]*1000

# Plots of fitted power colored by "group" or category
fig, ax = snp.newfig("ur")
snp.catplot(combo_grp, combo_pow)
snp.labs("Geometry", "Fitted Critical Power (mW)", "Critical Power at 96K in Ambient")
plt.savefig("AllGeometries_Fitted_Rjmax_" + str(rjmax) + "_Rjmin_" + str(rjmin) + ".png")

# Plot of average fitted power by "group" or category, with std deviation overlaid
groups = df_combo["group"].unique()
mu_ps = [np.average(df_combo["powj"].where(df_combo["group"] == grp).dropna()*1000) for grp in groups]
sig_ps = [np.std(df_combo["powj"].where(df_combo["group"] == grp).dropna()*1000) for grp in groups]
fig, ax = snp.newfig("ur")
snp.labs("Geometry", "Fitted Critical Power (mW)", "Critical Power at 96K")
snp.catplot(groups, mu_ps)
plt.errorbar([4, 3, 2, 1, 0], mu_ps, yerr=sig_ps, marker="", markersize=10, color="black", linestyle="")
plt.savefig("AllGeometries_Fitted_Pc_Avg_ErrorBars_Rjmax_" + str(rjmax) + "_Rjmin_" + str(rjmin) + ".png")





######################################################
# PLOT COMBINED RESULTS FROM ALL SUBDIRECTORIES ####
# FANCY SPLITTING BY DIAMETER AND LENGTH ####
######################################################
df_combo = pd.DataFrame()  # Don't run this if you are combining df's from mulitple chips

# Move to the place where this data is stored
path = "C:/Users/Sonya/Documents/My Box Files/MolT Project/160927_SDS20_Chip_4-10/EM/"
os.chdir(path)
rjmax = 200
rjmin = 0
dfpickle = "fit_results_df_Rjmax_" + str(rjmax) + "_Rjmin_" + str(rjmin) + ".pickle"

for (path, dirs, files) in os.walk(path):
    if (dfpickle) in files:
        fpath = path + "/" + dfpickle
        with open(fpath, "rb") as myfile:
            print(path)
            df = pickle.load(myfile)
            df = df[df["pc_conforming"] == 1]
            df_combo = pd.concat([df, df_combo])

combo_grp = df_combo["group"]
combo_pow = df_combo["powj"]*1000
diams = [float(grp.split("x")[0]) for grp in combo_grp]
lengths = [float(grp.split("x")[1]) for grp in combo_grp]
mydf = pd.DataFrame()
mydf["d"] = diams
mydf["l"] = lengths
mydf["p"] = combo_pow.values

l60 = mydf[mydf["l"] == 60]
l400 = mydf[mydf["l"] == 400]


fig, ax = snp.newfig("ur")
snp.labs("Diameter (nm)", "Fitted Critical Power (mW)", "Critical Power at 96K")

mu_ps = [np.average(l400["p"].where(l400["d"] == diam).dropna()) for diam in [30, 50, 60]]
sig_ps = [np.std(l400["p"].where(l400["d"] == diam).dropna()) for diam in [30, 50, 60]]
ax.plot([30, 50, 60], mu_ps, label="Length 400 nm", markersize=10, linestyle="")
plt.errorbar([30, 50, 60], mu_ps, yerr=sig_ps, marker="", markersize=10, color="black", linestyle="")

mu_ps = [np.average(l60["p"].where(l60["d"] == diam).dropna()) for diam in [30, 50]]
sig_ps = [np.std(l60["p"].where(l60["d"] == diam).dropna()) for diam in [30, 50]]
ax.plot([30, 50], mu_ps, label="Length 60 nm", markersize=10, linestyle="")
plt.errorbar([30, 50], mu_ps, yerr=sig_ps, marker="", markersize=10, color="black", linestyle="")

ax.set_xlim([25, 65])
ax.set_ylim([0.05, 0.45])
ax.legend(loc="upper left")

plt.savefig("DiamAndLength_Fitted_Pc_Avg_ErrorBars_Rjmax_" + str(rjmax) + "_Rjmin_" + str(rjmin) + ".png")





######################################################
# PLOT CONFORM_PC 1/0 VERSUS INITIAL DEV RESISTANCE ####
######################################################
rjmax = 200
rjmin = 0
dfpickle = "fit_results_df_Rjmax_" + str(rjmax) + "_Rjmin_" + str(rjmin) + ".pickle"

path = "C:/Users/Sonya/Documents/My Box Files/molT Project/160924_SDS20_Chip_6-5/EM/EM_S0"
os.chdir(path)
with open(dfpickle, "rb") as myfile:
    df = pickle.load(myfile)
fig, ax = snp.newfig("ur")
snp.labs("Initial Total Resistance ($\Omega$)", "Conform to $P_c$ Model (Y/N)", "$R_{tot}^0$ and Pc Conformity")
ax.plot(df["res_tot_0"], df["pc_conforming"]+0.2, linestyle="", markersize=8, label="S0", color="red")
ax.legend(loc="upper right")
ax.set_ylim([-0.05, 1.25])
plt.savefig("PcConformity_vs_InitialR.png")





######################################################
# PLOT EFFECT OF RJMIN AND RJMAX ON FITTED PC ####
######################################################
path = "C:/Users/Sonya/Documents/My Box Files/molT Project/160924_SDS20_Chip_6-5/EM/EM_S4"
os.chdir(path)
rjmax = [80, 100, 120, 140, 160, 180, 200, 220, 240]
rjmin = 0
mu_ps = []
sig_ps = []

for rj in rjmax:
    fit_all(path, group="50x400", rjmax=rj, temp=94, env="He", chip="6-5", rjmin=rjmin)
    dfpickle = "fit_results_df_Rjmax_" + str(rj) + "_Rjmin_" + str(rjmin) + ".pickle"
    with open(dfpickle, "rb") as myfile:
        df = pickle.load(myfile)
    df = df[df["pc_conforming"] == 1]
    mu_ps.append(np.average(df["powj"]))
    sig_ps.append(np.std(df["powj"]))
mu_ps_mw = [1000*elem for elem in mu_ps]
sig_ps_mw = [1000*elem for elem in sig_ps]

fig, ax = snp.newfig("ur")
snp.labs("Max $R_j$ Included in Fit", "Average Fitted $P_c$ (mW)", "Effect of Fit Range on Average $P_c$")
plt.errorbar(rjmax, mu_ps_mw, yerr=sig_ps_mw, marker="o", markersize=10, color="black", linestyle="")
ax.set_xlim([70, 250])
plt.savefig("Avg_Pc_Versus_Fitting_Range_Rjmax_Rjmin0.png")

fig, ax = snp.newfig("ur")
snp.labs("Max $R_j$ Included in Fit", "Std. Dev. of Average $P_c$ (mW)", "Effect of Fit Range on Std. Dev. of Avg. $P_c$")
ax.plot(rjmax, sig_ps_mw, markersize=8)
plt.savefig("StdDev_AvgPc_Versus_Fitting_Range_Rjmax_Rjmin0.png")

with open("AvgPc_and_StdDev_For_Variable_Rjmax_Rjmin0.pickle", "wb") as myfile:
    pickle.dump([rjmax, mu_ps_mw, sig_ps_mw], myfile)




print("debug")
