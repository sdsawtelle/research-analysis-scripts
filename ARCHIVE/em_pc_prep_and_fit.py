"""This module provides functionality to plot, prepare and fit standard EM traces within the constant-power model for
electromigration. The user is able to restrict what parts of the trace should be included in analysis to exclude, for
instance, later ramp cycles once the junction resistance is high and the device is unstable. Once the trace has been culled
in this fashion, the analysis consists of identifying the current and voltage points where EM occurs, and then fitting
that set of points using a constant critical power model.

Specific usage is to first call pic_all() to make plots of all traces, so that crappy ones can be removed from the
folder. Then call prep_all() to do all initial preparation of the good traces like selecting ramps for analysis and
setting res_tot_0. This creates and pickles a dict of dicts whose keys are the trace names and whose values are a dict
of the prepared DF and the res_tot_0 value. Optionally now you can call revise_prep() if you want to revise the
preparation of any specific trace - it will update the pickled dictionary.

Now call fit_all() with kwargs rjmin and rjmax to fit the ramps between rjmin and rjmax for each trace to the constant
power model. This function creates and pickles a dataframe whose indices are the trace names and whose values are
various demographic info about the device as well as the fit results. It also creates and pickles a dict of dicts with
the same information. The pickled dataframe has a column 'conforms' which denotes whether the device appears to conform
to the constant power model or not. This is initialized to 1 for all traces but values can later be changed by calling
set_pc_conforms(fname). The dict and df pickles are named with the specific rjmin and rjmax values that were used for
the fitting, so users can try different values of these kwargs and not overwrite previous results."""

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
# PREPARING SINGLE EM TRACE #####
######################################################
def get_trace(fname):
    """Read an EM trace text file to a DF and fix column names."""
    df = pd.read_csv(fname, header=1)
    df.columns = [col.strip(" ") for col in df.columns]  # Some headers have extra spaces
    return df


def extract_ramps(df):
    """Take a raw DF of v, r, i and create a DF of ramp cycles, let user select "good" cycles for fitting."""
    # Figure out how many different ramp cycles there are and get indexes for their endpoints
    # v1 = 0.1, 0.11, 0.12, 0.13, 0.14, 0.11
    # v2 = v1.shift(-1)
    # v2 = 0.11, 0.12, 0.13, 0.14, 0.11, nan
    # v1 > v2 only at the index for the last point in a ramp cycle (note final point will never be included due to NaN).
    rampbackpoints = df.voltage > df.voltage.copy().shift(-1)
    rampbackpoints.iloc[0] = rampbackpoints.iloc[-1] = True  # Manually add 2nd and last data points as ramp back points
    rampindx = rampbackpoints[rampbackpoints].index.tolist()  # Get Series index values for all the ramp back points.
    nramps = len(rampindx) - 1  # Recall each ramp consumes two rampindex points, a start and stop

    # Split the trace into ramp cycles and store relevant quantities in a DF
    df_ramps = pd.DataFrame(index=range(0,nramps,1), columns=["volt", "curr", "res", "last_v", "last_i", "first_r", "falseramp"])
    for i in range(0, nramps, 1):
        start = rampindx[i]+1  # Note this discards the first data point of the first ramp cycle... it's OK
        stop = rampindx[i+1]
        if start == stop:  # Discard any ramps which are only one point long (happens at end of trace sometimes)
            df_ramps.drop(i, axis=0, inplace=True)
        else:
            df_ramps.loc[i, "volt"] = np.array(df.loc[start:stop:1, "voltage"])
            df_ramps.loc[i, "last_v"] = df_ramps.loc[i, "volt"][-1]
            df_ramps.loc[i, "curr"] = np.array(df.loc[start:stop:1, "current"])
            df_ramps.loc[i, "last_i"] = df_ramps.loc[i, "curr"][-1]
            df_ramps.loc[i, "res"] = df_ramps.loc[i, "volt"]/df_ramps.loc[i, "curr"]
            if len(df_ramps.loc[i, "volt"]) > 10:
                df_ramps.loc[i, "first_r"] = np.mean(df_ramps.loc[i, "volt"][2:10]/df_ramps.loc[i, "curr"][2:10])
            else:
                df_ramps.loc[i, "first_r"] = np.mean(df_ramps.loc[i, "volt"][0:2] / df_ramps.loc[i, "curr"][0:2])

    # df_ramps = find_false_ramps(df_ramps)
    df_ramps["falseramp"] = False
    return df_ramps


def plot_ramps(df_ramps, falsecol=False, rjmax=None, trace=""):
    """Plot ramp cycle DF for ramps below rjmax, optionally with false ramps in magenta color."""
    fig1, ax1 = snp.newfig("ul")
    snp.labs("Voltage (V)", "Resistance ($\Omega$)", "EM Trace - " + trace)
    # Plot the full resistance trace before we drop rows for outliers etc.
    if not rjmax:
        for indx, row in df_ramps.iterrows():
            if row["falseramp"] and falsecol == True:
                ax1.plot(row["volt"], row["res"], marker='s', color="magenta", markersize=5, mew=0.0, label=indx)
            else:
                ax1.plot(row["volt"], row["res"],  mew=0.0, label=indx)
    else:
        start_r = df_ramps.loc[0, "first_r"]
        rmax = start_r + rjmax
        for indx, row in df_ramps.iterrows():
            if row["last_v"] / row["last_i"] <= rmax:
                if row["falseramp"] and falsecol == True:
                    ax1.plot(row["volt"], row["res"], marker='s', color="magenta", markersize=5, mew=0.0, label=indx)
                else:
                    ax1.plot(row["volt"], row["res"], mew=0.0, label=indx)
            else:
                break
    return fig1, ax1


def choose_res_tot_0(df):
    """Plot a ramp cycle DF and have user input the initial total resistance of the device."""
    fig, ax = plot_ramps(df, rjmax=100)
    print("\n\n\n Please determine the initial total resistance for this device. \n\n\n")
    snp.whichtrace()  # Just to get the program to pause on the plot
    res_tot_0 = float(input("Please input the value for initial total resistance in Pc fitting (in Ohms):"))
    plt.close(fig)
    return res_tot_0


def choose_last_ramp(df):
    """Plot ramp cycle DF and have user select the last cycle to be considered in analysis."""
    print("\n\n\n Please select the last valid ramp cycle for analysis. \n\n\n")
    fig, ax = plot_ramps(df)
    last_trace = snp.whichtrace(ax)  # User manually clicks a point on graph
    plt.close(fig)
    last_trace = int(last_trace[0])  # whichtrace() returns a list of string labels
    return last_trace


def choose_good_ramps(df):
    """Plot ramp cycle DF and let user select which cycles will be included in analysis."""
    print(df.head(10))
    fig, ax = plot_ramps(df, falsecol=True)  # Plot all ramp cycles with color coding
    # Let user select which ramp cycles have been incorrectly labeled
    tracdict = {trac.get_label(): trac for trac in ax.lines}  # Will use this later to identify specific traces
    print("\n\n\n Select ramps which should be toggled to/from false ramp status. \n\n\n")
    fliptraces = snp.whichtrace(ax)  # User manually clicks points on graph
    while fliptraces:  # Will be false when the user just clicks the mouse button and list is empty
        for trac in fliptraces:
            df.loc[int(trac), "falseramp"] = True
        fliptraces = snp.whichtrace(ax)  # User manually clicks points on graph

    plt.close(fig)
    return df


######################################################
# REVISE PREPARATION OF SINGLE EM TRACE #####
######################################################
def revise_prep(fname):
    """For the single text file trace, extract ramps, let the user select good ramps and set res_tot_0 and store the
    resulting DF and res_tot_0 in a pickled dict."""

    # Create (or load if it exists) the dict that will store prepped DFs and res_tot_0 for all traces.
    if os.path.isfile("prepped_traces_dict.pickle"):
        with open("prepped_traces_dict.pickle", "rb") as myfile:
            prepped_traces_dict = pickle.load(myfile)
    else:
        prepped_traces_dict = {}

    trace = fname.strip(".txt")
    df = get_trace(fname)
    df = extract_ramps(df)

    # Do Analysis
    res_tot_0 = choose_res_tot_0(df)
    lastrow = choose_last_ramp(df)
    df = df.loc[:lastrow]
    # df = toggle_ramps(df.loc[:lastrow + 1])
    df = choose_good_ramps(df)
    df_allgood = df[df["falseramp"] != True]

    fig, ax = plot_ramps(df, falsecol=True, trace=trace)
    plt.savefig(trace + "_TRACE_FOR_FITTING.png")
    plt.close(fig)

    prepped_traces_dict[trace] = {"r0": res_tot_0, "df": df_allgood}

    # Pickle after every device, b/c sometimes stuff crashes etc.
    with open("prepped_traces_dict.pickle", "wb") as myfile:
        pickle.dump(prepped_traces_dict, myfile)


######################################################
# WRAPPER FOR PLOTTING ALL TRACES (TO DISCARD BAD ONES)
######################################################
def pic_all(path, rjmax=500):
    """Extract ramps and plot trace for all text files in path. Then the user can remove bad traces."""
    fnames = snp.txtfilenames(path)
    for fname in fnames:
        # Prepare Data Frame
        trace = fname.strip(".txt")
        df_trace = get_trace(fname)
        df = extract_ramps(df_trace)
        fig, ax = plot_ramps(df, rjmax=rjmax, trace=trace)
        snp.labs("Voltage (V)", "Resistance ($\Omega$))", "EM Trace - " + trace)
        plt.savefig(trace + "_EM_Rj_below_" + str(rjmax) + ".png")
        plt.close(fig)


######################################################
# WRAPPER FOR PREPARING ALL TRACES #####
######################################################
def prep_all(path):
    """For all text files in path, extract ramps, let the user select good ramps and set res_tot_0 and store the
    resulting DFs and res_tot_0's in a pickled dict."""
    
    # Create (or load if it exists) the dict that will store prepped DFs and res_tot_0 for all traces.
    if os.path.isfile("prepped_traces_dict.pickle"):
        with open("prepped_traces_dict.pickle", "rb") as myfile:
            prepped_traces_dict = pickle.load(myfile)
    else:
        prepped_traces_dict = {}

    # Do preparations on each file
    fnames = snp.txtfilenames(path)
    for fname in fnames:
        print(fname)
        # Prepare Data Frame
        trace = fname.strip(".txt")
        df_trace = get_trace(fname)
        df = extract_ramps(df_trace)

        # Do Analysis
        res_tot_0 = choose_res_tot_0(df)
        lastrow = choose_last_ramp(df)
        df = df.loc[:lastrow]
        # df = toggle_ramps(df.loc[:lastrow + 1])
        df = choose_good_ramps(df)
        df_allgood = df[df["falseramp"] != True]

        fig, ax = plot_ramps(df, falsecol=True, trace=trace)
        plt.savefig(trace + "_TRACE_FOR_FITTING.png")
        plt.close(fig)

        prepped_traces_dict[trace] = {"r0": res_tot_0, "df": df_allgood}
        # Pickle after every device, b/c sometimes stuff crashes etc.
        with open("prepped_traces_dict.pickle", "wb") as myfile:
            pickle.dump(prepped_traces_dict, myfile)


######################################################
# WRAPPER FOR FITTING ALL PREPARED DFS #####
######################################################
def fit_all(path, group, rjmax, temp=298, env="ambient", chip="3-5", rjmin=0):
    """Fit constant power model to all prepared dataframes in path, and store results in new dict and df."""
    # Read in the dictionary of prepped traces (dfs and res_tot_0's) and get list of traces
    with open("prepped_traces_dict.pickle", "rb") as myfile:
        prepped_traces_dict = pickle.load(myfile)
    traces = list(prepped_traces_dict.keys())

    # Create (or load if exists) dict and df that will store fit results on all traces
    dfpickle = "fit_results_df_Rjmax_" + str(rjmax) + "_Rjmin_" + str(rjmin) + ".pickle"
    dictpickle = "fit_results_dict_Rjmax_" + str(rjmax) + "_Rjmin_" + str(rjmin) + ".pickle"
    if os.path.isfile(dfpickle):
        with open(dfpickle, "rb") as myfile:
            fit_df = pickle.load(myfile)
        conforms = fit_df["pc_conforming"].values
        fit_df = fit_df.drop("pc_conforming", axis=1)
    else:
        fit_df = pd.DataFrame(columns=["powj", "rj0", "res_tot_0", "group", "rjmax", "temp", "env", "chip"],
                              index=traces)
    if os.path.isfile(dictpickle):
        with open(dictpickle, "rb") as myfile:
            fit_dict = pickle.load(myfile)
    else:
        fit_dict = {}

    # Do fitting on each trace and store results + demographics in the dict and df
    for trace in traces:
        df = prepped_traces_dict[trace]["df"]
        res_tot_0 = prepped_traces_dict[trace]["r0"]
        rmax = rjmax + res_tot_0
        rmin = rjmin + res_tot_0
        trunc_df = df[rmin <= df["first_r"]]
        trunc_df = trunc_df[trunc_df["first_r"] <= rmax]
        volt = trunc_df["last_v"].values
        curr = trunc_df["last_i"].values
        lsq_fit = do_fit(volt, curr, res_tot_0)

        # Make pretty plots and save them
        fig, ax = plot_fit(trace, volt, curr, lsq_fit, res_tot_0)
        plt.savefig(trace + "_FIT.png")
        plt.close(fig)

        # Save all Results
        fit_dict[trace] = {"rj_max": rjmax, "rj0": lsq_fit.x[1], "powj": lsq_fit.x[0], "res_tot_0": res_tot_0,
                           "group": group, "volt": volt, "curr": curr, "temp": temp, "env": env, "chip": chip,}
        fit_df.loc[trace] = [lsq_fit.x[0], lsq_fit.x[1], res_tot_0, group, rjmax, temp, env, chip]

    # Update the DF storing fit info to reflect which devices "conform" to the model - if this is the first time an
    # analysis was done on these traces then initialize all to "1" meaning they all conform, otherwise use the
    # preexisting conform values. After inspecting the plots of fits, this can be changed by calling set_pc_conform().
    if "conforms" in locals():
        fit_df["pc_conforming"] = conforms
    else:
        fit_df["pc_conforming"] = 1

    with open(dfpickle, "wb") as myfile:
        pickle.dump(fit_df, myfile)
    with open(dictpickle, "wb") as myfile:
        pickle.dump(fit_dict, myfile)


######################################################
# FITTING SINGLE PREPARED DF #####
######################################################
def do_fit(volt, curr, res_tot_0):
    """Do constant power fit to the voltage / current ramp endpoints from a single prepared DF."""
    # Define a function to compute the residuals between the real data and the predicted current
    def residuals(theta, volt, curr, res_tot_0):
        predicted_volt = predict_data(theta, curr, res_tot_0)
        # Note, without .astype() the returned array would actually have data type = "object" and would store pointers
        # to the float values rather than storing the values themselves. This would throw an error in the LSQ.
        return (volt - predicted_volt).astype(np.float64)

    # Fit the model using LSQ loss (alternately, specify loss to get more outlier-robust fitting)
    theta0 = [1e-4, 10]  # first param is critical power, second param is initial Rj
    lsq = least_squares(residuals, theta0, args=(volt, curr, res_tot_0), loss="soft_l1")
    return lsq


def predict_data(theta, curr, res_tot_0):
    """Compute a predicted current given input resistance and model parameters."""
    powj = theta[0]
    rj_init = theta[1]
    predicted_volt = powj/curr + curr*(res_tot_0 - rj_init)
    return predicted_volt


def plot_fit(trace, volt, curr, lsq_fit, res_tot_0):
    """Plot the critical (I,V) points with the constant-power fit overlaid."""
    fig, ax = snp.newfig("ul")
    powj = lsq_fit.x[0]
    rj0 = lsq_fit.x[1]
    rn = np.array(volt/curr)
    rn = rn.astype("float64")
    rjs = rn - rn[0] + rj0
    snp.labs("Junction Resistance ($\Omega$)", "Critical Current (mA)", "EM Onset Points - " + trace)
    ax.plot(rjs, curr*1000, marker="o", color="red", markersize=9, linestyle="", label="data")
    predict_curr = np.sqrt((powj/(rn - res_tot_0 + rj0)).astype("float64"))
    ax.plot(rjs, predict_curr*1000, label="fit ($P_c$ = %.3f mW, $R_0^J$ = %.0f $\Omega$)" %(powj*1000, rj0))
    ax.legend(loc="lower left")

    # Inset the Residuals
    left, bottom, width, height = [0.6, 0.6, 0.25, 0.25]  # Unitless percentages of the fig size. (0,0 is bottom left)
    ax_inset = fig.add_axes([left, bottom, width, height])
    ax_inset.set_title("Residuals of Voltage", fontsize=12)
    ax_inset.set_xlabel("Critical Point", fontsize=12)
    ax_inset.plot(predict_data(lsq_fit.x, curr, res_tot_0) - volt, marker="x", markersize=8, linestyle="", color="red")
    ax_inset.axhline(0, color="black")
    ax_inset.get_xaxis().set_visible(False)
    ax_inset.get_yaxis().set_visible(False)

    # Textbox with more info
    ax.text(0.02, 0.2, "$R_0^{tot}$ = %.0f $\Omega$" % res_tot_0,
            transform=ax.transAxes, fontsize=14, verticalalignment="top")
    return fig, ax


######################################################
# SET CONFORMING STATUS OF SINGLE TRACE #####
######################################################
def set_pc_conform(dfpickle, name="", value=0):
    with open(dfpickle, "rb") as myfile:
        fit_df = pickle.load(myfile)

    if name in fit_df.index:
        fit_df.loc[name, "pc_conforming"] = value
    else:
        print("That trace is not in the dataframe.")

    with open(dfpickle, "wb") as myfile:
        pickle.dump(fit_df, myfile)





######################################################
# PREPARE AND FIT ALL TRACES IN DIR #####
######################################################
# Set values for all relevant variables
chip = "9-11"
group = "40x400"
temp = 95.0
env = "He_gas"

# Move to the place where this data is stored
path = "C:/Users/Sonya/Documents/My Box Files/MolT Project/170113_SDS20_Chip_9-11/EM/40x400"
os.chdir(path)

# Look at pics to discard any really crap traces from being analyzed
pic_all(path)

# Prepare traces for fitting (set initial resistance, choose valid ramp cycles)
prep_all(path)
# Run revise_prep(fname=fname_with_txt_extension) to redo specific devices

# Set values for all relevant variables
rjmax = 200
rjmin = 0
# Fit all prepped traces
fit_all(path, group=group, rjmax=rjmax, temp=temp, env=env, chip=chip, rjmin=rjmin)
# Run set_pc_conform(dfpickle, name="", value=0) to change which fitted devs are considered well described by Pc model



print("debug")
