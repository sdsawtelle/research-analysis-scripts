"""

HAS CURRENT-DEPENDENT SERIES RESISTANCE FITTING!!!

This module provides functionality to plot, prepare and fit EM behavior where traces are taken partially at one
temperature, and then finished at a second temperature. Each EM trace is fitted to the constant-power model for
electromigration. The user is able to restrict what parts of the trace should be included in analysis to exclude, for
instance, later ramp cycles once the junction resistance is high and the device is unstable. Once the trace has been
culled in this fashion, the analysis consists of identifying the current and voltage points where EM occurs, and then
fitting that set of points using a constant critical power model.

The data for this experiment consist of EM text files where each device (specified by a unique feedthrough and port
combination, as in "FT1_0102") should have two EM trace files associated with it, one containing "_partial" in the name
and one containing "_final." For full analysis we will compare behavior between these two files for each device.

Specific usage is to first separate the traces by partial vs final EM into two foldes by calling split(). The user is
encouraged to refer to experimental notes to eliminate all known garbage traces from the directory. This includes traces
where the device was "stuck" or had an uncontrolled break.

After separation begin independent analysis of the partial EM traces. Call pic_all() to make plots of all traces, so
that crappy ones can be removed from the folder. Then call prep_all() to do all initial preparation of the good traces
like selecting ramps for analysis and setting res_tot_0. This creates and pickles a dict of dicts whose keys are the
trace names and whose values are a dict of the prepared DF and the res_tot_0 value. Optionally now you can call
revise_prep() if you want to revise the preparation of any specific trace - it will update the pickled dictionary.

Now call fit_all() with kwargs rjmin and rjmax to fit the ramps between rjmin and rjmax for each trace to the constant
power model. This function creates and pickles a dataframe whose indices are the trace names and whose values are
various demographic info about the device as well as the fit results. It also creates and pickles a dict of dicts with
the same information. The pickled dataframe has a column 'conforms' which denotes whether the device appears to conform
to the constant power model or not. This is initialized to 1 for all traces but values can later be changed by calling
set_pc_conforms(fname). The dict and df pickles are named with the specific rjmin and rjmax values that were used for
the fitting, so users can try different values of these kwargs and not overwrite previous results.

Finally, repeat the analysis for the final EM traces, but instead of calling fit_all() you should call
fit_all_constrained(). The difference is that the constrained fitting function will look for a neighboring directory
called "PARTIAL", it will open the pickled dataframe in this directory and find the fitted Rj0 and series resistance for
each device. Then it will use these values to constrain the fits to Pc for the final EM traces.

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
import re

# Boilerplate settings
matplotlib.interactive = True
snp.prettyplot(matplotlib)
plt.close("all")



######################################################
# PLOTTING AND PICTURES #####
######################################################
def plot_ramps(df_ramps, fitcol=False, deltarj=None, type="partial", trace="", newfig=True):
    """Plot ramp cycle DF for ramps below rjmax, optionally with false ramps in magenta color."""
    if newfig:
        fig1, ax1 = snp.newfig("ul")
        snp.labs("Voltage (V)", "Resistance ($\Omega$)", "EM Trace - " + trace)
    else:
        ax1 = plt.gca()
        fig1 = plt.gcf()

    # Plot the full trace (rather than zooming in to the partial/final join region)
    if not deltarj:
        for indx, row in df_ramps.iterrows():
            if row["false_ramp"] and fitcol == True:
                ax1.plot(row["volt"], row["res"], marker='s', color="magenta", markersize=4, mew=0.0,
                         label=indx)
            else:
                ax1.plot(row["volt"], row["res"], mew=0.0, label=indx)
                
    # Only plot zoomed in to the join region between partial and final traces
    else:
        # Plot each ramp cycle in a different color (or in magenta if false ramp)
        for indx, row in df_ramps.iterrows():
            if row["false_ramp"] and fitcol == True:
                ax1.plot(row["volt"], row["res"], marker='s', color="magenta", markersize=4, mew=0.0,
                         label=indx)
            else:
                ax1.plot(row["volt"], row["res"], mew=0.0, label=indx)
                
        if type == "partial":
            minres = df_ramps.iloc[len(df_ramps) - 1, df_ramps.columns.get_loc("res")][0] - deltarj
            ax1.set_ylim([minres, ax1.get_ylim()[1]])
        if type == "final":
            maxres = df_ramps.iloc[0, df_ramps.columns.get_loc("res")][0] + deltarj + 5
            ax1.set_ylim([ax1.get_ylim()[0], maxres])
            
            
    return fig1, ax1


def plot_traces_combined(path, zoom=False, maxrj=10, deltarj=15):
    """For every unique device plot the full partial EM, full final EM, and selected ramps for fitting from both partial
    and final."""
    with open("fit_results_dict_deltarj_%i.pickle" % (deltarj,), "rb") as myfile:
        fit_dict = pickle.load(myfile)

    # Get unique feedthrough/port identifiers corresponding to unique devices
    fnames = snp.txtfilenames(path)
    ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    ports = list(set(ports))

    # For each unique device identifier, do the ramp analysis for both traces
    for port in ports:
        fig, ax = snp.newfig()
        snp.labs("Voltage (V)", "Resistance ($\Omega$)", "EM Device - " + port)
        traces = [fname for fname in fnames if port in fname]

        for trace in traces:
            full_df = get_trace(trace)
            ax.plot(full_df["voltage"], full_df["resistance"], linestyle="None", markersize=1, color="black")
            fit_df = fit_dict[trace.strip(".txt")]["df"]
            fit_df["false_ramp"] = ~ fit_df["false_ramp"]
            plot_ramps(fit_df, fitcol=True, newfig=False)

            if zoom:
                if "partial" in trace:
                    minres = fit_df.iloc[-1, fit_df.columns.get_loc("res")][0] - maxrj
                    ax.set_ylim([minres, ax.get_ylim()[1]])
                if "final" in trace:
                    maxres = fit_dict[trace.strip(".txt")]["res_tot_0"] + maxrj + 5
                    ax.set_ylim([ax.get_ylim()[0], maxres])
        plt.savefig("TRACES_FOR_FIT_%s.png" % (port,))
        plt.close(fig)


def plot_fits_combined(path):
    """For every unique device plot the fit from the partial and final EM's side-by-side."""
    with open("fit_results_df.pickle", "rb") as myfile:
        fitdf = pickle.load(myfile)

    with open("fit_results_dict.pickle", "rb") as myfile:
        fitdict = pickle.load(myfile)

    # Get unique feedthrough/port identifiers corresponding to unique devices
    fnames = snp.txtfilenames(path)
    ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    ports = list(set(ports))

    # For each unique device identifier, do the ramp analysis for both traces
    for port in ports:
        fig, axs = plt.subplots(ncols=2, figsize=[14, 6])
        axs = np.ravel(axs)

        # Get the partial and final traces for this device, sort so partial is first
        traces = [fname.strip(".txt") for fname in fnames if port in fname]
        traces = [tr for tr in traces if "partial" in tr] + [tr for tr in traces if "final" in tr]
        # ctimes = [os.path.getctime(fname) for fname in traces]
        # traces = [trace.strip(".txt") for (ctime, trace) in sorted(zip(ctimes, traces))]

        res = 1
        for idx, trace in enumerate(traces):
            df = fitdf.loc[trace, :]
            dic = fitdict[trace]
            res = dic["crit_pow"] / res
            plot_fit_variableseries(fig, axs[idx], trace, dic["volt"], dic["curr"], dic["crit_pow"], dic["rj0"], dic["r_series"])

        fig.suptitle("Fit to $P_c$ Model: Ratio = %.2f" % (res,))
        plt.savefig("FITS_%s" % (port,))
    plt.close("all")


def pic_all(path, zoom=True, deltar=25):
    """For every unique device plot the full partial EM, full final EM, zoomed in to the region join region"""

    # Get unique feedthrough/port identifiers corresponding to unique devices
    fnames = snp.txtfilenames(path)
    ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    ports = list(set(ports))

    # For each unique device identifier, do the ramp analysis for both traces
    for port in ports:
        fig, ax = snp.newfig()
        snp.labs("Voltage (V)", "Resistance ($\Omega$)", "EM Device - " + port)
        traces = [fname for fname in fnames if port in fname]

        for trace in traces:
            full_df = get_trace(trace)
            ax.plot(full_df["voltage"], full_df["resistance"], linestyle="None")
            ramps_df = extract_ramps(full_df)

            if zoom:
                if "partial" in trace:
                    minres = ramps_df.loc[len(ramps_df)-1, "res"][0] - deltar
                    ax.set_ylim([minres, ax.get_ylim()[1]])
                if "final" in trace:
                    maxres = ramps_df.loc[0, "res"][0] + deltar + 5
                    ax.set_ylim([ax.get_ylim()[0], maxres])
        # plt.savefig("TRACES_FOR_FIT_%s.png" % (port,))
        # plt.close(fig)



######################################################
# PREPARING FOR FIT AND EXTRACTING RAMP CYCLES  #####
######################################################
def prep_all(path):
    """For all unique devices in path, let the user choose which ramps to fit for both the partial and final EM traces
    of that device and set res_tot_0 from the first chosen ramp of the partial trace. Save the chosen ramps as DFs and
    pickle them with res_tot_0 as a dict."""

    # Get unique feedthrough/port identifiers corresponding to unique devices
    fnames = snp.txtfilenames(path)
    ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    ports = list(set(ports))

    # For each unique device identifier, do the ramp analysis for both traces
    for port in ports:
        print("--------------------------------------------------------")
        print("--------------------------------------------------------")
        print("Prepare Traces for Device %s" %(port,))
        print("--------------------------------------------------------")
        print("--------------------------------------------------------")
        prep_device(port, path)


def prep_device(port, path, deltarj=20):
    # Create (or load if it exists) the dict that will store prepped DFs and res_tot_0 for all traces.
    if os.path.isfile("prepped_traces_dict.pickle"):
        with open("prepped_traces_dict.pickle", "rb") as myfile:
            prepped_traces_dict = pickle.load(myfile)
    else:
        prepped_traces_dict = {}

    # Get the two traces for this device (partial and final
    fnames = snp.txtfilenames(path)
    traces = [fname for fname in fnames if port in fname]
    traces = [(tr, "partial", "final") for tr in traces if "partial" in tr] + \
             [(tr, "final", "partial") for tr in traces if "final" in tr]  #Order them correctly
    
    # Do ramp selection and res_tot_0 choice for both the partial and final traces
    for tr in traces:
        fname, this_tr, other_tr = tr  # Unpack the useful quantities for this trace

        # Prepare Data Frames
        df = extract_ramps(get_trace(fname))
        df_other = get_trace([tr[0] for tr in traces if other_tr in tr[0]][0])

        # Get the temperature from the header
        with open(fname, "r") as f:
            tempstr = [val for val in f.readlines(1)[0].split(";") if "TEMP" in val]  # Get the header and chunk it
            tempstr = re.search("[\d]+[.]*[\d]+", tempstr[0])
            temp = float(tempstr.group(0))

        # # If this is a final EM trace, select a "first good ramp"
        # if this_r == "final":
        #     idx = choose_first_ramp(df)

        # Choose first and last valid ramps
        if "final" in this_tr:
            df = df.iloc[choose_first_ramp(df, df_other):, :]
            df = df.iloc[:choose_last_ramp(df), :]

        if "partial" in this_tr:
            df = df.iloc[choose_first_ramp(df, df_other):, :]

        # Select ramps for fitting for this trace (plot the other trace together with it)
        df = choose_false_ramps(df, df_other, type=this_tr, deltarj=deltarj + 5)
        if this_tr == "partial":  # If this is a partial EM trace, discard the last "fake" ramp
            df.iloc[-1, df.columns.get_loc("false_ramp")] = True
        df_allgood = df[df["false_ramp"] == False]

        # Save the prepared dataframe and res_tot_0 to the existing dictionary and re-pickle
        prepped_traces_dict[fname.strip(".txt")] = {"df": df_allgood, "temp": temp}
        with open("prepped_traces_dict.pickle", "wb") as myfile:  # Pickle after every dev, b/c sometimes stuff crashes
            pickle.dump(prepped_traces_dict, myfile)


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
    rampbackpoints.iloc[0] = rampbackpoints.iloc[
        -1] = True  # Manually add 2nd and last data points as ramp back points
    rampindx = rampbackpoints[
        rampbackpoints].index.tolist()  # Get Series index values for all the ramp back points.
    nramps = len(rampindx) - 1  # Recall each ramp consumes two rampindex points, a start and stop

    # Split the trace into ramp cycles and store relevant quantities in a DF
    df_ramps = pd.DataFrame(index=range(0, nramps, 1),
                            columns=["volt", "curr", "res", "last_v", "last_i", "first_r", "false_ramp"])
    for i in range(0, nramps, 1):
        start = rampindx[i] + 1  # Note this discards the first data point of the first ramp cycle... it's OK
        stop = rampindx[i + 1]
        if start == stop:  # Discard any ramps which are only one point long (happens at end of trace sometimes)
            df_ramps.drop(i, axis=0, inplace=True)
        else:
            df_ramps.loc[i, "volt"] = np.array(df.loc[start:stop:1, "voltage"])
            df_ramps.loc[i, "last_v"] = df_ramps.loc[i, "volt"][-1]
            df_ramps.loc[i, "curr"] = np.array(df.loc[start:stop:1, "current"])
            df_ramps.loc[i, "last_i"] = df_ramps.loc[i, "curr"][-1]
            df_ramps.loc[i, "res"] = df_ramps.loc[i, "volt"] / df_ramps.loc[i, "curr"]
            if len(df_ramps.loc[i, "volt"]) > 10:
                df_ramps.loc[i, "first_r"] = np.mean(
                    df_ramps.loc[i, "volt"][2:10] / df_ramps.loc[i, "curr"][2:10])
            else:
                df_ramps.loc[i, "first_r"] = np.mean(
                    df_ramps.loc[i, "volt"][0:2] / df_ramps.loc[i, "curr"][0:2])

    # df_ramps = find_false_ramps(df_ramps)
    df_ramps["false_ramp"] = False
    return df_ramps


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
    print("\n\n\n Please select the LAST valid ramp cycle for analysis. \n\n\n")
    fig, ax = plot_ramps(df)
    last_trace = snp.whichtrace(ax)  # User manually clicks a point on graph
    plt.close(fig)
    last_trace = int(last_trace[0])  # whichtrace() returns a list of string labels
    return last_trace


def choose_first_ramp(df, df_other):
    """Plot ramp cycle DF and have user select the last cycle to be considered in analysis."""
    print("\n\n\n Please select the FIRST valid ramp cycle for analysis. \n\n\n")
    fig, ax = plot_ramps(df, df_other, type="final")
    first_trace = snp.whichtrace(ax)  # User manually clicks a point on graph
    plt.close(fig)
    first_trace = int(first_trace[0])  # whichtrace() returns a list of string labels
    return first_trace


def choose_false_ramps(df, other_df, type="partial", deltarj=30):
    """Plot ramp cycle DF and let user select which ramps are "false" rampbacks."""
    # print(df.head(10))
    fig, ax = plot_ramps(df, fitcol=False, deltarj=deltarj, type=type)  # Plot all ramp cycles with color coding
    ax.plot(other_df["voltage"], other_df["resistance"], linestyle="None")
    # Let user select which ramp cycles have been incorrectly labeled
    tracdict = {trac.get_label(): trac for trac in ax.lines}  # Will use this later to identify specific traces
    print("\n\n\n Select any ramps which are false ramp backs. \n\n\n")
    fliptraces = snp.whichtrace(ax)  # User manually clicks points on graph
    while fliptraces:  # Will be false when the user just clicks the mouse button and list is empty
        for trac in fliptraces:
            df.loc[int(trac), "false_ramp"] = True
        fliptraces = snp.whichtrace(ax)  # User manually clicks points on graph

    plt.close(fig)
    return df



######################################################
# FITTING PARTIAL EM TRACES #####
######################################################
def fit_all_variableseries(path, ratios, group, env, chip, deltarj):
    """Fit constant power model to all prepared dataframes in path, and store results in new dict and df."""
    # Read in the dictionary of prepped traces (dfs and res_tot_0's) and get list of traces
    with open("prepped_traces_dict.pickle", "rb") as myfile:
        prepped_traces_dict = pickle.load(myfile)
    all_traces = list(prepped_traces_dict.keys())

    # Create (or load if exists) dict and df that will store fit results on all traces
    dfpickle = "fit_results_df_deltarj_%i.pickle" % (deltarj,)
    dictpickle = "fit_results_dict_deltarj_%i.pickle" % (deltarj,)
    if os.path.isfile(dfpickle):
        with open(dfpickle, "rb") as myfile:
            fit_df = pickle.load(myfile)
        conforms = fit_df["pc_conforming"].values  # Save the conforming designation, will write it back in later
        fit_df = fit_df.drop("pc_conforming", axis=1)
    else:
        fit_df = pd.DataFrame(columns=["crit_pow", "r_series", "rj0", "res_tot_0", "group", "trace_type",
                                       "temp", "env", "chip"], index=all_traces)
    if os.path.isfile(dictpickle):
        with open(dictpickle, "rb") as myfile:
            fit_dict = pickle.load(myfile)
    else:
        fit_dict = {}

    ########### FIT PARTIAL EM TRACES ##################
    # Restrict to just the partial EM traces
    traces = [trace for trace in all_traces if "partial" in trace]

    # Do fitting on each trace and store results + demographics in the dict and df
    for trace in traces:
        # Restrict to only the ramps within the desired resistance range
        df = prepped_traces_dict[trace]["df"]
        temp = prepped_traces_dict[trace]["temp"]
        cutoff = df.iloc[-1, df.columns.get_loc("first_r")] - deltarj
        df = df[df["first_r"] > cutoff]
        volt = df["last_v"].values
        curr = df["last_i"].values
        res_tot_0 = df.iloc[0, df.columns.get_loc("first_r")]

        # Fit for the slope of R vs. I to get factor for variable R_series
        curr_dRdT = df.iloc[0, df.columns.get_loc("curr")][:30]
        curr_zero = curr_dRdT[0]
        res_dRdT = df.iloc[0, df.columns.get_loc("res")][:30]
        def residuals_dRdT(theta, curr, res):
            a, b = theta
            return res - (a * curr + b)
        lsq_eta = least_squares(residuals_dRdT, [6000, 400], args=(curr_dRdT, res_dRdT), loss="soft_l1")
        alpha = lsq_eta.x[0]

        # Do the Fit and save results
        lsq_fit = do_fit_variableseries(volt, curr, alpha, curr_zero)
        fit_dict[trace] = {"df": df, "r_series": lsq_fit.x[1], "crit_pow": lsq_fit.x[0],
                           "rj0": res_tot_0 - lsq_fit.x[1], "res_tot_0": res_tot_0, "group": group,
                           "trace_type": "partial", "volt": volt, "curr": curr, "temp": temp, "env": env, "chip": chip}
        fit_df.loc[trace] = [lsq_fit.x[0], lsq_fit.x[1], res_tot_0 - lsq_fit.x[1],
                             res_tot_0, group, "partial", temp, env, chip]

    ########### FIT FINAL EM TRACES ##################
    # Restrict to just the final EM traces
    traces = [trace for trace in all_traces if "final" in trace]

    # Do fitting on each trace and store results + demographics in the dict and df
    for trace in traces:
        # Prepare ramp cycles for fitting
        df = prepped_traces_dict[trace]["df"]
        temp = prepped_traces_dict[trace]["temp"]
        cutoff = df.iloc[0, df.columns.get_loc("first_r")] + deltarj
        df = df[df["first_r"] < cutoff]
        volt = df["last_v"].values
        curr = df["last_i"].values
        res_tot_0 = df.iloc[0, df.columns.get_loc("first_r")]

        # Get the fitted rj0 and r_series from the corresponding partial trace and scale them
        port = "_".join(trace.split("_")[-2:])
        rj0 = fit_df.loc[fit_df.index.str.contains("partial_" + port), "rj0"].values[0] * ratios[port]
        r_series = fit_df.loc[fit_df.index.str.contains("partial_" + port), "r_series"].values[0] * ratios[port]


        # Fit for the slope of R vs. I to get factor for variable R_series
        curr_dRdT = df.iloc[0, df.columns.get_loc("curr")][:30]
        res_dRdT = df.iloc[0, df.columns.get_loc("res")][:30]
        def residuals_dRdT(theta_eta, curr, res):
            a, b = theta_eta
            return res - (a * curr + b)
        lsq_eta = least_squares(residuals_dRdT, [6000, 400], args=(curr_dRdT, res_dRdT), loss="soft_l1")
        alpha = lsq_eta.x[0]



        # Use scaled r_series to fit and save results.
        lsq_fit = do_fit_variableseries(volt, curr, alpha, curr_zero, r_series)
        # lsq_fit = do_fit_variableseries(volt, curr, alpha)

        fit_dict[trace] = {"df": df, "r_series": r_series, "crit_pow": lsq_fit.x[0],
                           "rj0": rj0, "res_tot_0": res_tot_0, "group": group,
                           "trace_type": "final", "volt": volt, "curr": curr, "temp": temp, "env": env, "chip": chip}
        fit_df.loc[trace] = [lsq_fit.x[0], r_series, rj0,
                             res_tot_0, group, "final", temp, env, chip]

    # If the "conforms" column already existed then we set it to the pre-existing values
    if "conforms" in locals():
        fit_df["pc_conforming"] = conforms
    else:
        fit_df["pc_conforming"] = 1

    with open(dfpickle, "wb") as myfile:
        pickle.dump(fit_df, myfile)
    with open(dictpickle, "wb") as myfile:
        pickle.dump(fit_dict, myfile)


def do_fit_variableseries(volt, curr, eta, eta_curr, r_series=None):
    """Do constant power fit to the voltage / current ramp endpoints from a single prepared DF."""
    # If no r_series is passed, fit where p_c and r_series are both free parameters
    if not r_series:
        # Define a function to compute the residuals between the real data and the predicted current
        def residuals(theta, eta, eta_curr, volt, curr):
            p_c = theta[0]
            r_series = theta[1]
            predicted_volt = predict_volt_variableseries(p_c, eta, eta_curr, r_series, curr)
            # Note, without .astype() the returned array would actually have data type = "object" and would store pointers
            # to the float values rather than storing the values themselves. This would throw an error in the LSQ.
            return (volt - predicted_volt).astype(np.float64)

        # Fit the model using LSQ loss (alternately, specify loss to get more outlier-robust fitting)
        theta0 = [1e-4, 10]  # first param is critical power, second param is initial Rj
        lsq = least_squares(residuals, theta0, args=(eta, eta_curr, volt, curr), loss="soft_l1")

    # If a value for r_series is passed, then only treat p_c as free
    if r_series:
        # Define a function to compute the residuals between the real data and the predicted current
        def residuals(theta, r_series, eta, eta_curr, volt, curr):
            p_c = theta[0]
            predicted_volt = predict_volt_variableseries(p_c, eta, eta_curr, r_series, curr)
            # Note, without .astype() the returned array would actually have data type = "object" and would store pointers
            # to the float values rather than storing the values themselves. This would throw an error in the LSQ.
            return (volt - predicted_volt).astype(np.float64)

        # Fit the model using LSQ loss (alternately, specify loss to get more outlier-robust fitting)
        theta0 = [1e-4]  # first param is critical power, second param is initial Rj
        lsq = least_squares(residuals, theta0, args=(r_series, eta, eta_curr, volt, curr), loss="soft_l1")

    return lsq


def predict_volt_variableseries(p_c, eta, eta_curr, r_series, curr):
    """Compute a predicted current given input resistance and model parameters."""
    adjusted_r_series = r_series + eta * (curr - eta_curr)
    predicted_volt = p_c / curr + curr * adjusted_r_series
    return predicted_volt


def plot_fit_variableseries(fig, ax, trace, volt, curr, p_c, rj0, r_series):
    """Plot the critical (I,V) points with the constant-power fit overlaid."""
    # fig, ax = snp.newfig("ul")
    rn = np.array(volt / curr)
    rn = rn.astype("float64")
    rjs = rn - rn[0] + rj0
    ax.set_xlabel("Junction Resistance ($\Omega$)")
    ax.set_ylabel("Critical Current (mA)")
    if "partial" in trace:
        label = "data initial EM"
    else:
        label = "data final EM"
    ax.plot(rjs, curr * 1000, marker="o", color="red", markersize=9, linestyle="", label=label)
    predict_curr = np.sqrt((p_c / (rn - r_series)).astype("float64"))
    ax.plot(rjs, predict_curr * 1000, label="fit ($P_c$ = %.3f mW, $R_0^J$ = %.0f $\Omega$)" % (p_c * 1000, rj0))
    ax.legend(loc="lower left")

    # # Inset the Residuals
    # left, bottom, width, height = [0.6, 0.6, 0.25, 0.25]  # Unitless percentages of the fig size. (0,0 is bottom left)
    # ax_inset = fig.add_axes([left, bottom, width, height])
    # ax_inset.set_title("Residuals of Voltage", fontsize=12)
    # ax_inset.set_xlabel("Critical Point", fontsize=12)
    # ax_inset.plot(predict_volt_variableseries(p_c, r_series, curr) - volt, marker="x", markersize=8, linestyle="", color="red")
    # ax_inset.axhline(0, color="black")
    # ax_inset.get_xaxis().set_visible(False)
    # ax_inset.get_yaxis().set_visible(False)

    # Textbox with more info
    ax.text(0.02, 0.2, "$R_0^{tot}$ = %.0f $\Omega$" % (rj0 + r_series,),
            transform=ax.transAxes, fontsize=14, verticalalignment="top")
    return ax




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
# GET RATIOS OF RESISTANCE BETWEEN END OF PARTIAL EM AND BEGINNING OF FINAL EM #####
######################################################
def resistance_ratios_from_iv(path):
    # Get unique feedthrough/port identifiers corresponding to unique devices
    # path = "C:/Users/Sonya/Documents/My Box Files/MolT Project/170112_SDS20_Chip_7-11/IV_125K"
    fnames = snp.txtfilenames(path)
    ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    ports = list(set(ports))

    # For each unique identifier, compare ratio of its two IV files
    ratios = {}
    for port in ports:
        traces = [os.path.join(path, fname) for fname in fnames if port in fname]
        mtimes = [os.path.getmtime(fname) for fname in traces]
        [trace for (mtime, trace) in sorted(zip(mtimes, traces))]
        df = pd.read_csv(traces[0], header=0)
        df.columns = ["voltage", "current"]  # Some headers have extra spaces
        df1 = pd.read_csv(traces[1], header=0)
        df1.columns = ["voltage", "current"]  # Some headers have extra spaces
        res1 = df1["voltage"] / df1["current"]
        res0 = df["voltage"] / df["current"]
        ratios[port] = (res1[-10:] / res0[-10:]).mean()

    # Print the results
    for item in ratios.items():
        print(item)
    print("Mean Ratio is %.2f +/- %.2f" % (np.mean(list(ratios.values())), np.std(list(ratios.values()))))
    return ratios


def resistance_ratios_from_em(path):
    """Partial EM traces always end with a "semi-ramp" which is the first 25 points of a ramp cycle. We want to find the
    corresponding voltage points on the first ramp of the final EM trace, in order to estimate the
    environmental-temperature-induced increase in resistance."""
    
    # Get unique feedthrough/port identifiers corresponding to unique devices
    fnames = snp.txtfilenames(path)
    ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    ports = list(set(ports))

    # For each unique device, calculate the ratio of resistance going from T1 to T2
    ratios = {}
    for port in ports:
        # Order the partial and final traces so we know which direction to take the ratio
        traces = [os.path.join(path, fname) for fname in fnames if port in fname]
        traces = [tr for tr in traces if "partial" in tr] + [tr for tr in traces if "final" in tr]
        
        # Read in DFs for partial and final EM, pull the last ramp cycle of partial EM (a "semi-ramp")
        df_partial = extract_ramps(get_trace(traces[0]))
        df_final = extract_ramps(get_trace(traces[1]))
        # Get current and resistance on the last semi-ramp of the partial EM trace
        ip = df_partial.iloc[-1:, 1].values[0]
        rp = df_partial.iloc[-1:, 2].values[0]  
        
        # Look through ramp cycles of final EM, starting with first, to find a match to the semi-ramp voltage
        ramp = 0
        loopflag = 1
        while loopflag:
            print("trying")
            iff = df_final.iloc[ramp, 1]  # Get the current on the first ramp of final EM
            rf = df_final.iloc[ramp, 2]  # Get the resistance on the first ramp of final EM
            start_idx = np.argmin(np.abs(iff - ip[0]))
            # If the best "match" is not within 1 mV then we need to look in the next ramp cycle
            if np.abs((iff - ip[0])[start_idx]) <= 0.00005:
                loopflag = 0
                nsteps = min(len(iff) - start_idx, 24)
                stop_idx = start_idx + nsteps
                # Note, in case the first matching ramp from final EM stops in the middle of the last ramp of partial
                # EM, we need to potentially truncate ip
                res_ratios = rf[start_idx:stop_idx] / rp[:nsteps]
            ramp += 1
        ratios[port] = res_ratios.mean()
        # ratios[port] = 1

    # Print the results
    for item in ratios.items():
        print(item)
    print("RESISTANCE Mean Ratio is %.2f +/- %.2f" % (np.mean(list(ratios.values())), np.std(list(ratios.values()))))
    return ratios


def analyze_pc(path, deltarj):
    """Make a DF where each row is a device and it stores pc for T1 and T2."""
    with open("fit_results_df_deltarj_%i.pickle" %(deltarj,), "rb") as myfile:
        fitdf = pickle.load(myfile)

    with open("fit_results_dict_deltarj_%i.pickle" %(deltarj,), "rb") as myfile:
        fitdict = pickle.load(myfile)

    # Get unique feedthrough/port identifiers corresponding to unique devices
    fnames = snp.txtfilenames(path)
    ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    ports = list(set(ports))
    df = pd.DataFrame(columns=["T1", "T2", "T_ratio", "pc1", "pc2", "p_ratio", "r_series_T1", "rj0_T1",
                               "ratio_RT1_RT2", "group", "env", "chip"], index=ports)
    # For each unique device identifier, do the ramp analysis for both traces
    for port in ports:
        traces = [fname.strip(".txt") for fname in fnames if port in fname]
        partial = fitdict[[trace for trace in traces if "partial" in trace][0]]
        final = fitdict[[trace for trace in traces if "final" in trace][0]]
        pratio = final["crit_pow"] / partial["crit_pow"]
        tratio = final["temp"] / partial["temp"]
        df.loc[port, ["T1", "pc1", "r_series_T1", "rj0_T1", "group", "env", "chip"]] = [partial["temp"],
                        partial["crit_pow"], partial["r_series"], partial["rj0"], partial["group"],
                        partial["env"], partial["chip"]]
        df.loc[port, ["T2", "pc2", "T_ratio", "p_ratio"]] = [final["temp"], final["crit_pow"], tratio, pratio]
        df.loc[port, "ratio_RT1_RT2"] = ratios[port]

    with open("pc_analysis.pickle", "wb") as myfile:
        pickle.dump(df, myfile)

    print(df["p_ratio"])
    print("CRITICAL POWER Mean Ratio is %.2f +/- %.2f" % (df["p_ratio"].mean(), df["p_ratio"].std()))


def aggregate_pc(paths):
    df = pd.DataFrame()
    for path in paths:
        dfpath = path + "/pc_analysis.pickle"
        with open(dfpath, "rb") as myfile:
            temp = pickle.load(myfile)
        df = pd.concat([df, temp], axis=0)

    fig, ax = plt.subplots()
    snp.labs("Ratio of T2/T1", "Ratio of p2/p1", "Critical Power Ratios for Varitemp EM")
    cm = snp.cmap(ncols = len(df["group"].unique()))
    for grp in df["group"].unique():
        ax.scatter(df.loc[df["group"] == grp, "T_ratio"], df.loc[df["group"] == grp, "p_ratio"], label=grp, c=next(cm))
    ax.legend(loc="upper center")
    return df








paths = [
"C:/Users/Sonya/Documents/My Box Files/MolT Project/161230_SDS20_Chip_11-10/161230_Chip_11-10_EM_95K/30x10",
"C:/Users/Sonya/Documents/My Box Files/MolT Project/170101_SDS20_Chip_6-9/170101_Chip_6-9_EM_95K/30x10",
"C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/170114_Chip_7-11_EM_95K/30x10",
"C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/170114_Chip_7-11_EM_95K/50x400",
"C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/170114_Chip_7-11_EM_125K/30x10",
"C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/170114_Chip_7-11_EM_125K/50x400"]

datafiles = [
["C:/Users/Sonya/Documents/My Box Files/MolT Project/161230_SDS20_Chip_11-10/161230_Chip_11-10_EM_95K/30x10", "11-10", "30x10"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170101_SDS20_Chip_6-9/170101_Chip_6-9_EM_95K/30x10", "6-9", "30x10"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/170114_Chip_7-11_EM_95K/30x10", "7-11", "30x10"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/170114_Chip_7-11_EM_95K/50x400", "7-11", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/170114_Chip_7-11_EM_125K/30x10", "7-11", "30x10"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/170114_Chip_7-11_EM_125K/50x400", "7-11", "50x400"]
]


# datafiles = [
# ["C:/Users/Sonya/Documents/My Box Files/MolT Project/170101_SDS20_Chip_6-9/170101_Chip_6-9_EM_95K/30x10", "6-9", "30x10"],
# ]

######################################################
# PREPARE ALL TRACES #####
######################################################
for dataf in datafiles:
    # Move to the place where this data is stored
    path, chip, group = dataf
    os.chdir(path)

    print("-----------------------------------------------------")
    print("Analysis for Chip %s with deltarj = %i" % (chip, deltarj))
    print("-----------------------------------------------------")

    env = "He_gas"

    pic_all(path)
    plt.pause(0.001)
    input("Press enter when you have finished discarding weird traces.")
    plt.close("all")

    prep_all(path, deltarj=30)  # Make this large enough to catch all false ramps that might possibly be used


######################################################
# FIT ALL TRACES #####
######################################################
deltarj = 15
for dataf in datafiles:
    if chip in ["11-10", "6-9"]:
        fnames = snp.txtfilenames(path)
        ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
        ports = list(set(ports))
        ratios = {port: 1 for port in ports}
    else:
        ratios = resistance_ratios_from_em(path)

    fit_all_variableseries(path, ratios, group=group, env=env, chip=chip, deltarj=deltarj)
    plot_traces_combined(path, zoom=True, maxrj=deltarj + 15, deltarj=deltarj)
    # plot_fits_combined(path)
    analyze_pc(path, deltarj)




datafiles = [
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/test", "7-11", "50x400"]
]


deltarj = 20
for dataf in datafiles:
    # Move to the place where this data is stored
    path, chip, group = dataf
    os.chdir(path)
    print("-----------------------------------------------------")
    print("Analysis for Chip %s with deltarj = %i" % (chip, deltarj))
    print("-----------------------------------------------------")

    # Set values for all relevant variables
    env = "He_gas"
    prep_all(path)  # Make this large enough to catch all false ramps that might possibly be used


    if chip in ["11-10", "6-9"]:
        fnames = snp.txtfilenames(path)
        ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
        ports = list(set(ports))
        ratios = {port: 1 for port in ports}
    else:
        ratios = resistance_ratios_from_em(path)

    fit_all_variableseries(path, ratios, group=group, env=env, chip=chip, deltarj=deltarj)
    # plot_traces_combined(path, zoom=True, maxrj=deltarj + 15, deltarj=deltarj)
    # plot_fits_combined(path)
    analyze_pc(path, deltarj)



aggregate_pc([df[0] for df in datafiles])


print("debug")


