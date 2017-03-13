"""This module provides functionality to plot, prepare and fit EM behavior where traces are taken partially at one
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
from sklearn.metrics import r2_score
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
# PLOTTING AND IMAGING #####
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


def plot_fits_combined(path, deltarj=15):
    """For every unique device plot the fit from the partial and final EM's side-by-side."""
    with open("fit_results_df_deltarj_%i.pickle" % (deltarj,), "rb") as myfile:
        fitdf = pickle.load(myfile)

    with open("fit_results_dict_deltarj_%i.pickle" % (deltarj,), "rb") as myfile:
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
            plot_fit(fig, axs[idx], trace, dic["volt"], dic["curr"], dic["crit_pow"], dic["rj0"], dic["r_series"])

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


def plot_trace_fit_combo(path, zoom=True, maxrj=10, deltarj=25):
        with open("fit_results_df_deltarj_%i.pickle" % (deltarj,), "rb") as myfile:
            fitdf = pickle.load(myfile)

        with open("fit_results_dict_deltarj_%i.pickle" % (deltarj,), "rb") as myfile:
            fitdict = pickle.load(myfile)

        # Get unique feedthrough/port identifiers corresponding to unique devices
        fnames = snp.txtfilenames(path)
        ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
        ports = list(set(ports))

        # For each unique device identifier, do the ramp analysis for both traces
        for port in ports:
            fig = plt.figure(1, figsize=[14, 8])  # create one of the figures that must appear with the chart
            axs = [plt.subplot2grid((2, 8), (0, 5), colspan=3), plt.subplot2grid((2, 8), (1, 5), colspan=3),
                   plt.subplot2grid((2, 8), (0, 0), colspan=5, rowspan=2)]
            # Get the partial and final traces for this device, sort so partial is first
            traces = [fname.strip(".txt") for fname in fnames if port in fname]
            traces = [tr for tr in traces if "partial" in tr] + [tr for tr in traces if "final" in tr]



            res = 1
            for idx, trace in enumerate(traces):
                df = fitdf.loc[trace, :]
                dic = fitdict[trace]
                res = dic["crit_pow"] / res
                ax = axs[idx]
                plot_fit(fig, ax, trace, dic["volt"], dic["curr"], dic["crit_pow"], dic["rj0"], dic["r_series"])

            ax = axs[2]
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Resistance ($\Omega$)")
            for trace in traces:
                full_df = get_trace(trace + ".txt")
                ax.plot(full_df["voltage"], full_df["resistance"], linestyle="None", markersize=1, color="black")
                fit_df = fitdict[trace.strip(".txt")]["df"]
                fit_df["false_ramp"] = ~ fit_df["false_ramp"]
                plot_ramps(fit_df, fitcol=True, newfig=False)

                if zoom:
                    if "partial" in trace:
                        minres = fit_df.iloc[-1, fit_df.columns.get_loc("res")][0] - maxrj
                        ax.set_ylim([minres, ax.get_ylim()[1]])
                    if "final" in trace:
                        maxres = fitdict[trace.strip(".txt")]["res_tot_0"] + maxrj + 5
                        ax.set_ylim([ax.get_ylim()[0], maxres])

            fig.tight_layout()
            fig.suptitle("$P_c$ Fit Dev %s - Ratio %.2f" % (port, res))
            plt.subplots_adjust(top=0.92)
            # fig.suptitle("Fit to $P_c$ Model: Ratio = %.2f" % (res,))
            # plt.savefig("TRACES_WITH_FITS_%s" % (port,))
            plt.gcf().savefig("Trace_Fit_Combo_%s" % (port,))
            plt.close("all")


def plot_trace_fit_combo_finalem(path):
    with open("prepped_traces_dict.pickle", "rb") as myfile:
        prepdict = pickle.load(myfile)

    with open("fit_results_df_FINALEM.pickle", "rb") as myfile:
        fitdf = pickle.load(myfile)

    with open("fit_results_dict_FINALEM.pickle", "rb") as myfile:
        fitdict = pickle.load(myfile)

    # Get unique feedthrough/port identifiers corresponding to unique devices
    fnames = snp.txtfilenames(path)
    ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    ports = list(set(ports))

    # For each unique device identifier, do the ramp analysis for both traces
    for port in ports:
        fig = plt.figure(1, figsize=[12, 6])  # create one of the figures that must appear with the chart
        axs = [plt.subplot2grid((1, 2), (0, 0)), plt.subplot2grid((1, 2), (0, 1))]
        # Get the partial and final traces for this device, sort so partial is first
        trace = [fname.strip(".txt") for fname in fnames if port in fname and "final" in fname][0]

        ax = axs[1]
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Resistance ($\Omega$)")
        full_df = get_trace(trace + ".txt")
        ax.plot(full_df["voltage"], full_df["resistance"], linestyle="None", markersize=1, color="black")
        fit_df = prepdict[trace]["df"]
        fit_df["false_ramp"] = ~ fit_df["false_ramp"]
        plot_ramps(fit_df, fitcol=True, newfig=False)

        dic = fitdict[trace]
        ax = axs[0]
        plot_fit(fig, ax, trace, dic["volt"], dic["curr"], dic["crit_pow"], dic["rj0"], dic["r_series"])

        fig.tight_layout()
        fig.suptitle("$P_c$ Fit Dev %s" % (port,))
        plt.subplots_adjust(top=0.90)
        # fig.suptitle("Fit to $P_c$ Model: Ratio = %.2f" % (res,))
        # plt.savefig("TRACES_WITH_FITS_%s" % (port,))
        plt.gcf().savefig("Trace_Fit_Combo_FINALEM_%s" % (port,))
        plt.close("all")



######################################################
# PREPARING DATA FOR FITTING  #####
######################################################
def prep_all(path, overwrite=True):
    """For all unique devices in path, let the user choose which ramps to fit for both the partial and final EM traces
    of that device and set res_tot_0 from the first chosen ramp of the partial trace. Save the chosen ramps as DFs and
    pickle them with res_tot_0 as a dict."""

    # Get unique feedthrough/port identifiers corresponding to unique devices
    fnames = snp.txtfilenames(path)
    ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    ports = list(set(ports))

    if not overwrite:
        with open("prepped_traces_dict.pickle", "rb") as myfile:
            dic = pickle.load(myfile)

    # For each unique device identifier, do the ramp analysis for both traces
    for port in ports:
        if overwrite:
            print("--------------------------------------------------------")
            print("--------------------------------------------------------")
            print("Prepare Traces for Device %s" % (port,))
            print("--------------------------------------------------------")
            print("--------------------------------------------------------")
            prep_device(port, path)
        else:
            bools = [port in key for key in dic.keys()]
            if not any(bools):
                print("--------------------------------------------------------")
                print("--------------------------------------------------------")
                print("Prepare Traces for Device %s" % (port,))
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
        df.iloc[-1, df.columns.get_loc("false_ramp")] = True  # discard the last "fake" ramp
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
            df.loc[int(trac), "false_ramp"] = ~ df.loc[int(trac), "false_ramp"]
        fliptraces = snp.whichtrace(ax)  # User manually clicks points on graph

    plt.close(fig)
    return df



######################################################
# FITTING EM TRACES #####
######################################################
def fit_all(path, ratios, group, env, chip, deltarj):
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
                                       "temp", "env", "chip", "r_sqrd"], index=all_traces)
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

        # Do the Fit and save results
        lsq_fit, r_sqrd = do_fit(volt, curr)
        fit_dict[trace] = {"df": df, "r_series": lsq_fit.x[1], "crit_pow": lsq_fit.x[0], "r_sqrd": r_sqrd,
                           "rj0": res_tot_0 - lsq_fit.x[1], "res_tot_0": res_tot_0, "group": group,
                           "trace_type": "partial", "volt": volt, "curr": curr, "temp": temp, "env": env, "chip": chip}
        fit_df.loc[trace] = [lsq_fit.x[0], lsq_fit.x[1], res_tot_0 - lsq_fit.x[1],
                             res_tot_0, group, "partial", temp, env, chip, r_sqrd]

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

        # Use scaled r_series to fit and save results.
        lsq_fit, r_sqrd = do_fit(volt, curr, r_series)
        fit_dict[trace] = {"df": df, "r_series": r_series, "crit_pow": lsq_fit.x[0], "r_sqrd": r_sqrd,
                           "rj0": rj0, "res_tot_0": res_tot_0, "group": group,
                           "trace_type": "final", "volt": volt, "curr": curr, "temp": temp, "env": env, "chip": chip}
        fit_df.loc[trace] = [lsq_fit.x[0], r_series, rj0,
                             res_tot_0, group, "final", temp, env, chip, r_sqrd]

    # If the "conforms" column already existed then we set it to the pre-existing values
    if "conforms" in locals():
        fit_df["pc_conforming"] = conforms
    else:
        fit_df["pc_conforming"] = 1

    with open(dfpickle, "wb") as myfile:
        pickle.dump(fit_df, myfile)
    with open(dictpickle, "wb") as myfile:
        pickle.dump(fit_dict, myfile)


def fit_all_finalem(path, group, env, chip):
    """Fit constant power model to all prepared dataframes in path, and store results in new dict and df."""
    # Read in the dictionary of prepped traces (dfs and res_tot_0's) and get list of traces
    with open("prepped_traces_dict.pickle", "rb") as myfile:
        prepped_traces_dict = pickle.load(myfile)
    all_traces = list(prepped_traces_dict.keys())

    # Restrict to just the partial EM traces
    traces = [trace for trace in all_traces if "final" in trace]

    # Create (or load if exists) dict and df that will store fit results on all traces
    dfpickle = "fit_results_df_FINALEM.pickle"
    dictpickle = "fit_results_dict_FINALEM.pickle"
    if os.path.isfile(dfpickle):
        with open(dfpickle, "rb") as myfile:
            fit_df = pickle.load(myfile)
        conforms = fit_df["pc_conforming"].values  # Save the conforming designation, will write it back in later
        fit_df = fit_df.drop("pc_conforming", axis=1)
    else:
        fit_df = pd.DataFrame(columns=["crit_pow", "r_series", "rj0", "res_tot_0", "group", "trace_type",
                                       "temp", "env", "chip", "r_sqrd"], index=traces)
    if os.path.isfile(dictpickle):
        with open(dictpickle, "rb") as myfile:
            fit_dict = pickle.load(myfile)
    else:
        fit_dict = {}

    ########### FIT FINAL EM TRACES ##################

    # Do fitting on each trace and store results + demographics in the dict and df
    for trace in traces:
        # Restrict to only the ramps within the desired resistance range
        df = prepped_traces_dict[trace]["df"]
        temp = prepped_traces_dict[trace]["temp"]
        volt = df["last_v"].values
        curr = df["last_i"].values
        res_tot_0 = df.iloc[0, df.columns.get_loc("first_r")]

        # Do the Fit and save results
        lsq_fit, r_sqrd = do_fit(volt, curr)
        fit_dict[trace] = {"df": df, "r_series": lsq_fit.x[1], "crit_pow": lsq_fit.x[0], "r_sqrd": r_sqrd,
                           "rj0": res_tot_0 - lsq_fit.x[1], "res_tot_0": res_tot_0, "group": group,
                           "trace_type": "final_full", "volt": volt, "curr": curr, "temp": temp, "env": env, "chip": chip}
        fit_df.loc[trace] = [lsq_fit.x[0], lsq_fit.x[1], res_tot_0 - lsq_fit.x[1],
                             res_tot_0, group, "final_full", temp, env, chip, r_sqrd]

    # If the "conforms" column already existed then we set it to the pre-existing values
    if "conforms" in locals():
        fit_df["pc_conforming"] = conforms
    else:
        fit_df["pc_conforming"] = 1

    with open(dfpickle, "wb") as myfile:
        pickle.dump(fit_df, myfile)
    with open(dictpickle, "wb") as myfile:
        pickle.dump(fit_dict, myfile)


def do_fit(volt, curr, r_series=None):
    """Do constant power fit to the voltage / current ramp endpoints from a single prepared DF."""
    # If no r_series is passed, fit where p_c and r_series are both free parameters
    if not r_series:
        # Define a function to compute the residuals between the real data and the predicted current
        def residuals(theta, volt, curr):
            p_c = theta[0]
            r_series = theta[1]
            predicted_volt = predict_volt(p_c, r_series, curr)
            # Note, without .astype() the returned array would actually have data type = "object" and would store pointers
            # to the float values rather than storing the values themselves. This would throw an error in the LSQ.
            return (volt - predicted_volt).astype(np.float64)

        # Fit the model using LSQ loss (alternately, specify loss to get more outlier-robust fitting)
        theta0 = [1e-4, 10]  # first param is critical power, second param is initial Rj
        lsq = least_squares(residuals, theta0, args=(volt, curr), loss="soft_l1")
        r_sqrd = r2_score(volt, predict_volt(lsq.x[0], lsq.x[1], curr))

    # If a value for r_series is passed, then only treat p_c as free
    if r_series:
        # Define a function to compute the residuals between the real data and the predicted current
        def residuals(theta, r_series, volt, curr):
            p_c = theta[0]
            predicted_volt = predict_volt(p_c, r_series, curr)
            # Note, without .astype() the returned array would actually have data type = "object" and would store pointers
            # to the float values rather than storing the values themselves. This would throw an error in the LSQ.
            return (volt - predicted_volt).astype(np.float64)

        # Fit the model using LSQ loss (alternately, specify loss to get more outlier-robust fitting)
        theta0 = [1e-4]  # first param is critical power, second param is initial Rj
        lsq = least_squares(residuals, theta0, args=(r_series, volt, curr), loss="soft_l1")
        r_sqrd = r2_score(volt, predict_volt(lsq.x[0], r_series, curr))

    return lsq, r_sqrd


def predict_volt(p_c, r_series, curr):
    """Compute a predicted current given input resistance and model parameters."""
    predicted_volt = p_c / curr + curr * r_series
    return predicted_volt


def plot_fit(fig, ax, trace, volt, curr, p_c, rj0, r_series):
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
    # ax_inset.plot(predict_volt(p_c, r_series, curr) - volt, marker="x", markersize=8, linestyle="", color="red")
    # ax_inset.axhline(0, color="black")
    # ax_inset.get_xaxis().set_visible(False)
    # ax_inset.get_yaxis().set_visible(False)

    # Textbox with more info
    ax.text(0.02, 0.2, "$R_0^{tot}$ = %.0f $\Omega$" % (rj0 + r_series,),
            transform=ax.transAxes, fontsize=14, verticalalignment="top")
    return ax


def set_pc_conform(dfpickle, names=[], values=[]):
    with open(dfpickle, "rb") as myfile:
        fit_df = pickle.load(myfile)

    for name, value in zip(names, values):
        if not any(fit_df.index.str.contains(name)):
            print("That trace is not in the dataframe.")
        else:
            fit_df.loc[fit_df.index.str.contains(name), "pc_conforming"] = value

        # if name in fit_df.index:
        #     fit_df.loc[name, "pc_conforming"] = value
        # else:
        #     print("That trace is not in the dataframe.")

    with open(dfpickle, "wb") as myfile:
        pickle.dump(fit_df, myfile)



######################################################
# FITTING EM TRACES WITH CURRENT-DEPENDENT-SERIES RESISTANCE #####
######################################################
def fit_all_variableseries(path, ratios, group, env, chip, deltarj):
    """Fit constant power model to all prepared dataframes in path, and store results in new dict and df."""
    # Read in the dictionary of prepped traces (dfs and res_tot_0's) and get list of traces
    with open("prepped_traces_dict.pickle", "rb") as myfile:
        prepped_traces_dict = pickle.load(myfile)
    all_traces = list(prepped_traces_dict.keys())

    # Create (or load if exists) dict and df that will store fit results on all traces
    dfpickle = "fit_results_df_deltarj_%i_varseries.pickle" % (deltarj,)
    dictpickle = "fit_results_dict_deltarj_%i_varseries.pickle" % (deltarj,)
    if os.path.isfile(dfpickle):
        with open(dfpickle, "rb") as myfile:
            fit_df = pickle.load(myfile)
        conforms = fit_df["pc_conforming"].values  # Save the conforming designation, will write it back in later
        fit_df = fit_df.drop("pc_conforming", axis=1)
    else:
        fit_df = pd.DataFrame(columns=["crit_pow", "r_series", "rj0", "res_tot_0", "group", "trace_type",
                                       "temp", "env", "chip", "r_sqrd"], index=all_traces)
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
        lsq_fit, r_sqrd = do_fit_variableseries(volt, curr, alpha, curr_zero)
        fit_dict[trace] = {"df": df, "r_series": lsq_fit.x[1], "crit_pow": lsq_fit.x[0],
                           "rj0": res_tot_0 - lsq_fit.x[1], "res_tot_0": res_tot_0, "group": group,
                           "trace_type": "partial", "volt": volt, "curr": curr, "temp": temp, "env": env,
                           "chip": chip, "r_sqrd": r_sqrd}
        fit_df.loc[trace] = [lsq_fit.x[0], lsq_fit.x[1], res_tot_0 - lsq_fit.x[1],
                             res_tot_0, group, "partial", temp, env, chip, r_sqrd]

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
        lsq_fit, r_sqrd = do_fit_variableseries(volt, curr, alpha, curr_zero, r_series)
        # lsq_fit = do_fit_variableseries(volt, curr, alpha)

        fit_dict[trace] = {"df": df, "r_series": r_series, "crit_pow": lsq_fit.x[0],
                           "rj0": rj0, "res_tot_0": res_tot_0, "group": group,
                           "trace_type": "final", "volt": volt, "curr": curr, "temp": temp, "env": env,
                           "chip": chip, "r_sqrd": r_sqrd}
        fit_df.loc[trace] = [lsq_fit.x[0], r_series, rj0,
                             res_tot_0, group, "final", temp, env, chip, r_sqrd]

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
        r_sqrd = r2_score(volt, predict_volt_variableseries(lsq.x[0], eta, eta_curr, lsq.x[1], curr))

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
        r_sqrd = r2_score(volt, predict_volt_variableseries(lsq.x[0], eta, eta_curr, r_series, curr))

    return lsq, r_sqrd


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
# ASSESS TEMPERATURE-INDUCED RESISTANCE CHANGES #####
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
        print(port)
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
            # print("trying")
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

            if ramp >= len(df_final):
                loopflag = 0
                iff = df_final.iloc[0, 1]
                rf = df_final.iloc[0, 2]
                res_ratios = rf[-25:] / rp

        ratios[port] = res_ratios.mean()
        # ratios[port] = 1

    # Print the results
    for item in ratios.items():
        print(item)
    print("RESISTANCE Mean Ratio is %.2f +/- %.2f" % (np.mean(list(ratios.values())), np.std(list(ratios.values()))))
    return ratios


def resistance_ratios_from_em2(path):
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
        print(port)
        # Order the partial and final traces so we know which direction to take the ratio
        traces = [os.path.join(path, fname) for fname in fnames if port in fname]
        traces = [tr for tr in traces if "partial" in tr] + [tr for tr in traces if "final" in tr]

        # Read in DFs for partial and final EM, pull the last ramp cycle of partial EM (a "semi-ramp")
        df_partial = extract_ramps(get_trace(traces[0]))
        df_final = extract_ramps(get_trace(traces[1]))
        # Get current and resistance on the last semi-ramp of the partial EM trace
        ip = df_partial.iloc[-1:, 1].values[0]
        rp = df_partial.iloc[-1:, 2].values[0]

        iff = df_final.iloc[0, 1]  # Get the current on the first ramp of final EM
        rf = df_final.iloc[0, 2]  # Get the resistance on the first ramp of final EM
        # discard the last 150 mVs of the first ramp of final EM
        iff, rf = iff[:-150], rf[:-150]
        # capture the next 50 mVs to compare with the last ramp of partial EM
        iff, rf = iff[-50:], rf[-50:]

        start_idx = np.argmin(np.abs(ip - iff[0]))
        res_ratios = rf / rp[start_idx:start_idx+50]
        #
        # # Look through ramp cycles of final EM, starting with first, to find a match to the semi-ramp voltage
        # ramp = 0
        # loopflag = 1
        # while loopflag:
        #     # print("trying")
        #     iff = df_final.iloc[ramp, 1]  # Get the current on the first ramp of final EM
        #     rf = df_final.iloc[ramp, 2]  # Get the resistance on the first ramp of final EM
        #     start_idx = np.argmin(np.abs(iff - ip[0]))
        #     # If the best "match" is not within 1 mV then we need to look in the next ramp cycle
        #     if np.abs((iff - ip[0])[start_idx]) <= 0.00005:
        #         loopflag = 0
        #         nsteps = min(len(iff) - start_idx, 24)
        #         stop_idx = start_idx + nsteps
        #         # Note, in case the first matching ramp from final EM stops in the middle of the last ramp of partial
        #         # EM, we need to potentially truncate ip
        #         res_ratios = rf[start_idx:stop_idx] / rp[:nsteps]
        #     ramp += 1
        #
        #     if ramp >= len(df_final):
        #         loopflag = 0
        #         iff = df_final.iloc[0, 1]
        #         rf = df_final.iloc[0, 2]
        #         res_ratios = rf[-25:] / rp

        ratios[port] = res_ratios.mean()
        # ratios[port] = 1

    # Print the results
    for item in ratios.items():
        print(item)
    print("RESISTANCE Mean Ratio is %.2f +/- %.2f" % (np.mean(list(ratios.values())), np.std(list(ratios.values()))))
    return ratios


######################################################
# FORMAT, AGGREGATE AND VISUALIZE RESULTS #####
######################################################
def analyze_pc(path, deltarj, rsqrd_cutoffs, varseries=False):
    """Make a DF where each row is a device and it stores pc for T1 and T2."""
    if not varseries:
        with open("fit_results_df_deltarj_%i.pickle" %(deltarj,), "rb") as myfile:
            fitdf = pickle.load(myfile)
        with open("fit_results_dict_deltarj_%i.pickle" %(deltarj,), "rb") as myfile:
            fitdict = pickle.load(myfile)
    else:
        with open("fit_results_df_deltarj_%i_varseries.pickle" %(deltarj,), "rb") as myfile:
            fitdf = pickle.load(myfile)
        with open("fit_results_dict_deltarj_%i_varseries.pickle" %(deltarj,), "rb") as myfile:
            fitdict = pickle.load(myfile)

    with open("fit_results_df_FINALEM.pickle", "rb") as myfile:
        finaldf = pickle.load(myfile)
        newidx = ["_".join(idx.split("_")[-2:]) for idx in finaldf.index]
        finaldf.index = newidx

    # Get unique feedthrough/port identifiers corresponding to unique devices
    fnames = snp.txtfilenames(path)
    ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    ports = list(set(ports))
    df = pd.DataFrame(columns=["T1", "T2", "T_ratio", "pc1", "pc2", "p_ratio", "r_series_T1", "rj0_T1",
                               "ratio_RT1_RT2", "group", "env", "chip", "rsqrd_T1", "rsqrd_T2",
                               "pc_finalem", "rsqrd_finalem"], index=ports)
    # For each unique device identifier, do the ramp analysis for both traces
    for port in ports:
        traces = [fname.strip(".txt") for fname in fnames if port in fname]
        partial = fitdict[[trace for trace in traces if "partial" in trace][0]]
        final = fitdict[[trace for trace in traces if "final" in trace][0]]
        pratio = final["crit_pow"] / partial["crit_pow"]
        tratio = final["temp"] / partial["temp"]
        df.loc[port, ["T1", "pc1", "r_series_T1", "rj0_T1", "group", "env", "chip", "rsqrd_T1"]] = [partial["temp"],
                        partial["crit_pow"], partial["r_series"], partial["rj0"], partial["group"],
                        partial["env"], partial["chip"], partial["r_sqrd"]]
        df.loc[port, ["T2", "pc2", "T_ratio", "p_ratio", "rsqrd_T2"]] = [final["temp"], final["crit_pow"], tratio,
                                                                         pratio, final["r_sqrd"]]
        df.loc[port, "ratio_RT1_RT2"] = ratios[port]

        df.loc[port, ["pc_finalem", "rsqrd_finalem"]] = [finaldf.loc[port, "crit_pow"], finaldf.loc[port, "r_sqrd"]]

    with open("pc_analysis_deltarj_%i.pickle" % (deltarj,), "wb") as myfile:
        pickle.dump(df, myfile)

    df1 = df[(df["rsqrd_T1"] > rsqrd_cutoffs[0]) & (df["rsqrd_T2"] > rsqrd_cutoffs[1])]
    print("CRITICAL POWER RATIOS (R^2(T1) > %.3f, R^2(T2) > %.3f):" % (rsqrd_cutoffs[0], rsqrd_cutoffs[1]))
    print(df1["p_ratio"])
    print("Mean of Critical Power Ratios is %.2f +/- %.2f" % (df1["p_ratio"].mean(), df1["p_ratio"].std()))

    df2 = df[(df["rsqrd_finalem"] > 0.995)]
    print("ABSOLUTE CRITICAL POWERS (R^2 > .995):")
    print(df2["pc_finalem"])
    print("Mean of Absolute Critical Powers is %.2f +/- %.2f mW" % ((1000*df2["pc_finalem"]).mean(), (1000*df2["pc_finalem"]).std()))


def aggregate_pc(paths, deltarj, rsqrd_cutoffs, mkplts=False):
    rsqrd_cutoff_T1, rsqrd_cutoff_T2 = rsqrd_cutoffs
    df = pd.DataFrame()
    for path in paths:
        dfpath = path + "/pc_analysis_deltarj_%i.pickle" % (deltarj,)
        with open(dfpath, "rb") as myfile:
            temp = pickle.load(myfile)
        df = pd.concat([df, temp], axis=0)

    df1 = df[(df["rsqrd_T1"] > rsqrd_cutoff_T1) & (df["rsqrd_T2"] > rsqrd_cutoff_T2)]

    if mkplts:
        grps = []
        means = []
        stddev = []
        for nm, grp in df1.groupby("group"):
            grps.append(nm)
            means.append(grp["p_ratio"].mean())
            stddev.append(grp["p_ratio"].std())
        fig, ax = snp.newfig("ur")
        snp.labs("Geometry", "$P_c$ Ratio", "$P_c$ Ratios (%i, %.3f, %.3f)" % (deltarj, rsqrd_cutoff_T1, rsqrd_cutoff_T2))
        snp.catplot(grps, means)
        plt.errorbar(np.arange(len(grps)), means, yerr=stddev, marker="", markersize=10, color="black", linestyle="")
        # plt.errorbar([0, 1, 2, 3, 4], means, yerr=stddev, marker="", markersize=10, color="black", linestyle="")
        # ax.set_ylim([0.85, 1.1])
        plt.savefig("Pc_Ratios_Aggregated.png")



        df2 = df[(df["rsqrd_finalem"] > 0.99)]
        grps = []
        means = []
        stddev = []
        for nm, grp in df2.groupby("group"):
            grps.append(nm)
            means.append((1000*grp["pc_finalem"]).mean())
            stddev.append((1000*grp["pc_finalem"]).std())
        fig, ax = snp.newfig("ur")
        snp.labs("Geometry", "Fitted Critical Power (mW)", "Critical Power at 96K")
        snp.catplot(grps, means)
        plt.errorbar(np.arange(len(grps)), means, yerr=stddev, marker="", markersize=10, color="black", linestyle="")
        # plt.errorbar([0, 1, 2, 3, 4], means, yerr=stddev, marker="", markersize=10, color="black", linestyle="")
        plt.savefig("Pc_FinalEM_Aggregated.png")

        grps = []
        means = []
        stddev = []
        for nm, grp in df2.groupby("group"):
            grps.append(nm)
            grp["pc_finalem"] = grp["pc_finalem"] / grp["pc_finalem"].mean()
            means.append(grp["pc_finalem"].mean())
            stddev.append(grp["pc_finalem"].std())
        fig, ax = snp.newfig("ur")
        snp.labs("Geometry", "Fitted Critical Power (mW)", "Critical Power at 96K")
        snp.catplot(grps, means)
        plt.errorbar(np.arange(len(grps)), means, yerr=stddev, marker="", markersize=10, color="black", linestyle="")
        # ax.set_ylim([0.7, 1.1])
        plt.savefig("Pc_FinalEM_Scaled_Aggregated.png")

        plt.close("all")
    #
    # fig, ax = plt.subplots()
    # snp.labs("T2", "$P_c$ (mW)", "Absolute $P_c$ Varitemp EM ($R^2 > 0.995$)")
    # cm = snp.cmap(ncols = len(df2["group"].unique()))
    # for grp in df2["group"].unique():
    #     ax.scatter(df2.loc[df2["group"] == grp, "T2"], df2.loc[df2["group"] == grp, "pc_finalem"], label=grp, c=next(cm))
    # ax.legend(loc="upper center")

    return df, df1


def aggregate_tc(paths, deltarj, rsqrd_cutoffs, mkplts=False):
    rsqrd_cutoff_T1, rsqrd_cutoff_T2 = rsqrd_cutoffs
    df = pd.DataFrame()
    for path in paths:
        dfpath = path + "/pc_analysis_deltarj_%i.pickle" % (deltarj,)
        with open(dfpath, "rb") as myfile:
            temp = pickle.load(myfile)
        df = pd.concat([df, temp], axis=0)

    df1 = df[(df["rsqrd_T1"] > rsqrd_cutoff_T1) & (df["rsqrd_T2"] > rsqrd_cutoff_T2)]

    if mkplts:
        means = []
        stddev = []
        for nm, grp in df1.groupby("group"):
            fig, ax = snp.newfig()
            snp.labs("Ratio of Temperatures", "Ratio of Critical Powers", "Pc Ratio vs Temp Ratio - %s" % (nm,))
            ax.scatter(grp["T_ratio"], grp["p_ratio"], label=nm)
            plt.savefig("Pc_Ratios_%s.png" % (nm,))

        for nm, grp in df1.groupby("group"):
            fig, ax = snp.newfig()
            snp.labs("Temperatures", "Critical Power (mW)", "Pc vs. Temp - %s" % (nm,))
            ax.scatter(grp["T2"], 1000*grp["pc_finalem"], label=nm)
            plt.savefig("Pc_Absolutes_%s.png" % (nm,))

        #     grps.append(nm)
        #     means.append(grp["p_ratio"].mean())
        #     stddev.append(grp["p_ratio"].std())
        # fig, ax = snp.newfig("ur")
        # snp.labs("Geometry", "$P_c$ Ratio", "$P_c$ Ratios (%i, %.3f, %.3f)" % (deltarj, rsqrd_cutoff_T1, rsqrd_cutoff_T2))
        # snp.catplot(grps, means)
        # plt.errorbar(np.arange(len(grps)), means, yerr=stddev, marker="", markersize=10, color="black", linestyle="")
        # # plt.errorbar([0, 1, 2, 3, 4], means, yerr=stddev, marker="", markersize=10, color="black", linestyle="")
        # # ax.set_ylim([0.85, 1.1])
        # plt.savefig("Pc_Ratios_Aggregated.png")
        #
        #
        #
        # df2 = df[(df["rsqrd_finalem"] > 0.99)]
        # grps = []
        # means = []
        # stddev = []
        # for nm, grp in df2.groupby("group"):
        #     grps.append(nm)
        #     means.append((1000*grp["pc_finalem"]).mean())
        #     stddev.append((1000*grp["pc_finalem"]).std())
        # fig, ax = snp.newfig("ur")
        # snp.labs("Geometry", "Fitted Critical Power (mW)", "Critical Power at 96K")
        # snp.catplot(grps, means)
        # plt.errorbar(np.arange(len(grps)), means, yerr=stddev, marker="", markersize=10, color="black", linestyle="")
        # # plt.errorbar([0, 1, 2, 3, 4], means, yerr=stddev, marker="", markersize=10, color="black", linestyle="")
        # plt.savefig("Pc_FinalEM_Aggregated.png")
        #
        # grps = []
        # means = []
        # stddev = []
        # for nm, grp in df2.groupby("group"):
        #     grps.append(nm)
        #     grp["pc_finalem"] = grp["pc_finalem"] / grp["pc_finalem"].mean()
        #     means.append(grp["pc_finalem"].mean())
        #     stddev.append(grp["pc_finalem"].std())
        # fig, ax = snp.newfig("ur")
        # snp.labs("Geometry", "Fitted Critical Power (mW)", "Critical Power at 96K")
        # snp.catplot(grps, means)
        # plt.errorbar(np.arange(len(grps)), means, yerr=stddev, marker="", markersize=10, color="black", linestyle="")
        # # ax.set_ylim([0.7, 1.1])
        # plt.savefig("Pc_FinalEM_Scaled_Aggregated.png")

        # plt.close("all")
    #
    # fig, ax = plt.subplots()
    # snp.labs("T2", "$P_c$ (mW)", "Absolute $P_c$ Varitemp EM ($R^2 > 0.995$)")
    # cm = snp.cmap(ncols = len(df2["group"].unique()))
    # for grp in df2["group"].unique():
    #     ax.scatter(df2.loc[df2["group"] == grp, "T2"], df2.loc[df2["group"] == grp, "pc_finalem"], label=grp, c=next(cm))
    # ax.legend(loc="upper center")

    return 0


def optimize_pc(path, deltarj, rsqrd_cutoffs, varseries=False):
    """Make a DF where each row is a device and it stores pc for T1 and T2."""
    if not varseries:
        with open("fit_results_df_deltarj_%i.pickle" %(deltarj,), "rb") as myfile:
            fitdf = pickle.load(myfile)
        with open("fit_results_dict_deltarj_%i.pickle" %(deltarj,), "rb") as myfile:
            fitdict = pickle.load(myfile)
    else:
        with open("fit_results_df_deltarj_%i_varseries.pickle" %(deltarj,), "rb") as myfile:
            fitdf = pickle.load(myfile)
        with open("fit_results_dict_deltarj_%i_varseries.pickle" %(deltarj,), "rb") as myfile:
            fitdict = pickle.load(myfile)

    # Get unique feedthrough/port identifiers corresponding to unique devices
    fnames = snp.txtfilenames(path)
    ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    ports = list(set(ports))
    df = pd.DataFrame(columns=["T1", "T2", "T_ratio", "pc1", "pc2", "p_ratio", "r_series_T1", "rj0_T1",
                               "ratio_RT1_RT2", "group", "env", "chip", "rsqrd_T1", "rsqrd_T2",
                               "pc_finalem", "rsqrd_finalem"], index=ports)
    # For each unique device identifier, do the ramp analysis for both traces
    for port in ports:
        traces = [fname.strip(".txt") for fname in fnames if port in fname]
        partial = fitdict[[trace for trace in traces if "partial" in trace][0]]
        final = fitdict[[trace for trace in traces if "final" in trace][0]]
        pratio = final["crit_pow"] / partial["crit_pow"]
        tratio = final["temp"] / partial["temp"]
        df.loc[port, ["T1", "pc1", "r_series_T1", "rj0_T1", "group", "env", "chip", "rsqrd_T1"]] = [partial["temp"],
                        partial["crit_pow"], partial["r_series"], partial["rj0"], partial["group"],
                        partial["env"], partial["chip"], partial["r_sqrd"]]
        df.loc[port, ["T2", "pc2", "T_ratio", "p_ratio", "rsqrd_T2"]] = [final["temp"], final["crit_pow"], tratio,
                                                                         pratio, final["r_sqrd"]]
        df.loc[port, "ratio_RT1_RT2"] = ratios[port]

        df.loc[port, ["pc_finalem", "rsqrd_finalem"]] = np.nan

    with open("pc_analysis_deltarj_%i.pickle" % (deltarj,), "wb") as myfile:
        pickle.dump(df, myfile)

    df1 = df[(df["rsqrd_T1"] > rsqrd_cutoffs[0]) & (df["rsqrd_T2"] > rsqrd_cutoffs[1])]
    return df1["p_ratio"]


def optimize_pc_exclude(path, deltarj, rsqrd_cutoffs,  exclude=[], varseries=False):
    """Make a DF where each row is a device and it stores pc for T1 and T2."""
    if not varseries:
        with open("fit_results_df_deltarj_%i.pickle" %(deltarj,), "rb") as myfile:
            fitdf = pickle.load(myfile)
        with open("fit_results_dict_deltarj_%i.pickle" %(deltarj,), "rb") as myfile:
            fitdict = pickle.load(myfile)
    else:
        with open("fit_results_df_deltarj_%i_varseries.pickle" %(deltarj,), "rb") as myfile:
            fitdf = pickle.load(myfile)
        with open("fit_results_dict_deltarj_%i_varseries.pickle" %(deltarj,), "rb") as myfile:
            fitdict = pickle.load(myfile)

    # Get unique feedthrough/port identifiers corresponding to unique devices
    fnames = snp.txtfilenames(path)
    ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    ports = list(set(ports))
    df = pd.DataFrame(columns=["T1", "T2", "T_ratio", "pc1", "pc2", "p_ratio", "r_series_T1", "rj0_T1",
                               "ratio_RT1_RT2", "group", "env", "chip", "rsqrd_T1", "rsqrd_T2",
                               "pc_finalem", "rsqrd_finalem"], index=ports)
    # For each unique device identifier, do the ramp analysis for both traces
    for port in ports:
        traces = [fname.strip(".txt") for fname in fnames if port in fname]
        partial = fitdict[[trace for trace in traces if "partial" in trace][0]]
        final = fitdict[[trace for trace in traces if "final" in trace][0]]
        pratio = final["crit_pow"] / partial["crit_pow"]
        tratio = final["temp"] / partial["temp"]
        df.loc[port, ["T1", "pc1", "r_series_T1", "rj0_T1", "group", "env", "chip", "rsqrd_T1"]] = [partial["temp"],
                        partial["crit_pow"], partial["r_series"], partial["rj0"], partial["group"],
                        partial["env"], partial["chip"], partial["r_sqrd"]]
        df.loc[port, ["T2", "pc2", "T_ratio", "p_ratio", "rsqrd_T2"]] = [final["temp"], final["crit_pow"], tratio,
                                                                         pratio, final["r_sqrd"]]
        df.loc[port, "ratio_RT1_RT2"] = ratios[port]

        df.loc[port, ["pc_finalem", "rsqrd_finalem"]] = np.nan

    with open("pc_analysis_deltarj_%i.pickle" % (deltarj,), "wb") as myfile:
        pickle.dump(df, myfile)

    df.drop(exclude, axis=0, inplace=True)
    df1 = df[(df["rsqrd_T1"] > rsqrd_cutoffs[0]) & (df["rsqrd_T2"] > rsqrd_cutoffs[1])]
    return df1["p_ratio"].mean(), df1["p_ratio"].std()/np.sqrt(len(df1))










datafiles = [
["C:/Users/Sonya/Documents/My Box Files/MolT Project/161230_SDS20_Chip_11-10/161230_Chip_11-10_EM_95K/30x10", "11-10", "30x10"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170101_SDS20_Chip_6-9/170101_Chip_6-9_EM_95K/30x10", "6-9", "30x10"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/170114_Chip_7-11_EM_95K/30x10", "7-11", "30x10"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/170114_Chip_7-11_EM_95K/50x400", "7-11", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/170114_Chip_7-11_EM_125K/30x10", "7-11", "30x10"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/170114_Chip_7-11_EM_125K/50x400", "7-11", "50x400"]
]

datafiles = [
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/30x60", "10-9", "30x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/30x400", "10-9", "30x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/50x60", "10-9", "50x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/50x400", "10-9", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/60x400", "10-9", "60x400"]
]


datafiles = [
["C:/Users/Sonya/Documents/My Box Files/molT Project/170124_SDS20_Chip_4-11/50x60", "4-11", "50x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170124_SDS20_Chip_4-11/30x60", "4-11", "30x60"]
]

datafiles = [
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/50x60", "12-9", "50x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/50x400", "12-9", "50x400"]
]


datafiles = [
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/30x60", "12-9", "30x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/50x60", "12-9", "50x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/50x400", "12-9", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/60x400", "12-9", "60x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/30x400_newalgo", "12-9", "30x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/30x400_oldalgo", "12-9", "30x400"]
]


datafiles = [
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/50x60", "12-8", "50x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/50x400", "12-8", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/30x400", "12-8", "30x400"]
]

datafiles = [
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/50x400", "12-8", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/30x400", "12-8", "30x400"]
]

datafiles = [
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/50x60", "12-8", "50x60"]
]

#
# datafiles = [
# ["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/30x60", "12-9", "30x60"],
# ["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/60x400", "12-9", "60x400"]]
#
# datafiles = [["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/30x400", "12-9", "30x400"]]
datafiles = [
["C:/Users/Sonya/Documents/My Box Files/molT Project/170202_SDS20_Chip_11-9/30x400", "11-9", "30x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170202_SDS20_Chip_11-9/50x60", "11-9", "30x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170202_SDS20_Chip_11-9/50x400", "11-9", "60x400"]
]



for dataf in datafiles:
    path, chip, group = dataf
    os.chdir(path)
    print("-----------------------------------------------------")
    print("Analysis for Chip %s" % (chip,))
    print("-----------------------------------------------------")

    # Set values for all relevant variables
    env = "He_gas"
    prep_all(path, overwrite=False)  # Make this large enough to catch all false ramps that might possibly be used
    # pic_all(path)

deltarj = 15
rsqrd_cutoffs = [0.99, 0.99]
for dataf in datafiles:
    path, chip, group = dataf
    os.chdir(path)

    # Set values for all relevant variables
    env = "He_gas"
    if chip in ["11-10", "6-9"]:
        fnames = snp.txtfilenames(path)
        ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
        ports = list(set(ports))
        ratios = {port: 0.999 for port in ports}
    elif chip in ["4-11", "10-9", "7-11", "7-9", "9-11", "7-10"]:
        ratios = resistance_ratios_from_em(path)
    elif chip == "12-9" and (group == "50x60" or group == "50x400" or "oldalgo" in path):
        ratios = resistance_ratios_from_em(path)
    else:
        ratios = resistance_ratios_from_em2(path)

    # fnames = snp.txtfilenames(path)
    # ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    # ports = list(set(ports))
    # ratios = {port: 0.999 for port in ports}

    fit_all(path, ratios, group=group, env=env, chip=chip, deltarj=deltarj)
    fit_all_finalem(path, group=group, env=env, chip=chip)
    # plot_trace_fit_combo(path, zoom=True, maxrj=deltarj + 15, deltarj=deltarj)
    # plot_trace_fit_combo_finalem(path)
    analyze_pc(path, deltarj, rsqrd_cutoffs=rsqrd_cutoffs)

# Aggregate Results
os.chdir("C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8")
df_full, df = aggregate_pc([df[0] for df in datafiles], deltarj=deltarj, rsqrd_cutoffs=rsqrd_cutoffs, mkplts=True)
df1 = df[["p_ratio", "group", "rsqrd_T1", "rsqrd_T2"]]
df1
df_full1 = df_full[["p_ratio", "group", "rsqrd_T1", "rsqrd_T2"]]
df_full1

df_full2 = df_full[["pc_finalem", "rsqrd_finalem"]]
df_full2

deltarj = 17
rsqrd_cutoffs = [0.992, 0.992]
for dataf in datafiles:
    path, chip, group = dataf
    os.chdir(path)

    # Set values for all relevant variables
    env = "He_gas"
    if chip in ["11-10", "6-9"]:
        fnames = snp.txtfilenames(path)
        ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
        ports = list(set(ports))
        ratios = {port: 0.999 for port in ports}
    elif chip in ["4-11", "10-9", "7-11", "7-9", "9-11", "7-10"]:
        ratios = resistance_ratios_from_em(path)
    elif chip == "12-9" and (group == "50x60" or group == "50x400" or "oldalgo" in path):
        ratios = resistance_ratios_from_em(path)
    else:
        ratios = resistance_ratios_from_em2(path)

    # fnames = snp.txtfilenames(path)
    # ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    # ports = list(set(ports))
    # ratios = {port: 0.999 for port in ports}

    fit_all_variableseries(path, ratios, group=group, env=env, chip=chip, deltarj=deltarj)
    # fit_all_finalem(path, group=group, env=env, chip=chip)
    # plot_trace_fit_combo(path, zoom=True, maxrj=deltarj + 15, deltarj=deltarj)
    # plot_trace_fit_combo_finalem(path)
    analyze_pc(path, deltarj, rsqrd_cutoffs=rsqrd_cutoffs, varseries=False)

os.chdir("C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8")
trash = aggregate_tc([df[0] for df in datafiles], deltarj=deltarj, rsqrd_cutoffs=rsqrd_cutoffs, mkplts=True)

















datafiles = [
["C:/Users/Sonya/Documents/My Box Files/MolT Project/161230_SDS20_Chip_11-10/pc_ratio/30x10", "11-10", "30x10"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170101_SDS20_Chip_6-9/pc_ratio/30x10", "6-9", "30x10"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/pc_ratio_95K/30x10", "7-11", "30x10"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/pc_ratio_95K/50x400", "7-11", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/pc_ratio_125K/30x10", "7-11", "30x10"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/pc_ratio_125K/50x400", "7-11", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170115_SDS20_Chip_7-9/pc_ratio/30x10", "7-9", "30x10"],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170115_SDS20_Chip_7-9/pc_ratio/50x400", "7-9", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/30x60", "10-9", "30x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/30x400", "10-9", "30x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/50x60", "10-9", "50x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/50x400", "10-9", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/60x400", "10-9", "60x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170124_SDS20_Chip_4-11/50x60", "4-11", "50x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170124_SDS20_Chip_4-11/30x60", "4-11", "30x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/30x60", "12-9", "30x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/50x60", "12-9", "50x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/50x400", "12-9", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/60x400", "12-9", "60x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/30x400_newalgo", "12-9", "30x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/30x400_oldalgo", "12-9", "30x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/50x60", "12-8", "50x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/50x400", "12-8", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/30x400", "12-8", "30x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/30x60", "12-8", "30x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/60x400", "12-8", "60x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170202_SDS20_Chip_11-9/50x60", "11-9", "50x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170202_SDS20_Chip_11-9/50x400", "11-9", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170202_SDS20_Chip_11-9/30x400", "11-9", "30x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170202_SDS20_Chip_11-9/30x60", "11-9", "30x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170202_SDS20_Chip_11-9/60x400", "11-9", "60x400"]
]

#####################################################################
####### EFFECT OF DELTARJ AND R^2 FOR ALL PREPPED TRACES ##########
#####################################################################
dataf = datafiles[24]
deltarjs = np.arange(10, 18, 1)
rsqrd_cutoffs = np.linspace(0.992, 0.998, 7)
params = [(x, y) for x in deltarjs for y in rsqrd_cutoffs]

path, chip, group = dataf
print(path)
os.chdir(path)
env = "He_gas"
for f in os.listdir(path):
    if "fit_results" in f or "pc_analysis" in f or "Trace_Fit_Combo" in f:
        os.remove(f)

if chip in ["11-10", "6-9"]:
    fnames = snp.txtfilenames(path)
    ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    ports = list(set(ports))
    ratios = {port: 0.999 for port in ports}
elif chip in ["4-11", "10-9", "7-11", "7-9", "9-11", "7-10"]:
    ratios = resistance_ratios_from_em(path)
elif chip == "12-9" and (group == "50x60" or group == "50x400" or "oldalgo" in path):
    ratios = resistance_ratios_from_em(path)
else:
    ratios = resistance_ratios_from_em2(path)

settings = []
results = pd.Series()
for idx, param in enumerate(params):
    deltarj, rsqrd_cutoff = param
    fit_all(path, ratios, group=group, env=env, chip=chip, deltarj=deltarj)
    if idx == 0:
        plot_trace_fit_combo(path, zoom=True, maxrj=deltarj + 15, deltarj=deltarj)
    ps = optimize_pc(path, deltarj, rsqrd_cutoffs=[rsqrd_cutoff, rsqrd_cutoff])
    settings += [idx]*len(ps)
    results = pd.concat([results, ps])
plt.close("all")
fig, ax = snp.newfig()
snp.labs("deltarj and $R^2$ Cutoff Setting", "$P_c$ for Points above $R^2$ Cutoff",
         "Investigate Parameter Effect - %s,%s" % (chip, group))
# Plot a vertical scatter plot of Pc of included devices at each setting, color code by device ID
df = pd.DataFrame({"pc_ratio": results, "ports": results.index, "params": settings})
cmap = snp.cmap(ncols=len(df["ports"].unique()))
for nm, grp in df.groupby("ports"):
    ax.plot(grp["params"].values, grp["pc_ratio"].values, color=next(cmap), markersize=5, label=nm, linestyle="None")
ax.legend(loc="upper right")
plt.savefig("OptimalParams_Scatter.png")


#####################################################################
#######  REVISE PREP FOR ANY DEVICES THAT MIGHT BENEFIT   ##########
#####################################################################
prep_device("FT1_1618", path, deltarj=20)
prep_device("FT1_0403", path, deltarj=20)
prep_device("FT1_2321", path, deltarj=20)
prep_device("FT2_0908", path, deltarj=20)


#####################################################################
####### LIST THE OUTLIER DEVICES THAT WILL BE EXCLUDED ##########
#####################################################################
exclude = ["FT1_1618"]


#####################################################################
####### EFFECT OF DELTARJ AND R^2 FOR INCLUDED TRACES ONLY ##########
#####################################################################
fig, ax = snp.newfig()
snp.labs("deltarj and $R^2$ Cutoff Setting", "$P_c$ for Points above $R^2$ Cutoff",
         "Investigate Parameter Effect - %s,%s" % (chip, group))
mus, stds = [], []
for idx, param in enumerate(params):
    deltarj, rsqrd_cutoff = param
    mu, std = optimize_pc_exclude(path, deltarj, rsqrd_cutoffs=[rsqrd_cutoff, rsqrd_cutoff], exclude=exclude)
    mus.append(mu)
    stds.append(std)
ax.plot(np.arange(len(params)), mus, linestyle="None")
plt.errorbar(np.arange(len(params)), mus, yerr=stds, marker="", markersize=10, color="black", linestyle="")
plt.savefig("OptimalParams_Errorbar.png")


#####################################################################
####### CLEAN UP AND FIT WITH THE CHOSEN OPTIMAL SETTINGS ##########
#####################################################################
# Remove all the df fitting files and pc_analysis in this directory
for f in os.listdir(path):
    if "fit_results" in f or "pc_analysis" in f or "Trace_Fit_Combo" in f:
        os.remove(f)
deltarj, rsqrd_cutoff = params[42]
print(deltarj, rsqrd_cutoff)
fit_all(path, ratios, group=group, env=env, chip=chip, deltarj=deltarj)
plot_trace_fit_combo(path, zoom=True, maxrj=deltarj + 15, deltarj=deltarj)
dfpickle = "fit_results_df_deltarj_%i.pickle" % (deltarj,)
set_pc_conform(dfpickle, names=exclude, values=[0]*len(exclude))

with open(dfpickle, "rb") as myfile:
    fit_df = pickle.load(myfile)

#####################################################################
####### TRACK OPTIMAL PARAMETERS FOR ALL CHIPS ##########
#####################################################################
optimal_params = [
["C:/Users/Sonya/Documents/My Box Files/MolT Project/161230_SDS20_Chip_11-10/pc_ratio/30x10", "11-10", "30x10", 11, 0.995],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170101_SDS20_Chip_6-9/pc_ratio/30x10", "6-9", "30x10", 10, 0.992],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/pc_ratio_95K/30x10", "7-11", "30x10", 12, 0.998],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/pc_ratio_95K/50x400", "7-11", "50x400", 17, 0.993],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/pc_ratio_125K/30x10", "7-11", "30x10", 10, 0.992],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170114_SDS20_Chip_7-11/pc_ratio_125K/50x400", "7-11", "50x400", 11, 0.993],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170115_SDS20_Chip_7-9/pc_ratio/30x10", "7-9", "30x10", 10, 0.993],
["C:/Users/Sonya/Documents/My Box Files/MolT Project/170115_SDS20_Chip_7-9/pc_ratio/50x400", "7-9", "50x400", 14, 0.992],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/30x60", "10-9", "30x60", 12, 0.993],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/30x400", "10-9", "30x400", 13, 0.992],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/50x60", "10-9", "50x60", 10, 0.992],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/50x400", "10-9", "50x400", 13, 0.993],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/60x400", "10-9", "60x400", 10, 0.992],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170124_SDS20_Chip_4-11/50x60", "4-11", "50x60", 12, 0.993],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170124_SDS20_Chip_4-11/30x60", "4-11", "30x60", 11, 0.993],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/30x60", "12-9", "30x60", 12, 0.992],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/50x60", "12-9", "50x60"], 10, 0.993,
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/50x400", "12-9", "50x400", 12, 0.993],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/60x400", "12-9", "60x400", 10, 0.996],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/30x400_newalgo", "12-9", "30x400", 17, 0.992],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/30x400_oldalgo", "12-9", "30x400", 12, 0.994],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/50x60", "12-8", "50x60", 10, 0.994],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/50x400", "12-8", "50x400", 16, 0.998],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/30x60", "12-8", "30x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/60x400", "12-8", "60x400"]
]





##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
#####################################################################
#####################################################################
#####################################################################
datafiles = [
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/30x60", "10-9", "30x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/30x400", "10-9", "30x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/50x60", "10-9", "50x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/50x400", "10-9", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170120_SDS20_Chip_10-9/60x400", "10-9", "60x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170124_SDS20_Chip_4-11/50x60", "4-11", "50x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170124_SDS20_Chip_4-11/30x60", "4-11", "30x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/30x60", "12-9", "30x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/50x60", "12-9", "50x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/50x400", "12-9", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/60x400", "12-9", "60x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/30x400_newalgo", "12-9", "30x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170127_SDS20_Chip_12-9/30x400_oldalgo", "12-9", "30x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/50x60", "12-8", "50x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/50x400", "12-8", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/30x400", "12-8", "30x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/30x60", "12-8", "30x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170130_SDS20_Chip_12-8/60x400", "12-8", "60x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170202_SDS20_Chip_11-9/50x60", "11-9", "50x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170202_SDS20_Chip_11-9/50x400", "11-9", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170202_SDS20_Chip_11-9/30x400", "11-9", "30x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170202_SDS20_Chip_11-9/30x60", "11-9", "30x60"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170202_SDS20_Chip_11-9/60x400", "11-9", "60x400"]
]

#####################################################################
####### EFFECT OF DELTARJ AND R^2 FOR ALL PREPPED TRACES ##########
#####################################################################
dataf = datafiles[24]
deltarjs = np.arange(10, 18, 1)
rsqrd_cutoffs = np.linspace(0.992, 0.998, 7)
params = [(x, y) for x in deltarjs for y in rsqrd_cutoffs]

path, chip, group = dataf
print(path)
os.chdir(path)
env = "He_gas"
for f in os.listdir(path):
    if "fit_results_FINALEM" in f or "pc_analysis" in f or "Trace_Fit_Combo_FINALEM" in f:
        os.remove(f)

prep_all_FINALEM(path)

settings = []
results = pd.Series()
for idx, param in enumerate(params):
    deltarj, rsqrd_cutoff = param
    fit_all(path, ratios, group=group, env=env, chip=chip, deltarj=deltarj)
    if idx == 0:
        plot_trace_fit_combo(path, zoom=True, maxrj=deltarj + 15, deltarj=deltarj)
    ps = optimize_pc(path, deltarj, rsqrd_cutoffs=[rsqrd_cutoff, rsqrd_cutoff])
    settings += [idx]*len(ps)
    results = pd.concat([results, ps])
plt.close("all")
fig, ax = snp.newfig()
snp.labs("deltarj and $R^2$ Cutoff Setting", "$P_c$ for Points above $R^2$ Cutoff",
         "Investigate Parameter Effect - %s,%s" % (chip, group))
# Plot a vertical scatter plot of Pc of included devices at each setting, color code by device ID
df = pd.DataFrame({"pc_ratio": results, "ports": results.index, "params": settings})
cmap = snp.cmap(ncols=len(df["ports"].unique()))
for nm, grp in df.groupby("ports"):
    ax.plot(grp["params"].values, grp["pc_ratio"].values, color=next(cmap), markersize=5, label=nm, linestyle="None")
ax.legend(loc="upper right")
plt.savefig("OptimalParams_Scatter.png")


#####################################################################
#######  REVISE PREP FOR ANY DEVICES THAT MIGHT BENEFIT   ##########
#####################################################################
prep_device("FT1_1618", path, deltarj=20)
prep_device("FT1_0403", path, deltarj=20)
prep_device("FT1_2321", path, deltarj=20)
prep_device("FT2_0908", path, deltarj=20)


#####################################################################
####### LIST THE OUTLIER DEVICES THAT WILL BE EXCLUDED ##########
#####################################################################
exclude = ["FT1_1618"]


#####################################################################
####### EFFECT OF DELTARJ AND R^2 FOR INCLUDED TRACES ONLY ##########
#####################################################################
fig, ax = snp.newfig()
snp.labs("deltarj and $R^2$ Cutoff Setting", "$P_c$ for Points above $R^2$ Cutoff",
         "Investigate Parameter Effect - %s,%s" % (chip, group))
mus, stds = [], []
for idx, param in enumerate(params):
    deltarj, rsqrd_cutoff = param
    mu, std = optimize_pc_exclude(path, deltarj, rsqrd_cutoffs=[rsqrd_cutoff, rsqrd_cutoff], exclude=exclude)
    mus.append(mu)
    stds.append(std)
ax.plot(np.arange(len(params)), mus, linestyle="None")
plt.errorbar(np.arange(len(params)), mus, yerr=stds, marker="", markersize=10, color="black", linestyle="")
plt.savefig("OptimalParams_Errorbar.png")


#####################################################################
####### CLEAN UP AND FIT WITH THE CHOSEN OPTIMAL SETTINGS ##########
#####################################################################
# Remove all the df fitting files and pc_analysis in this directory
for f in os.listdir(path):
    if "fit_results" in f or "pc_analysis" in f or "Trace_Fit_Combo" in f:
        os.remove(f)
deltarj, rsqrd_cutoff = params[42]
print(deltarj, rsqrd_cutoff)
fit_all(path, ratios, group=group, env=env, chip=chip, deltarj=deltarj)
plot_trace_fit_combo(path, zoom=True, maxrj=deltarj + 15, deltarj=deltarj)
dfpickle = "fit_results_df_deltarj_%i.pickle" % (deltarj,)
set_pc_conform(dfpickle, names=exclude, values=[0]*len(exclude))

with open(dfpickle, "rb") as myfile:
    fit_df = pickle.load(myfile)
