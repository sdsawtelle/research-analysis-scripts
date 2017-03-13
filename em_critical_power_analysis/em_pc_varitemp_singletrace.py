"""This module provides functionality to plot, prepare and fit EM behavior where traces are taken partially at one
temperature, and then finished at a second temperature, BUT THE ANALYSIS ONLY INCLUDES THE FINAL TRACES. Each EM trace
is fitted to the constant-power model for electromigration. The user is able to restrict what parts of the trace should
be included in analysis to exclude, for instance, later ramp cycles once the junction resistance is high and the device
is unstable. Once the trace has been culled in this fashion, the analysis consists of identifying the current and
voltage points where EM occurs, and then fitting that set of points using a constant critical power model.

The data for this experiment consist of EM text files where each device (specified by a unique feedthrough and port
combination, as in "FT1_0102") should have two EM trace files associated with it, one containing "_partial" in the name
and one containing "_final." For full analysis we will ignore the partial traces and focus on the final traces.

The user is encouraged to refer to experimental notes to eliminate all known garbage traces from the directory. This
includes traces where the device was "stuck" or had an uncontrolled break.
The script below all the function definitions can be used to execute a full preparation and analysis of devices. Call
prep_all_FINALEM() to do all initial preparation of the good traces like selecting ramps for analysis. This creates and
pickles a dict of dicts whose keys are the trace names and whose values are a dict of the prepared DF and the res_tot_0
value. Optionally now you can call revise_prep() if you want to revise the preparation of any specific trace - it will
update the pickled dictionary. Now call fit_all_FINALEM() with kwargs rjmin and rjmax to fit the ramps between rjmin and
rjmax for each trace to the constant power model. This function creates and pickles a dataframe whose indices are the
trace names and whose values are various demographic info about the device as well as the fit results. It also creates
and pickles a dict of dicts with the same information. The pickled dataframe has a column 'conforms' which denotes
whether the device appears to conform to the constant power model or not. This is initialized to 1 for all traces but
values can later be changed by calling set_pc_conforms(fname). The dict and df pickles are named with the specific rjmin
and rjmax values that were used for the fitting, so users can try different values of these kwargs and not overwrite
previous results."""

__author__ = 'Sonya'
__email__ = "sdsawtelle@gmail.com"
__version__ = "3.5"

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pylab
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
plt.interactive(True)
snp.prettyplot(matplotlib)
plt.close("all")


######################################################
# PLOTTING AND IMAGING #####
######################################################
def plot_ramps(df_ramps, fitcol=False, zoom=None, trace="", newfig=True):
    """Plot ramp cycle DF for ramps below rjmax, optionally with false ramps in magenta color."""

    # Optionally create new figure object or get current axes
    if newfig:
        fig1, ax1 = snp.newfig("ul")
        snp.labs("Voltage (V)", "Resistance ($\Omega$)", "EM Trace - " + trace)
    else:
        ax1, fig1 = plt.gca(), plt.gcf()

    # Plot all the ramps, optionally with good ramps in magenta
    for indx, row in df_ramps.iterrows():
        if not row["false_ramp"] and fitcol is True:
            ax1.plot(row["volt"], row["res"], marker='s', color="magenta", markersize=4, mew=0.0, label=indx)
        else:
            ax1.plot(row["volt"], row["res"], mew=0.0, label=indx)

    # Optionally zoom into a low-resistance region
    if zoom:
        maxres = df_ramps.iloc[0, df_ramps.columns.get_loc("res")][0] + zoom + 5
        ax1.set_ylim([ax1.get_ylim()[0], maxres])

    # Make sure an uncontrolled break isn't causing a stupid axis scale
    miny, maxy = ax.get_ylim()
    if maxy > 800:
        ax.set_ylim([miny, 800])

    return fig1, ax1


def plot_trace_fit_combo(path, maxrj):
    with open("prepped_traces_dict.pickle", "rb") as myfile:
        prepdict = pickle.load(myfile)

    with open("fit_results_df_maxrj_%i.pickle" % (maxrj,), "rb") as myfile:
        fitdf = pickle.load(myfile)

    # with open("fit_results_dict_deltarj_%i.pickle" % (deltarj,), "rb") as myfile:
    #     fitdict = pickle.load(myfile)

    # Get unique feedthrough/port identifiers corresponding to unique devices
    fnames = snp.txtfilenames(path)
    ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    ports = list(set(ports))

    # For each unique port, plot the color-coded ramp cycles and the crit.compute_pow. fit
    for port in ports:
        fig = plt.figure(1, figsize=[12, 6])  # create one of the figures that must appear with the chart
        axs = [plt.subplot2grid((1, 2), (0, 0)), plt.subplot2grid((1, 2), (0, 1))]

        # Get the ramps and fit results for this device
        trace = [fname.strip(".txt") for fname in fnames if port in fname][0]
        fitdic = fitdf.loc[trace]
        full_df = prepdict[trace]["df"]
        cutoff = full_df.iloc[0, full_df.columns.get_loc("first_r")] + maxrj
        full_df.loc[full_df["first_r"] > cutoff, "false_ramp"] = True

        # Plot ramp cycles colored by whether included in analysis
        ax = axs[1]
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Resistance ($\Omega$)")
        plot_ramps(full_df, fitcol=True, newfig=False)

        # Plot data and fit result together
        ax = axs[0]
        plot_fit_variableseries(ax, fitdic["last_is"], fitdic["last_vs"], fitdic["crit_pow"], fitdic["r_series"], fitdic["alpha"])
        fig.tight_layout()
        fig.suptitle("$P_c$ Fit Dev %s" % (port,))
        plt.subplots_adjust(top=0.90)

        plt.gcf().savefig("Trace_Fit_Combo_%s" % (port,))
        plt.close("all")



######################################################
# PREPARING DATA FOR FITTING  #####
######################################################
def prep_all(path, overwrite=True):
    """For all unique devices in path, let the user choose which ramps to fit for both the partial and final EM traces
    of that device and set res_tot_0 from the first chosen ramp of the partial trace. Save the chosen ramps as DFs and
    pickle them with res_tot_0 as a dict."""

    # Get filenames and unique feedthrough/port identifiers corresponding to unique devices
    fnames = snp.txtfilenames(path)
    ports = ["_".join(name.strip(".txt").split("_")[-2:]) for name in fnames]
    ports = list(set(ports))

    # If not overwriting existing prep, open existing file to see which devs are done
    if not overwrite:
        try:
            with open("prepped_traces_dict.pickle", "rb") as myfile:
                dic = pickle.load(myfile)
        except:
            dic = {}

    # For each unique device, prep the trace (ptionally overwriting if already prepped)
    for port in ports:
        if overwrite:
            print("\n\n\n Prepare Traces for Device %s \n\n\n" % (port,))
            prep_device(port, path)
        else:
            bools = [port in key for key in dic.keys()]
            if not any(bools):
                print("\n\n\n Prepare Traces for Device %s \n\n\n" % (port,))
                prep_device(port, path)


def prep_device(port, path):

    # Create (or load if exists) the dict that will store prepped DFs and temperature for all traces.
    if os.path.isfile("prepped_traces_dict.pickle"):
        with open("prepped_traces_dict.pickle", "rb") as myfile:
            prepped_traces_dict = pickle.load(myfile)
    else:
        prepped_traces_dict = {}

    # Load data for this unique port ID (device)
    fname = [fname for fname in snp.txtfilenames(path) if port in fname][0]
    df = pd.read_csv(fname, header=1)
    df.columns = columns=["voltage", "current", "resistance", "volt_range", "curr_range"] 
    df = extract_ramps(df)  # Get all ramp cycles from data file
    with open(fname, "r") as f:  # Get the temperature from the header
        tempstr = [val for val in f.readlines(1)[0].split(";") if "TEMP" in val][0]  # Get the header and chunk it
        temp = float(re.search("[\d]+[.]*[\d]+", tempstr)[0])

    # Choose first and last valid ramps
    df.iloc[:choose_first_ramp(df), df.columns.get_loc("false_ramp")] = True
    df.iloc[choose_last_ramp(df) + 1:, df.columns.get_loc("false_ramp")] = True

    # Select any false ramps to be discarded from fitting
    df = choose_false_ramps(df)

    # Save the prepared dataframe and res_tot_0 to the existing dictionary and re-pickle
    prepped_traces_dict[fname.strip(".txt")] = {"df": df, "temp": temp}
    with open("prepped_traces_dict.pickle", "wb") as myfile:  # Pickle after every dev, b/c sometimes stuff crashes
        pickle.dump(prepped_traces_dict, myfile)

    return 1


def extract_ramps(df):
    """Take a raw DF of v, r, i and create a DF of ramp cycles."""

    # Figure out how many different ramp cycles there are and get indexes for their endpoints
    # v1 = 0.1, 0.11, 0.12, 0.13, 0.14, 0.11
    # v2 = v1.shift(-1)
    # v2 = 0.11, 0.12, 0.13, 0.14, 0.11, nan
    # v1 > v2 only at the index for the last point in a ramp cycle (final point will never be included due to NaN).
    rampbackpoints = df["voltage"] > df["voltage"].copy().shift(-1)
    rampbackpoints.iloc[0] = rampbackpoints.iloc[-1] = True  # Manually add 2nd and last points as ramp back points
    rampindx = rampbackpoints[rampbackpoints].index.tolist()  # Get Series index values for all the ramp back points.
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

    # Initially assume all ramps are good
    df_ramps["false_ramp"] = False

    return df_ramps


def choose_last_ramp(df):
    """Plot ramp cycle DF and have user select the last cycle to be considered in analysis."""
    print("\n\n\n Please select the LAST valid ramp cycle for analysis. \n\n\n")
    fig, ax = plot_ramps(df)
    last_trace = snp.whichtrace(ax)  # User manually clicks a point on graph
    plt.close(fig)
    last_trace = int(last_trace[0])  # whichtrace() returns a list of string labels
    return last_trace


def choose_first_ramp(df):
    """Plot ramp cycle DF and have user select the last cycle to be considered in analysis."""
    print("\n\n\n Please select the FIRST valid ramp cycle for analysis. \n\n\n")
    fig, ax = plot_ramps(df)
    first_trace = snp.whichtrace(ax)  # User manually clicks a point on graph
    plt.close(fig)
    first_trace = int(first_trace[0])  # whichtrace() returns a list of string labels
    return first_trace


def choose_false_ramps(df):
    """Plot ramp cycle DF and let user select which ramps are "false" rampbacks."""

    print("\n\n\n Select any ramps to toggle false ramp back identification. \n\n\n")
    fig, ax = plot_ramps(df, fitcol=False)
    fliptraces = snp.whichtrace(ax)  # User manually clicks false ramp traces on graph
    while fliptraces:  # False when the user only clicks the right mouse button and list is empty
        for trac in fliptraces:
            df.loc[int(trac), "false_ramp"] = ~ df.loc[int(trac), "false_ramp"]
        fliptraces = snp.whichtrace(ax)  # User manually clicks points on graph
    plt.close(fig)

    return df



######################################################
# FITTING EM TRACES #####
######################################################
def fit_all(path, group, env, chip, maxrj):
    """Fit constant power model to all prepared dataframes in path, and store results in new dict and df."""
    
    # Read in the dictionary of prepped traces holding DFs of good ramps and get list of traces
    with open("prepped_traces_dict.pickle", "rb") as myfile:
        prepped_traces_dict = pickle.load(myfile)
    traces = list(prepped_traces_dict.keys())

    # Create (or load if exists) dict and df that will store fit results on all traces
    dfpickle = "fit_results_df_maxrj_%i.pickle" % (maxrj,)
    if os.path.isfile(dfpickle):
        with open(dfpickle, "rb") as myfile:
            fit_df = pickle.load(myfile)
        conforms = fit_df["pc_conforming"].values  # Save the conforming designation, will write it back in later
        fit_df = fit_df.drop("pc_conforming", axis=1)
    else:
        fit_df = pd.DataFrame(columns=["crit_pow", "r_series", "rj0", "res_tot_0", "group",
                                       "temp", "env", "chip", "r_sqrd", "last_vs", "last_is", "first_rs"], index=traces)

    # Do fitting on each trace and store results + demographics in the dict and df
    for trace in traces:
        # Restrict to only the non-false ramps within the desired resistance range
        df = prepped_traces_dict[trace]["df"]
        df = df[~df["false_ramp"]]
        temp = prepped_traces_dict[trace]["temp"]
        cutoff = df.iloc[0, df.columns.get_loc("first_r")] + maxrj
        df = df[df["first_r"] < cutoff]

        # cutoff = df.iloc[0, df.columns.get_loc("first_r")] + 20
        # df = df[df["first_r"] > cutoff]

        last_vs = df["last_v"].values
        last_is = df["last_i"].values
        first_rs = df["first_r"].values
        res_tot_0 = first_rs[0]

        # Do the Fit and save results
        lsq_fit, r_sqrd = do_fit(last_vs, last_is)
        rj0 = res_tot_0 - lsq_fit.x[1]
        fit_df.loc[trace] = [lsq_fit.x[0], lsq_fit.x[1], rj0,
                             res_tot_0, group, temp, env, chip, r_sqrd,
                             last_vs, last_is, first_rs]

    # If the "conforms" column already existed then we set it to the pre-existing values
    if "conforms" in locals():
        fit_df["pc_conforming"] = conforms
    else:
        fit_df["pc_conforming"] = 1

    with open(dfpickle, "wb") as myfile:
        pickle.dump(fit_df, myfile)

    return fit_df


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


def plot_fit(ax, last_is, last_vs, p_c, r_series):
    """Plot the critical (I,V) points with the constant-power fit overlaid."""

    rntots = last_vs/last_is
    rjs = rntots - r_series
    ax.set_xlabel("Junction Resistance ($\Omega$)")
    ax.set_ylabel("Critical Current (mA)")
    ax.plot(last_vs, last_is * 1000, marker="o", color="red", markersize=9, linestyle="", label="data")
    predict_curr = np.sqrt((p_c / (rjs)).astype("float64"))
    ax.plot(last_vs, predict_curr * 1000, label="fit ($P_c$ = %.3f mW, $R_0^J$ = %.0f $\Omega$)" % (p_c * 1000, rjs[0]))
    ax.legend(loc="lower left")

    # # Inset the Residuals
    # left, bottom, width, height = [0.6, 0.6, 0.25, 0.25]  # Unitless percentages of the fig size. (0,0 is bottom left)
    # ax_inset = fig.add_axes([left, bottom, width, height])
    # ax_inset.set_title("Residuals of Voltage", fontsize=12)
    # ax_inset.set_xlabel("Critical Point", fontsize=12)
    # ax_inset.plot(predict_volt(p_c, r_series, last_is) - last_vs, marker="x", markersize=8, linestyle="", color="red")
    # ax_inset.axhline(0, color="black")
    # ax_inset.get_xaxis().set_visible(False)
    # ax_inset.get_yaxis().set_visible(False)

    # Textbox with more info
    ax.text(0.02, 0.2, "$R_0^{tot}$ = %.0f $\Omega$" % (rntots[0],),
            transform=ax.transAxes, fontsize=14, verticalalignment="top")
    return ax



#
#
# ######################################################
# # FITTING EM TRACES WITH CURRENT-DEPENDENT-SERIES RESISTANCE #####
# ######################################################
# def fit_all_variableseries(path, group, env, chip, maxrj):
#     """Fit constant power model to all prepared dataframes in path, and store results in new dict and df."""
#     # Read in the dictionary of prepped traces (dfs and res_tot_0's) and get list of traces
#     with open("prepped_traces_dict.pickle", "rb") as myfile:
#         prepped_traces_dict = pickle.load(myfile)
#     traces = list(prepped_traces_dict.keys())
#
#     # Create (or load if exists) dict and df that will store fit results on all traces
#     dfpickle = "fit_results_df_maxrj_%i.pickle" % (deltarj,)
#     dictpickle = "fit_results_dict_maxrj_%i.pickle" % (deltarj,)
#     if os.path.isfile(dfpickle):
#         with open(dfpickle, "rb") as myfile:
#             fit_df = pickle.load(myfile)
#         conforms = fit_df["pc_conforming"].values  # Save the conforming designation, will write it back in later
#         fit_df = fit_df.drop("pc_conforming", axis=1)
#     else:
#         fit_df = pd.DataFrame(columns=["crit_pow", "r_series", "rj0", "res_tot_0", "group",
#                                        "temp", "env", "chip", "r_sqrd", "last_vs", "last_is", "first_rs"], index=traces)
#
#     # Do fitting on each trace and store results + demographics in the dict and df
#     for trace in traces:
#         df = prepped_traces_dict[trace]["df"]
#
#         # Fit for the slope of R vs. I to get factor for variable R_series
#         curr_dRdT = df.iloc[0, df.columns.get_loc("curr")][:-150]
#         curr_zero = curr_dRdT[0]
#         res_dRdT = df.iloc[0, df.columns.get_loc("res")][:-150]
#         def residuals_dRdT(theta, curr, res):
#             a, b = theta
#             return res - (a * curr + b)
#         lsq_eta = least_squares(residuals_dRdT, [6000, 400], args=(curr_dRdT, res_dRdT), loss="soft_l1")
#         alpha = lsq_eta.x[0]
#
#         # Restrict to only the non-false ramps within the desired resistance range
#         df = df[~df["false_ramp"]]
#         temp = prepped_traces_dict[trace]["temp"]
#         cutoff = df.iloc[0, df.columns.get_loc("first_r")] + maxrj
#         df = df[df["first_r"] < cutoff]
#         last_vs = df["last_v"].values
#         last_is = df["last_i"].values
#         first_rs = df["first_r"].values
#         res_tot_0 = first_rs[0]
#
#         # Do the Fit and save results
#         lsq_fit, r_sqrd = do_fit_variableseries(last_vs, last_is, alpha, curr_zero)
#         rj0 = res_tot_0 - lsq_fit.x[1]
#         fit_df.loc[trace] = [lsq_fit.x[0], lsq_fit.x[1], rj0,
#                              res_tot_0, group, temp, env, chip, r_sqrd,
#                              last_vs, last_is, first_rs]
#
#
#
#     #
#     # ########### FIT PARTIAL EM TRACES ##################
#     # # Restrict to just the partial EM traces
#     # traces = [trace for trace in all_traces if "partial" in trace]
#     #
#     # # Do fitting on each trace and store results + demographics in the dict and df
#     # for trace in traces:
#     #     # Restrict to only the ramps within the desired resistance range
#     #     df = prepped_traces_dict[trace]["df"]
#     #     temp = prepped_traces_dict[trace]["temp"]
#     #     cutoff = df.iloc[-1, df.columns.get_loc("first_r")] - deltarj
#     #     df = df[df["first_r"] > cutoff]
#     #     volt = df["last_v"].values
#     #     curr = df["last_i"].values
#     #     res_tot_0 = df.iloc[0, df.columns.get_loc("first_r")]
#     #
#     #     # Fit for the slope of R vs. I to get factor for variable R_series
#     #     curr_dRdT = df.iloc[0, df.columns.get_loc("curr")][:30]
#     #     curr_zero = curr_dRdT[0]
#     #     res_dRdT = df.iloc[0, df.columns.get_loc("res")][:30]
#     #
#     #     def residuals_dRdT(theta, curr, res):
#     #         a, b = theta
#     #         return res - (a * curr + b)
#     #
#     #     lsq_eta = least_squares(residuals_dRdT, [6000, 400], args=(curr_dRdT, res_dRdT), loss="soft_l1")
#     #     alpha = lsq_eta.x[0]
#     #
#     #     # Do the Fit and save results
#     #     lsq_fit, r_sqrd = do_fit_variableseries(volt, curr, alpha, curr_zero)
#     #     fit_dict[trace] = {"df": df, "r_series": lsq_fit.x[1], "crit_pow": lsq_fit.x[0],
#     #                        "rj0": res_tot_0 - lsq_fit.x[1], "res_tot_0": res_tot_0, "group": group,
#     #                        "trace_type": "partial", "volt": volt, "curr": curr, "temp": temp, "env": env,
#     #                        "chip": chip, "r_sqrd": r_sqrd}
#     #     fit_df.loc[trace] = [lsq_fit.x[0], lsq_fit.x[1], res_tot_0 - lsq_fit.x[1],
#     #                          res_tot_0, group, "partial", temp, env, chip, r_sqrd]
#     #
#     # ########### FIT FINAL EM TRACES ##################
#     # # Restrict to just the final EM traces
#     # traces = [trace for trace in all_traces if "final" in trace]
#     #
#     # # Do fitting on each trace and store results + demographics in the dict and df
#     # for trace in traces:
#     #     # Prepare ramp cycles for fitting
#     #     df = prepped_traces_dict[trace]["df"]
#     #     temp = prepped_traces_dict[trace]["temp"]
#     #     cutoff = df.iloc[0, df.columns.get_loc("first_r")] + deltarj
#     #     df = df[df["first_r"] < cutoff]
#     #     volt = df["last_v"].values
#     #     curr = df["last_i"].values
#     #     res_tot_0 = df.iloc[0, df.columns.get_loc("first_r")]
#     #
#     #     # Get the fitted rj0 and r_series from the corresponding partial trace and scale them
#     #     port = "_".join(trace.split("_")[-2:])
#     #     rj0 = fit_df.loc[fit_df.index.str.contains("partial_" + port), "rj0"].values[0] * ratios[port]
#     #     r_series = fit_df.loc[fit_df.index.str.contains("partial_" + port), "r_series"].values[0] * ratios[port]
#     #
#     #     # Fit for the slope of R vs. I to get factor for variable R_series
#     #     curr_dRdT = df.iloc[0, df.columns.get_loc("curr")][:30]
#     #     res_dRdT = df.iloc[0, df.columns.get_loc("res")][:30]
#     #
#     #     def residuals_dRdT(theta_eta, curr, res):
#     #         a, b = theta_eta
#     #         return res - (a * curr + b)
#     #
#     #     lsq_eta = least_squares(residuals_dRdT, [6000, 400], args=(curr_dRdT, res_dRdT), loss="soft_l1")
#     #     alpha = lsq_eta.x[0]
#     #
#     #     # Use scaled r_series to fit and save results.
#     #     lsq_fit, r_sqrd = do_fit_variableseries(volt, curr, alpha, curr_zero, r_series)
#     #     # lsq_fit = do_fit_variableseries(volt, curr, alpha)
#     #
#     #     fit_dict[trace] = {"df": df, "r_series": r_series, "crit_pow": lsq_fit.x[0],
#     #                        "rj0": rj0, "res_tot_0": res_tot_0, "group": group,
#     #                        "trace_type": "final", "volt": volt, "curr": curr, "temp": temp, "env": env,
#     #                        "chip": chip, "r_sqrd": r_sqrd}
#     #     fit_df.loc[trace] = [lsq_fit.x[0], r_series, rj0,
#     #                          res_tot_0, group, "final", temp, env, chip, r_sqrd]
#
#     # If the "conforms" column already existed then we set it to the pre-existing values
#     if "conforms" in locals():
#         fit_df["pc_conforming"] = conforms
#     else:
#         fit_df["pc_conforming"] = 1
#
#     with open(dfpickle, "wb") as myfile:
#         pickle.dump(fit_df, myfile)
#
#     return fit_df
#
#
# def do_fit_variableseries(volt, curr, eta, eta_curr, r_series=None):
#     """Do constant power fit to the voltage / current ramp endpoints from a single prepared DF."""
#     # If no r_series is passed, fit where p_c and r_series are both free parameters
#     if not r_series:
#         # Define a function to compute the residuals between the real data and the predicted current
#         def residuals(theta, eta, eta_curr, volt, curr):
#             p_c = theta[0]
#             r_series = theta[1]
#             predicted_volt = predict_volt_variableseries(p_c, eta, eta_curr, r_series, curr)
#             # Note, without .astype() the returned array would actually have data type = "object" and would store pointers
#             # to the float values rather than storing the values themselves. This would throw an error in the LSQ.
#             return (volt - predicted_volt).astype(np.float64)
#
#         # Fit the model using LSQ loss (alternately, specify loss to get more outlier-robust fitting)
#         theta0 = [1e-4, 10]  # first param is critical power, second param is initial Rj
#         lsq = least_squares(residuals, theta0, args=(eta, eta_curr, volt, curr), loss="soft_l1")
#         r_sqrd = r2_score(volt, predict_volt_variableseries(lsq.x[0], eta, eta_curr, lsq.x[1], curr))
#
#     # If a value for r_series is passed, then only treat p_c as free
#     if r_series:
#         # Define a function to compute the residuals between the real data and the predicted current
#         def residuals(theta, r_series, eta, eta_curr, volt, curr):
#             p_c = theta[0]
#             predicted_volt = predict_volt_variableseries(p_c, eta, eta_curr, r_series, curr)
#             # Note, without .astype() the returned array would actually have data type = "object" and would store pointers
#             # to the float values rather than storing the values themselves. This would throw an error in the LSQ.
#             return (volt - predicted_volt).astype(np.float64)
#
#         # Fit the model using LSQ loss (alternately, specify loss to get more outlier-robust fitting)
#         theta0 = [1e-4]  # first param is critical power, second param is initial Rj
#         lsq = least_squares(residuals, theta0, args=(r_series, eta, eta_curr, volt, curr), loss="soft_l1")
#         r_sqrd = r2_score(volt, predict_volt_variableseries(lsq.x[0], eta, eta_curr, r_series, curr))
#
#     return lsq, r_sqrd
#
#
# def predict_volt_variableseries(p_c, eta, eta_curr, r_series, curr):
#     """Compute a predicted current given input resistance and model parameters."""
#     adjusted_r_series = r_series + eta * (curr - eta_curr)
#     predicted_volt = p_c / curr + curr * adjusted_r_series
#     return predicted_volt
#
#
# def plot_fit_variableseries(ax, last_is, last_vs, p_c, r_series):
#     """Plot the critical (I,V) points with the constant-power fit overlaid."""
#
#     rntots = last_vs/last_is
#     rjs = rntots - r_series
#     ax.set_xlabel("Junction Resistance ($\Omega$)")
#     ax.set_ylabel("Critical Current (mA)")
#     ax.plot(last_vs, last_is * 1000, marker="o", color="red", markersize=9, linestyle="")
#     predict_curr = np.sqrt((p_c / (rjs)).astype("float64"))
#     ax.plot(last_is, predict_curr * 1000, label="fit ($P_c$ = %.3f mW, $R_0^J$ = %.0f $\Omega$)" % (p_c * 1000, rjs[0]))
#     ax.legend(loc="lower left")
#
#     # # Inset the Residuals
#     # left, bottom, width, height = [0.6, 0.6, 0.25, 0.25]  # Unitless percentages of the fig size. (0,0 is bottom left)
#     # ax_inset = fig.add_axes([left, bottom, width, height])
#     # ax_inset.set_title("Residuals of Voltage", fontsize=12)
#     # ax_inset.set_xlabel("Critical Point", fontsize=12)
#     # ax_inset.plot(predict_volt_variableseries(p_c, r_series, curr) - volt, marker="x", markersize=8, linestyle="", color="red")
#     # ax_inset.axhline(0, color="black")
#     # ax_inset.get_xaxis().set_visible(False)
#     # ax_inset.get_yaxis().set_visible(False)
#
#     # Textbox with more info
#     ax.text(0.02, 0.2, "$R_0^{tot}$ = %.0f $\Omega$" % (rntots[0],),
#             transform=ax.transAxes, fontsize=14, verticalalignment="top")
#     return ax



######################################################
# FITTING EM TRACES WITH CURRENT-DEPENDENT-SERIES RESISTANCE #####
######################################################
def fit_all_variableseries(path, group, env, chip, maxrj):
    """Fit constant power model to all prepared dataframes in path, and store results in new dict and df."""
    # Read in the dictionary of prepped traces (dfs and res_tot_0's) and get list of traces
    with open("prepped_traces_dict.pickle", "rb") as myfile:
        prepped_traces_dict = pickle.load(myfile)
    traces = list(prepped_traces_dict.keys())

    # Create (or load if exists) dict and df that will store fit results on all traces
    dfpickle = "fit_results_df_maxrj_%i.pickle" % (maxrj,)
    dictpickle = "fit_results_dict_maxrj_%i.pickle" % (maxrj,)
    if os.path.isfile(dfpickle):
        with open(dfpickle, "rb") as myfile:
            fit_df = pickle.load(myfile)
        conforms = fit_df["pc_conforming"].values  # Save the conforming designation, will write it back in later
        fit_df = fit_df.drop("pc_conforming", axis=1)
    else:
        fit_df = pd.DataFrame(columns=["crit_pow", "r_series", "rj0", "res_tot_0", "group",
                                       "temp", "env", "chip", "r_sqrd", "last_vs", "last_is", "first_rs", "alpha"], index=traces)

    # Do fitting on each trace and store results + demographics in the dict and df
    for trace in traces:
        df = prepped_traces_dict[trace]["df"]

        # Fit for the slope of R vs. I to get factor for variable R_series
        curr_dRdT = df.iloc[0, df.columns.get_loc("curr")][:-150]
        curr_zero = curr_dRdT[0]
        res_dRdT = df.iloc[0, df.columns.get_loc("res")][:-150]
        def residuals_dRdT(theta, curr, res):
            a, b = theta
            return res - (a * curr + b)
        lsq_eta = least_squares(residuals_dRdT, [6000, 400], args=(curr_dRdT, res_dRdT), loss="soft_l1")
        alpha = lsq_eta.x[0]

        # Restrict to only the non-false ramps within the desired resistance range
        df = df[~df["false_ramp"]]
        temp = prepped_traces_dict[trace]["temp"]
        cutoff = df.iloc[0, df.columns.get_loc("first_r")] + maxrj
        df = df[df["first_r"] < cutoff]
        last_vs = df["last_v"].values
        last_is = df["last_i"].values
        first_rs = df["first_r"].values
        res_tot_0 = first_rs[0]

        # Do the Fit and save results
        lsq_fit, r_sqrd = do_fit_variableseries(last_vs, last_is)
        rj0 = res_tot_0 - lsq_fit.x[1]
        fit_df.loc[trace] = [lsq_fit.x[0], lsq_fit.x[1], rj0,
                             res_tot_0, group, temp, env, chip, r_sqrd,
                             last_vs, last_is, first_rs, lsq_fit.x[2]]

    # If the "conforms" column already existed then we set it to the pre-existing values
    if "conforms" in locals():
        fit_df["pc_conforming"] = conforms
    else:
        fit_df["pc_conforming"] = 1

    with open(dfpickle, "wb") as myfile:
        pickle.dump(fit_df, myfile)

    return fit_df


def do_fit_variableseries(volt, curr, r_series=None):
    """Do constant power fit to the voltage / current ramp endpoints from a single prepared DF."""
    # If no r_series is passed, fit where p_c and r_series are both free parameters
    if not r_series:
        # Define a function to compute the residuals between the real data and the predicted current
        def residuals(theta, volt, curr):
            p_c = theta[0]
            r_series = theta[1]
            alpha = theta[2]
            predicted_volt = predict_volt_variableseries(p_c, alpha, r_series, curr)
            # Note, without .astype() the returned array would actually have data type = "object" and would store pointers
            # to the float values rather than storing the values themselves. This would throw an error in the LSQ.
            return (volt - predicted_volt).astype(np.float64)

        # Fit the model using LSQ loss (alternately, specify loss to get more outlier-robust fitting)
        theta0 = [1.8e-4, 401, 0.001]  # first param is critical power, second param is initial Rj
        lsq = least_squares(residuals, theta0, args=(volt, curr), loss="soft_l1",
                            bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, 31000]))
        r_sqrd = r2_score(volt, predict_volt_variableseries(lsq.x[0], lsq.x[2], lsq.x[1], curr))

    return lsq, r_sqrd


def predict_volt_variableseries(p_c, alpha, r_series, curr):
    """Compute a predicted current given input resistance and model parameters."""
    adjusted_r_series = r_series + alpha * (curr - curr[0])
    predicted_volt = p_c / curr + curr * adjusted_r_series
    return predicted_volt


def plot_fit_variableseries(ax, last_is, last_vs, p_c, r_series, alpha):
    """Plot the critical (I,V) points with the constant-power fit overlaid."""

    rntots = last_vs/last_is
    adjusted_r_series = r_series + alpha * (last_is - last_is[0])
    rjs = rntots - adjusted_r_series
    ax.set_xlabel("Junction Resistance ($\Omega$)")
    ax.set_ylabel("Critical Current (mA)")
    ax.plot(last_vs, last_is * 1000, marker="o", color="red", markersize=9, linestyle="")
    predict_curr = np.sqrt((p_c / (rjs)).astype("float64"))
    ax.plot(last_vs, predict_curr * 1000, label="fit ($P_c$ = %.3f mW, $R_0^J$ = %.0f $\Omega$)" % (p_c * 1000, rjs[0]))
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
    ax.text(0.02, 0.2, "$R_0^{tot}$ = %.0f $\Omega$" % (rntots[0],),
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




############################################################################################################
datafiles = [
["C:/Users/Sonya/Documents/My Box Files/molT Project/170224_SDS20_Chip_12-5/50x400_109", "12-5_109", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170224_SDS20_Chip_12-5/50x400_125", "12-5_125", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170224_SDS20_Chip_12-5/50x400_173", "12-5_173", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170224_SDS20_Chip_12-5/50x400_189", "12-5_189", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170225_SDS20_Chip_10-5/50x400_76", "10-5_76", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170225_SDS20_Chip_10-5/50x400_93", "10-5_93", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170225_SDS20_Chip_10-5/50x400_141", "10-5_141", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170225_SDS20_Chip_10-5/50x400_157", "10-5_157", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170225_SDS20_Chip_10-5/50x400_205", "10-5_205", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170226_SDS20_Chip_13-5/50x400_230", "13-5_230", "50x400"],
["C:/Users/Sonya/Documents/My Box Files/molT Project/170226_SDS20_Chip_13-5/50x400_252", "13-5_252", "50x400"]
]


####### PREP TRACES ##########
dataf = datafiles[6]
# dataf = ["C:/Users/Sonya/Documents/My Box Files/molT Project/test", "18-8", "30x400"]
for dataf in datafiles:
    path, chip, group = dataf
    env = "He_gas"
    os.chdir(path)
    print("\n\n\n Analysis for Chip %s %s in folder" % (chip, group), path, "\n\n\n")
    prep_all(path, overwrite=False)
    prep_device("FT2_2021", path)  # Revise prep for any device as needed


####### FIT FOR DIFFERENT MAXRJ VALUES ##########
for dataf in datafiles:
    path, chip, group = dataf
    env = "He_gas"
    os.chdir(path)
    # Do Fit for every value of maxrj
    maxrjs = np.arange(60, 150, 20)
    for maxrj in maxrjs:
        df = fit_all_variableseries(path, group=group, env=env, chip=chip, maxrj=maxrj)  # Pickle fit dict and dfs


####### EFFECT OF MAXRJ AND R^2 ##########
for dataf in datafiles:
    path, chip, group = dataf
    env = "He_gas"
    os.chdir(path)
    maxrjs = np.arange(60, 150, 20)
    rsqrd_cutoffs = np.linspace(0.975, 0.998, 6)
    params = [(x, y) for x in maxrjs for y in rsqrd_cutoffs]
    results = pd.DataFrame()
    settings = []

    # Concatenate results from all possible paramter values
    idx = 0
    for maxrj in maxrjs:
        with open("fit_results_df_maxrj_%i.pickle" % (maxrj,), "rb") as myfile:
            df = pickle.load(myfile)
        for rsqrd_cutoff in rsqrd_cutoffs:
            temp_df = df[df["r_sqrd"] > rsqrd_cutoff]
            settings += [idx]*len(df)
            results = pd.concat([results, df])
            idx += 1

    # Plot fitted Pc scatter points at every value of parameters
    fig, ax = snp.newfig()
    snp.labs("Max $R_j$ and $R^2$ Cutoff Setting", "$P_c$ for Points above $R^2$ Cutoff",
             "Investigate Parameter Effect - %s,%s" % (chip, group))
    df = pd.DataFrame({"crit_pow": results["crit_pow"], "ports": results.index, "params": settings})
    cmap = snp.cmap(ncols=len(df["ports"].unique()))
    for nm, grp in df.groupby("ports"):
        ax.plot(grp["params"].values, grp["crit_pow"].values, color=next(cmap), markersize=5, label=nm, linestyle="None")
    ax.legend(loc="upper right")
    plt.savefig("OptimalParams_Scatter.png")
    plt.close("all")


####### PLOT TRACES WITH FITS AND PLOT FITTED PARAMETERS ##########
# Make plots of fit and trace at the true parameters for the final analysis
maxrj, rsqrd_cutoff = params[28]
for dataf in datafiles:
    path, chip, group = dataf
    env = "He_gas"
    os.chdir(path)

    plot_trace_fit_combo(path, maxrj)

    # Plot for identifying outliers for this group
    with open("fit_results_df_maxrj_%i.pickle" % (maxrj,), "rb") as myfile:
        df = pickle.load(myfile)
    df = df[df["r_sqrd"] > rsqrd_cutoff]
    cmap = snp.cmap(ncols=len(df))
    fig, ax = snp.newfig()
    snp.labs("Fitted $P_c$ (mW)", "Fitted $R^J_0$ ($\Omega$)", "Identifying Outliers")
    for nm, row in df.iterrows():
        ax.plot(1000 * row["crit_pow"], row["rj0"], color=next(cmap), label=nm, linestyle="None", markersize=10)
    pylab.legend(loc='best')
    plt.savefig("Fitted_Params.png")
    plt.close("all")


####### AGGREGATE RESULTS ACROSS ALL CHIPS AND GROUPS ##########
env = "He_gas"
maxrj, rsqrd_cutoff = params[28]
for k in (3 + 6*np.arange(5)):
    maxrj, rsqrd_cutoff = params[k]
    df_all = pd.DataFrame()
    for dataf in datafiles:
        path, chip, group = dataf
        os.chdir(path)
        with open("fit_results_df_maxrj_%i.pickle" % (maxrj,), "rb") as myfile:
            df = pickle.load(myfile)
        df = df[df["r_sqrd"] > rsqrd_cutoff]
        df_all = pd.concat([df_all, df])

    # Map chips to their temperature groups
    # tmap = {"10-9": 95, "4-11": 125, "12-9": 154.2, "12-8": 180.4, "11-9": 209.5, "10-8": 125.1, "11-8": 239.4, "13-8": 99.8, "13-9": 80.1}
    tmap = {"12-5_109": 109, "12-5_125": 125, "12-5_173": 173, "10-5_76": 76.5, "12-5_189": 189.1,
            "10-5_93": 92.5, "10-5_141": 141, "10-5_157": 156.5, "10-5_205": 205.5, "13-5_230": 229.5, "13-5_252": 252.2}
    df_all["T_approx"] = df_all["chip"].map(tmap)

    df_all["crit_pow"] = df_all["crit_pow"].astype("float") * 1000
    df_all["ports"] = df_all.index
    df_all["pc_conforms"] = 1  # Set which devices are excluded as non-conforming or outliers

    # df_all.loc[(df_all["chip"] == "10-5_93") & (df_all["ports"] == "FT1_2526"), "pc_conforms"] = 0
    # df_all.loc[(df_all["chip"] == "10-5_93") & (df_all["ports"] == "FT1_2726"), "pc_conforms"] = 0
    # df_all.loc[(df_all["chip"] == "10-5_93") & (df_all["ports"] == "FT3_2726"), "pc_conforms"] = 0
    # df_all.loc[(df_all["chip"] == "10-5_93") & (df_all["ports"] == "FT2_1008"), "pc_conforms"] = 0
    # df_all.loc[(df_all["chip"] == "10-5_141") & (df_all["ports"] == "FT2_1213"), "pc_conforms"] = 0
    # df_all.loc[(df_all["chip"] == "10-5_141") & (df_all["ports"] == "FT2_1413"), "pc_conforms"] = 0
    # df_all.loc[(df_all["chip"] == "10-5_141") & (df_all["ports"] == "FT2_1513"), "pc_conforms"] = 0

    # df_all.loc[(df_all["chip"] == "10-5_76") & (df_all["ports"] == "FT1_1718"), "pc_conforms"] = 0
    # df_all.loc[(df_all["chip"] == "12-5_173") & (df_all["ports"] == "FT3_0103"), "pc_conforms"] = 0
    # df_all.loc[(df_all["chip"] == "12-5_189") & (df_all["ports"] == "FT3_1211"), "pc_conforms"] = 0


    # Outliers from Variable Series Resistance Fits with Maxrj = 140
    df_all.loc[(df_all["chip"] == "12-5_173") & (df_all["ports"].str.contains("FT3_0103")), "pc_conforms"] = 0
    df_all.loc[(df_all["chip"] == "10-5_157") & (df_all["ports"].str.contains("FT2_2526")), "pc_conforms"] = 0
    df_all.loc[(df_all["chip"] == "10-5_205") & (df_all["ports"].str.contains("FT3_1516")), "pc_conforms"] = 0
    df_all.loc[(df_all["chip"] == "13-5_230") & (df_all["ports"].str.contains("FT2_0908")), "pc_conforms"] = 0
    df_all.loc[(df_all["chip"] == "13-5_252") & (df_all["ports"].str.contains("FT3_2826")), "pc_conforms"] = 0
    df_all.loc[(df_all["chip"] == "13-5_252") & (df_all["ports"].str.contains("FT2_2726")), "pc_conforms"] = 0

    # df_all.loc[(df_all["chip"] == "12-5_189") & (df_all["ports"] == "FT3_1211"), "pc_conforms"] = 0
    # df_all.loc[(df_all["chip"] == "10-5_93") & (df_all["ports"] == "FT1_2726"), "pc_conforms"] = 0
    # df_all.loc[(df_all["chip"] == "10-5_76") & (df_all["ports"] == "FT1_0403"), "pc_conforms"] = 0
    # df_all.loc[(df_all["chip"] == "10-5_205") & (df_all["ports"] == "FT3_2021"), "pc_conforms"] = 0
    # df_all.loc[(df_all["chip"] == "13-5_252") & (df_all["ports"] == "FT3_2826"), "pc_conforms"] = 0



    df_all_copy = df_all.copy()
    # with open("AllChips_FewerOutliers.pickle", "wb") as myfile:
    #     pickle.dump(df_all_copy, myfile)

    # Restrict to only chips and groups of interest and only included devices
    df_all = df_all[df_all["pc_conforms"] == 1]

    # Aggregate means and std deviations by group
    mus = df_all[["group", "T_approx", "crit_pow"]].groupby(["group", "T_approx"]).mean()
    lens = df_all[["group", "T_approx", "crit_pow"]].groupby(["group", "T_approx"]).count()
    stds = df_all[["group", "T_approx", "crit_pow"]].groupby(["group", "T_approx"]).std()
    mses = stds / lens.apply(np.sqrt)  # Squared error of mean (sigma_sample/sqrt(sample size))

    # Plot aggregated means with error bars
    markercols = ["r", "r", "b", "b", "r", "r", "b", "b", "r", "g", "g"]
    fig, ax = snp.newfig()
    snp.labs("Temperature (K)", "Fitted $P_c$ (mW)", "$P_c$ vs. $T$ 50x400 - Maxrj = %i" % (maxrj,))
    for grp in mus.index.levels[0]:
        # fig, ax = snp.newfig()
        # snp.labs("Temperature (K)", "Fitted $P_c$ (mW)", "$P_c$ vs. $T$ for %s" % (grp,))
        means = mus.loc[grp, "crit_pow"].values
        errs = mses.loc[grp, "crit_pow"].values
        temps = mus.loc[grp].index
        plt.errorbar(temps, means, yerr=errs, marker="", markersize=10, color="black", linestyle="")
        # ax.plot(temps, means, linestyle="None", markersize=7, label=grp, color=markercols)
        ax.scatter(temps, means, linestyle="None", s=40, label=grp, color=markercols)
    pylab.legend(loc="best")
    ax.set_xlim([65, 265])
plt.savefig("Pc_FinalEM_Aggregated.png")

with open("means_and_mses_50x400_varseries.pickle", "wb") as myfile:
    pickle.dump([mus, mses], myfile)


####### DELETE UNNCESSARY FILES ##########
# Delete all the unnecessary fits and plot the fitted r_series and pc for the chosen cutoffs
maxrj, rsqrd_cutoff = params[28]
for dataf in datafiles:
    path, chip, group = dataf
    os.chdir(path)
    for f in os.listdir(path):
        condition1 = "fit_results" in f or "pc_analysis" in f
        # condition2 = "maxrj_%i" % (maxrj,) not in f
        condition2 = True
        if (condition1 and condition2):
            os.remove(f)



####### CURRENT-DEPENDENCE OF SERIES RESISTANCE ##########
# Color by chip
cmap = {"12-5_109": "r", "12-5_125": "r", "12-5_173": "r", "10-5_76": "b", "12-5_189": "r",
        "10-5_93": "b", "10-5_141": "b", "10-5_157": "b", "10-5_205": "b", "13-5_230": "g", "13-5_252": "g"}

# Color by temperature, ordered
tmap = {"12-5_109": 109, "12-5_125": 125, "12-5_173": 173, "10-5_76": 76.5, "12-5_189": 189.1,
        "10-5_93": 92.5, "10-5_141": 141, "10-5_157": 156.5, "10-5_205": 205.5, "13-5_230": 229.5, "13-5_252": 252.2}
temp = sorted([(key, tmap[key]) for key in tmap.keys()], key=lambda x: x[1])
clist = zip(temp, snp.cmap(ncols=len(temp), name="cool", clist=True))
cmap = {x[0]: y for x, y in clist}

fig, ax = plt.subplots()
snp.labs("I (mA)", "R ($\Omega$)", "Current-Dependent $R_L$ - All Temps")
for dataf in datafiles:
    path, chip, group = dataf
    col = cmap[chip]
    os.chdir(path)
    with open("prepped_traces_dict.pickle", "rb") as myfile:
        dic = pickle.load(myfile)
    for key in dic:
        df = dic[key]["df"]
        volts = df.iloc[0, df.columns.get_loc("volt")]
        normidx = np.argmin(np.abs(volts-0.71))
        ress = df.iloc[0, df.columns.get_loc("res")]
        curr = df.iloc[0, df.columns.get_loc("curr")]
        ax.plot(volts[10:-150], ress[10:-150], ms=0.1, color=col, label=chip.split("_")[1] + " K")
plt.tight_layout()
# Restrict to just a single plot label for each temperature
handles, labels = ax.get_legend_handles_labels()
unqs, unqs_indx = np.unique(labels, return_index=True)
handles, labels = np.array(handles)[unqs_indx], np.array(labels)[unqs_indx]
temps = np.array([int(lab.split(" ")[0]) for lab in labels])
handles, labels = handles[np.argsort(temps)], labels[np.argsort(temps)]
ax.legend(handles, labels, loc="best")


####### TEMPERATURE DEPENDENCE OF FITTED ALPHA ##########
# Color by temperature, ordered
tmap = {"12-5_109": 109, "12-5_125": 125, "12-5_173": 173, "10-5_76": 76.5, "12-5_189": 189.1,
        "10-5_93": 92.5, "10-5_141": 141, "10-5_157": 156.5, "10-5_205": 205.5, "13-5_230": 229.5, "13-5_252": 252.2}
temp = sorted([(key, tmap[key]) for key in tmap.keys()], key=lambda x: x[1])
clist = zip(temp, snp.cmap(ncols=len(temp), name="cool", clist=True))
cmap = {x[0]: y for x, y in clist}

maxrj, rsqrd_cutoff = params[28]
df_all = pd.DataFrame()
for dataf in datafiles:
    path, chip, group = dataf
    os.chdir(path)
    with open("fit_results_df_maxrj_%i.pickle" % (maxrj,), "rb") as myfile:
        df = pickle.load(myfile)
    df = df[df["r_sqrd"] > rsqrd_cutoff]
    df_all = pd.concat([df_all, df])

# Map chips to their temperature groups
tmap = {"12-5_109": 109, "12-5_125": 125, "12-5_173": 173, "10-5_76": 76.5, "12-5_189": 189.1,
        "10-5_93": 92.5, "10-5_141": 141, "10-5_157": 156.5, "10-5_205": 205.5, "13-5_230": 229.5, "13-5_252": 252.2}
df_all["T_approx"] = df_all["chip"].map(tmap)

df_all["crit_pow"] = df_all["crit_pow"].astype("float") * 1000
df_all["ports"] = df_all.index
df_all["pc_conforms"] = 1  # Set which devices are excluded as non-conforming or outliers
df_all.loc[(df_all["chip"] == "10-5_76") & (df_all["ports"] == "FT1_1718"), "pc_conforms"] = 0
df_all.loc[(df_all["chip"] == "12-5_173") & (df_all["ports"] == "FT3_0103"), "pc_conforms"] = 0
df_all.loc[(df_all["chip"] == "12-5_189") & (df_all["ports"] == "FT3_1211"), "pc_conforms"] = 0

df_all_copy = df_all.copy()
# Restrict to only chips and groups of interest and only included devices
df_all = df_all[df_all["pc_conforms"] == 1]


fig, ax = plt.subplots()
snp.labs("Initial R ($\Omega$)", r"Fitted $\alpha$", r"Fitted $\alpha$ - All Temps")
colors = df_all["chip"].map(cmap).values
ax.scatter(df_all["res_tot_0"], df_all["alpha"], c=colors)
plt.tight_layout()

