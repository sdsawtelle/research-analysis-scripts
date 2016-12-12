""" This module does a semi-automated analysis for data taken in the "constant-voltage" or "curvature EM" experiments.
These experiments generate a time-series of points each of which specifies voltage, current and resistance. The analysis
is concerned with how the resistance behaves once electromigration begins, and the data acquisition algorithm is
supposed to catch this initial EM and cause the voltage to stop being ramped and to instead dwell at a fixed V.
Sometimes the algorithm does not catch the EM early enough though so our analysis can include data points across a
voltage range. This analysis is only semi-automated b/c the user needs to make a decision about what section of each
trace to keep for analysis, as well as where on the traces we should apply different levels of smoothing. Finally the
user needs to decide what range of the data we fit an exponential to (the behavior deviates from exponential at some
point.)
"""

__author__ = 'Sonya'
__email__ = "sdsawtelle@gmail.com"
__version__ = "3.5"


import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline
import importlib

import snips as snp  # my useful snippets module


snp.prettyplot(mpl)  # Change default aesthetics for matplotlib
mpl.interactive(True)  # Turn on interactive backend for matplotlib
lowess = sm.nonparametric.lowess  # Rename module method for easier access


def smoothtrace(datadir, filename):
    '''This is the top level wrapper for the full smoothing of a given trace. It writes the smoothed stuff to a csv.'''
    datafile = datadir + filename
    expdata = snp.captureheads(datafile, 1)  # Capture header for future reference
    df = pd.read_csv(datafile, header=1, sep=",")  # header is the num lines to skip, after that first line is col names

    # Lets assume always the same ordering of columns, but not take the exact names for granted
    oldnames = df.columns.tolist()
    newnames = ["time", "volt", "curr", "res"]
    namedict = dict(pair for pair in zip(oldnames, newnames))
    df.rename(columns=namedict, inplace=True)  # inplace modifies this DF

    ### USE THIS SECTION FOR OLD FILES WITH ONLY THREE COLUMNS
    # # Lets assume always the same ordering of columns, but not take the exact names for granted
    # oldnames = df.columns.tolist()
    # newnames = ["volt", "curr", "res"]
    # namedict = dict(pair for pair in zip(oldnames, newnames))
    # df.rename(columns=namedict, inplace=True)  # inplace modifies this DF
    # df["time"] = df.index


    #################################
    # TRUNCATE THE DATA
    #################################
    # Plot full Resistance vs. Time trace and let user pick the start and stop points
    snp.newfig("ul")
    plt.plot(df.time.values, df.res.values, marker='o')
    plt.xlabel("Time (Arb. Units)")
    plt.ylabel("R ($/Omega$)")
    plt.title("Full R Trace")
    endptindxs, endptvals = snp.pickdatapoints(df.time.values, df.res.values, 2)  # ginput has same effect as plt.show()
    plt.close()
    starttime = df.time.values[endptindxs[0]]
    stoptime = df.time.values[endptindxs[1]]

    # Crop the df down to just the range between the two user-specified points
    df = df[(df.time >= starttime) & (df.time <= stoptime)]
    # Define a new variable, the junction resistance, by subtracting out the series resistance
    df["rj"]=df.res-df.res.iloc[0]

    # Create numpy arrays for more convenient manipulation
    rj = df.rj.values
    curr = df.curr.values
    time = df.time.values


    #################################
    # SMOOTH RESISTANCE
    #################################
    snp.newfig("ul")
    plt.plot(time, rj, color="red")
    plt.xlabel("Time (Arb. Units)")
    plt.ylabel("Rj ($/Omega$)")
    plt.title("Truncated R Trace")

    # Choose smoothing regions for the resistance (and current). The user chooses points which are the end of the flat
    # region, the end of the bottom elbow region, the end of the upshoot region and the end of the top elbow region.
    endptindxs, endptvals = snp.pickdatapoints(time, rj, 4)  # ginput has same effect as plt.show()
    endflat, endelbow, endupshoot, endturnover = endptindxs
    plt.close()

    flat = windowsmoother(rj[:endflat], time[:endflat], 8, 0.05, 0.1)
    elbow = windowsmoother(rj[endflat:endelbow], time[endflat:endelbow], 4, 0.1, 0.07)
    upshoot = rj[endelbow:endupshoot].tolist()
    # upshoot = windowsmoother(rj[endelbow:endupshoot], time[endelbow:endupshoot], 3, 0.02, 0.15)
    turnover = windowsmoother(rj[endupshoot:endturnover], time[endupshoot:endturnover], 6, 0.03, 0.05)
    topoff = windowsmoother(rj[endturnover:], time[endturnover:], 8, 0.05, 0.1)

    # Recombine all the regions into one smooth trace, store as a new variable
    smrj = np.array(flat + elbow + upshoot + turnover + topoff)

    # The above is really sloppy. Better would be using .split on an numpy array, but note then that it will be expecting
    # integer location indexing whereas window smoother is returning actual pandas series indexes. Consider making this
    # nicer in the future


    axres, figres = snp.newfig()
    axres.plot(time, rj, color="red")
    plt.xlabel("Time (Arb. Units)")
    plt.ylabel("R ($/Omega$)")
    plt.title("Smoothed Rj Trace")
    axres.plot(time, smrj, color="green")

    # This seems to make things worse always...?
    # smrjfinal = windowsmoother(smrj, time, 1, 0.01, 0.04)
    # ax1.plot(time, smrjfinal, color="blue")

    #################################
    # SMOOTH THE POWER
    #################################
    pow = curr ** 2 * smrj

    snp.newfig("ll")
    plt.plot(time, pow, color="red")
    plt.xlabel("Time (Arb. Units)")
    plt.ylabel("Power (W)")
    plt.title("Smoothed Power Trace")
    
    endptindxs, endptvals = snp.pickdatapoints(time, pow, 3)  # ginput has same effect as plt.show()
    endramp, endflat, endturn = endptindxs

    # rampi = windowsmoother(curr[:endramp], time[:endramp], 12, 0.03, 0.2)
    # flati = windowsmoother(curr[endramp:endflat], time[endramp:endflat], 8, 0.04, 0.08)
    # turni = windowsmoother(curr[endflat:endturn], time[endflat:endturn], 8, 0.04, 0.08)
    # downshooti = windowsmoother(curr[endturn:], time[endturn:], 3, 0.03, 0.04)
    # smcurr = np.array(rampi + flati + turni + downshooti)


    ramp_ramp = windowsmoother(pow[:endramp], time[:endramp], 8, 0.05, 0.1)
    flat_ramp = windowsmoother(pow[endramp:endflat], time[endramp:endflat], 8, 0.05, 0.1)
    turn_ramp = windowsmoother(pow[endflat:endturn], time[endflat:endturn], 4, 0.1, 0.07)
    downshoot_ramp = windowsmoother(pow[endturn:], time[endturn:], 2, 0.03, 0.06)
    smpow = np.array(ramp_ramp + flat_ramp + turn_ramp + downshoot_ramp)

    plt.plot(time, smpow, color="green")


    #################################
    # SPLINE THE RESISTANCE AND GET RATE
    #################################
    # Spline the resistance and current traces
    splfit_r = UnivariateSpline(time, smrj, k=3, s=100)
    spl_r = splfit_r(time)
    axres.plot(time, spl_r, color="orange")
    #
    # splfit_i = UnivariateSpline(df.time, df.i, k=3, s=70)
    # spl_i = splfit(df.time)
    #
    # Get the derivative of resistance with time from the analytical derivative of the spline
    splfit_rate = splfit_r.derivative(n=1)  # n is order of derivative
    spl_rate = splfit_rate(time)

    # snp.newfig("lr")
    # plt.plot(time, spl_rate, marker='o')
    # plt.xlabel("time (Arb. Units)")
    # plt.ylabel("Rate ")
    # plt.title("Rate ")

    #
    # #################################
    # # SMOOTH THE RATE
    # #################################
    # # Choose smoothing regions for the breaking rate. The user chooses points which are the end of the flat
    # # region, the end of the bottom elbow region, the left side of the peak, and the right side of the peak.
    # endptindxs, endptvals = snp.pickdatapoints(time, spl_rate, 4)  # ginput has same effect as plt.show()
    # endflat, endelbow, endupshoot, endturnover = endptindxs
    #
    # flat_rate = windowsmoother(spl_rate[:endflat], time[:endflat], 8, 0.05, 0.1)
    # elbow_rate = windowsmoother(spl_rate[endflat:endelbow], time[endflat:endelbow], 4, 0.1, 0.1)
    # upshoot_rate = spl_rate[endelbow:endupshoot].tolist()
    # turnover_rate = windowsmoother(spl_rate[endupshoot:endturnover], time[endupshoot:endturnover], 6, 0.03, 0.08)
    # topoff_rate = windowsmoother(spl_rate[endturnover:], time[endturnover:], 8, 0.05, 0.1)
    #
    # smrate = np.array(flat_rate + elbow_rate + upshoot_rate + turnover_rate + topoff_rate)
    #
    #
    # # smrate = windowsmoother(smrate, time, 3, 0.05, 0.08)
    # plt.plot(time, smrate, color="orange")


    #################################
    # SMOOTH THE POWER
    #################################
    # Get the power from the smoothed resistance and current
    # pow = smcurr**2*smrj
    # splfit_pow = UnivariateSpline(time, pow, k=3)
    # spl_pow = splfit_pow(time)
    # smpow = windowsmoother(pow, time, 1, 0.05, 0.03)

    # snp.newfig("ur")
    # plt.plot(time, pow, marker='o')
    # plt.xlabel("time (Arb. Units)")
    # plt.ylabel("Power")
    # plt.title("Power")
    # endptindxs, endptvals = snp.pickdatapoints(time, spl_rate, 4)  # ginput has same effect as plt.show()
    # endflat, endelbow, endupshoot, endturnover = endptindxs
    #
    # flat_pow = windowsmoother(pow[:endflat], time[:endflat], 8, 0.05, 0.12)
    # elbow_pow = windowsmoother(pow[endflat:endelbow], time[endflat:endelbow], 6, 0.03, 0.07)
    # # upshoot_pow = windowsmoother(pow[endelbow:endupshoot], time[endelbow:endupshoot], 3, 0.02, 0.15)
    # upshoot_pow = pow[endelbow:endupshoot].tolist()
    # turnover_pow = windowsmoother(pow[endupshoot:endturnover], time[endupshoot:endturnover], 6, 0.03, 0.07)
    # topoff_pow = windowsmoother(pow[endturnover:], time[endturnover:], 8, 0.05, 0.12)
    #
    # smpow = np.array(flat_pow + elbow_pow + upshoot_pow + turnover_pow + topoff_pow)
    # plt.plot(time, smpow, color="orange")


    #################################
    # RATE VS. POWER FITTING
    #################################
    # snp.newfig("ll")
    # plt.plot(smpow, spl_rate, marker='o')
    # plt.ylabel("dRj/dt (Ohms/ms)")
    # plt.xlabel("Pj (Watts)")
    # plt.title("Rate vs. Power")
    # plt.plot(smpow, spl_rate, color="orange")
    # plt.plot(smpow, smrate, color="green")


    #################################
    # SAVE THE NEW VARIABLES
    #################################
    df["smrj"] = smrj
    df["rate"] = spl_rate
    df["power"] = smpow

    #################################
    # WRITE THE DATAFRAME TO TEXT FILE
    #################################
    csvfile = datadir + "SmoothedData/SMOOTHED_" + filename
    if not os.path.isfile(csvfile):
        with open(csvfile, 'w') as savefile:
            df.to_csv(savefile, header=True, index=None, sep=",")
    else:
        choice = input("Would you like to overwrite the existing SMOOTHED file? (1 for yes, 0 for no)")
        if choice:
            with open(csvfile, 'w') as savefile:
                df.to_csv(savefile, header=True, index=None, sep=",")

    return 1



def windowsmoother(ydata, xdata, nsmooths=1, endpt_percent=0.02, smoothspan=0.05):
    tempy = list(ydata)
    tempx = list(xdata)
    endpt_step = round(endpt_percent * len(tempy))

    for n in range(nsmooths, -1, -1):
        start = n * endpt_step
        stop = len(tempy) - n * endpt_step
        tempy[start:stop] = lowess(tempy[start:stop], tempx[start:stop], smoothspan, is_sorted=False)[:, 1]

    return tempy


#################################
# TRUNCATE / SMOOTH ALL FILES IN DIRECTORY
#################################
# Get all RAW data file names, create directory to hold truncated/smoothed data files
# datadir = "C:/Users/Sonya/Documents/My Box Files/Projects/python-em-breaking-rate/chip-9-6/"
# filenames = snp.txtfilenames(datadir)
# if not os.path.exists(datadir+"SmoothedData"):
#     os.mkdir(datadir+"SmoothedData")
# for filename in filenames:
#     smoothtrace(datadir, filename)
#     input("Press 'Enter' to continue with next file")
#     plt.close("all")



#################################
# PLOTTING AND MORE SMOOTHING OF TRUNCATED/SMOOTHED DATA FILES
#################################
# # Get all SMOOTHED data file names
# datadir = "C:/Users/Sonya/Documents/My Box Files/Projects/python-em-breaking-rate/OriginalData/SmoothedData/"
# filenames = snp.txtfilenames(datadir)
# # Create dictionary for what color to draw the different devices
# colordict = {"SMOOTHED_EMC10_FT2_0708.txt": "green",
# "SMOOTHED_EMC11_FT2_0908.txt": "green",
# "SMOOTHED_EMC11_FT2_1008.txt": "green",
# "SMOOTHED_EMC12_FT2_2122.txt": "red",
# "SMOOTHED_EMC13_FT2_2322.txt": "red",
# "SMOOTHED_EMC13_FT2_2426.txt": "red",
# "SMOOTHED_EMC13_FT2_2526.txt": "red",
# "SMOOTHED_EMC13_FT2_2726.txt": "red",
# "SMOOTHED_EMC14_FT3_2321.txt": "magenta",
# "SMOOTHED_EMC15_FT3_IntDevs_2321.txt": "magenta",
# "SMOOTHED_EMC16_FT3_IntDevs_1801.txt": "blue",
# "SMOOTHED_EMC17_FT3_IntDevs_0201.txt": "blue",
# "SMOOTHED_EMC18_FT2_IntDevs_0503.txt": "red",
# "SMOOTHED_EMC19_FT1_IntDevs_0910.txt": "green",
# "SMOOTHED_EMC19_FT1_IntDevs_1110.txt": "green",
# "SMOOTHED_EMC19_FT1_IntDevs_1210.txt": "green",
# "SMOOTHED_EMC2_FT1_1110.txt": "black",
# "SMOOTHED_EMC3_FT1_1314.txt": "black",
# "SMOOTHED_EMC4_FT1_1514.txt": "black",
# "SMOOTHED_EMC5_FT2_0103.txt": "blue"}
#
#
# # Create all Figs in preparation for plotting
# ax1, fig1 = snp.newfig("ll")
# plt.ylabel("$dR_j/dt$ ($/Omega$/ms)")
# plt.xlabel("$P_j$ (mW)")
# plt.title("Smoothed Rate vs. Power")
#
# ax2, fig2 = snp.newfig("ul")
# plt.ylabel("$R_j$ ($/Omega$)")
# plt.xlabel("$P_j$ (mW)")
# plt.title("Resistance vs. Power")
#
# ax3, fig3 = snp.newfig("ur")
# plt.ylabel("$R_j$ ($/Omega$)")
# plt.xlabel("Time (ms)")
# plt.title("Resistance vs. Time")
#
# ax4, fig4 = snp.newfig("lr")
# plt.ylabel("$dR_j/dt$ ($/Omega$/ms)")
# plt.xlabel("Time (ms)")
# plt.title("Unsmoothed Rate")
#
#
# for filename in filenames:
#     csvfile = datadir + filename
#     df = pd.read_csv(csvfile, header=0, sep=",")  # header is the num lines to skip, after that first line is col names
#
#     ## PLOTTING OF STUFF DIRECTLY FROM DATAFRAME
#     # ax1.plot(df.power*1000, df.rate, marker=None, color=colordict[filename])
#     ax2.plot(df.power*1000, df.smrj, marker='o', color=colordict[filename])
#     # ax3.plot(df.time, df.smrj, marker=None, color=colordict[filename])
#
#     ## NUMERICAL DIFFERENTIATION OF FULL RESISTANCE AND SMOOTHING OF DERIVATIVE
#     # time = df.time.values
#     # time1 = np.append(time[1:], time[-1]+95)
#     # time2 = time[:]
#     # deltatime = time1-time2
#     # res = df.res.values
#     # res = res-res[0]
#     # res1 = np.append(res[1:], res[-1])
#     # res2 = res[:]
#     # deltares = res1-res2
#     # diffres = deltares/deltatime
#     # smdiffres = windowsmoother(diffres, time, nsmooths=1, endpt_percent=0.01, smoothspan=0.01)
#     # smdiffres = windowsmoother(smdiffres, time, nsmooths=1, endpt_percent=0.01, smoothspan=0.02)
#     # smdiffres = windowsmoother(smdiffres, time, nsmooths=1, endpt_percent=0.02, smoothspan=0.03)
#     # # smdiffres = windowsmoother(smdiffres, time, nsmooths=1, endpt_percent=0.04, smoothspan=0.06)
#     # # smdiffres = windowsmoother(smdiffres, time, nsmooths=1, endpt_percent=0.04, smoothspan=0.1)
#     # # ax4.plot(df.time[:-1], diffres[:-1], marker=None, color="black")
#     # # ax4.plot(df.time[:-1], smdiffres[:-1], marker=None, color="orange")
#     # ax4.plot(df.power[:-1]*1000, smdiffres[:-1], marker=None, color=colordict[filename])
#
#
#     ## SMOOTHING OF FULL RESISTANCE
#     # time = df.time.values
#     # res = df.res.values
#     # ax4.plot(df.time, res, marker=None, color="black")
#     # smres = windowsmoother(res, time, nsmooths=1, endpt_percent=0.01, smoothspan=0.01)
#     # ax4.plot(df.time, smres, marker=None, color="blue")
#     # smres = windowsmoother(smres, time, nsmooths=1, endpt_percent=0.01, smoothspan=0.02)
#     # ax4.plot(df.time, smres, marker=None, color="green")
#     # smres = windowsmoother(smres, time, nsmooths=1, endpt_percent=0.02, smoothspan=0.03)
#     # ax4.plot(df.time, smres, marker=None, color="orange")
#
#
#     # SPLINING TESTING
#     time = df.time.values
#     res = df.res.values
#
#     # Define smoothing parameters for smoothing spline
#     s10 = len(res)*0.0022326
#     s50 = len(res)*0.0111632
#     s100 = len(res)*0.022326
#     s200 = len(res)*0.04465282
#     s300 = len(res)*0.06697924
#     s400 = len(res)*0.0893056
#     s600 = 2*s300
#
#     # plot the original data for reference
#     # ax4.plot(df.time, res, marker='o', color=colordict[filename])
#
#     ## INDIVIDUAL DEVICE TESTING
#     # splfit_r = UnivariateSpline(time, res, k=3, s=s10)
#     # smres = splfit_r(time)
#     # ax4.plot(df.time, smres, marker=None, color="brown")
#     # # splfit_rate = splfit_r.derivative(n=1)  # n is order of derivative
#     # # spl_rate = splfit_rate(time)
#     # # ax1.plot(time, spl_rate, marker=None, color="brown")
#     #
#     # splfit_r = UnivariateSpline(time, smres, k=3, s=s50)
#     # smres = splfit_r(time)
#     # ax4.plot(df.time, smres, marker=None, color="blue")
#     # splfit_r = UnivariateSpline(time, smres, k=3, s=s100)
#     # smres = splfit_r(time)
#     # ax4.plot(df.time, smres, marker=None, color="darkturquoise")
#     # splfit_r = UnivariateSpline(time, smres, k=3, s=s200)
#     # smres = splfit_r(time)
#     # ax4.plot(df.time, smres, marker=None, color="green")
#     #
#     # splfit_r = UnivariateSpline(time, smres, k=3, s=s300)
#     # smres = splfit_r(time)
#     # ax4.plot(df.time, smres, marker=None, color="orange")
#     # # splfit_rate = splfit_r.derivative(n=1)  # n is order of derivative
#     # # spl_rate = splfit_rate(time)
#     # # ax1.plot(time, spl_rate, marker=None, color="orange")
#     #
#     # splfit_r = UnivariateSpline(time, smres, k=3, s=s400)
#     # smres = splfit_r(time)
#     # ax4.plot(df.time, smres, marker=None, color="red")
#     # splfit_rate = splfit_r.derivative(n=1)  # n is order of derivative
#     # spl_rate = splfit_rate(time)
#     # ax1.plot(time, spl_rate, marker=None, color="red")
#
#     # splfit_r = UnivariateSpline(time, smres, k=2, s=10*s300)
#     # smres = splfit_r(time)
#     # ax4.plot(df.time, smres, marker=None, color="darkviolet")
#     # splfit_rate = splfit_r.derivative(n=1)  # n is order of derivative
#     # spl_rate = splfit_rate(time)
#     # ax1.plot(time, spl_rate, marker=None, color="darkviolet")
#
#
#     ## PLOTTING ALL AT ONCE
#
#     # splfit_r = UnivariateSpline(time, res, k=5, s=s10)
#     # smres = splfit_r(time)
#     # splfit_r = UnivariateSpline(time, smres, k=5, s=s50)
#     # smres = splfit_r(time)
#     # splfit_r = UnivariateSpline(time, smres, k=5, s=s100)
#     # smres = splfit_r(time)
#     # splfit_r = UnivariateSpline(time, smres, k=5, s=s200)
#     # smres = splfit_r(time)
#     # splfit_r = UnivariateSpline(time, smres, k=5, s=s300)
#     # smres = splfit_r(time)
#     # # splfit_r = UnivariateSpline(time, smres, k=5, s=2*s300)
#     #
#     # splfit_rate = splfit_r.derivative(n=1)  # n is order of derivative
#     # spl_rate = splfit_rate(time)
#     #
#     # ax1.plot(df.power[:-3], spl_rate[:-3], marker=None, color=colordict[filename], label=filename)
#     # ax4.plot(time, smres, marker=None, color=colordict[filename], label=filename)
#
#
#     ## SPLINING THE RATE
#     # splfit_r = UnivariateSpline(time, res, k=5, s=s10)
#     # smres = splfit_r(time)
#     # splfit_r = UnivariateSpline(time, smres, k=5, s=s50)
#     # smres = splfit_r(time)
#     # splfit_rate = splfit_r.derivative(n=1)  # n is order of derivative
#     # # spl_rate = splfit_rate(time)
#     #
#     # time1 = np.append(time[1:], time[-1]+95)
#     # time2 = time[:]
#     # deltatime = time1-time2
#     # res1 = np.append(res[1:], res[-1])
#     # res2 = res[:]
#     # deltares = res1-res2
#     # diffres = deltares/deltatime
#     #
#     #
#     # smdiffres = windowsmoother(diffres, time, nsmooths=1, endpt_percent=0.01, smoothspan=0.01)
#     # smdiffres = windowsmoother(smdiffres, time, nsmooths=1, endpt_percent=0.01, smoothspan=0.02)
#     # smdiffres = windowsmoother(smdiffres, time, nsmooths=1, endpt_percent=0.02, smoothspan=0.03)
#     # smdiffres = windowsmoother(smdiffres, time, nsmooths=1, endpt_percent=0.02, smoothspan=0.05)
#     # # smdiffres = windowsmoother(smdiffres, time, nsmooths=1, endpt_percent=0.04, smoothspan=0.06)
#     # # smdiffres = windowsmoother(smdiffres, time, nsmooths=1, endpt_percent=0.04, smoothspan=0.1)
#     #
#     # spl_rate = smdiffres
#     # # splfit_rate = UnivariateSpline(time, spl_rate, k=5, s=s10)
#     # # spl_rate = splfit_rate(time)
#     # # splfit_rate = UnivariateSpline(time, spl_rate, k=5, s=s50)
#     # # spl_rate = splfit_rate(time)
#     # # splfit_rate = UnivariateSpline(time, spl_rate, k=5, s=s100)
#     # # spl_rate = splfit_rate(time)
#     # # splfit_r = UnivariateSpline(time, spl_rate, k=5, s=s200)
#     # # spl_rate = splfit_rate(time)
#     # # splfit_r = UnivariateSpline(time, spl_rate, k=5, s=s300)
#     # # spl_rate = splfit_rate(time)
#     # # # splfit_r = UnivariateSpline(time, smres, k=5, s=2*s300)
#     #
#     # # splfit_rate = splfit_r.derivative(n=1)  # n is order of derivative
#     # # spl_rate = splfit_rate(time)
#     #
#     # ax1.plot(df.power[:-15], spl_rate[:-15], marker=None, color="blue", label=filename)
#     # ax4.plot(time, smres, marker=None, color=colordict[filename], label=filename)
#
#
#
#     # ax4.cla()
#     # ax1.cla()
#     print("endloop")
#
#
#     # # smdiffres = windowsmoother(smdiffres, time, nsmooths=1, endpt_percent=0.04, smoothspan=0.06)
#     # # smdiffres = windowsmoother(smdiffres, time, nsmooths=1, endpt_percent=0.04, smoothspan=0.1)
#
#



#################################
# PLOTTING THE RAW DATA FILES
################################
#
# import plotly.plotly as py
# import plotly.graph_objs as go
# import plotly.tools as tls
# tls.set_credentials_file(username="sdsawtelle", api_key="d8ng7f2nah")

datadir = snp.getBoxroot()+os.path.normpath("/MolT Project/")
filenames = snp.txtfilenames(datadir)


ax, fig = snp.newfig("lr")
snp.labs(x="Time (ms)", y="$R_j$ ($/Omega$)", t="All Resistance Traces")

# axres, figres = snp.newfig("ur")
# plt.ylabel("$R_{initial}$ ($/Omega$)")
# plt.xlabel("device")
# plt.title("All Starting Resistances")


resistances = []
for filename in filenames:
    datafile = datadir + filename
    expdata = snp.captureheads(datafile, 1)  # Capture header for future reference
    df = pd.read_csv(datafile, header=1, sep=",")  # header is the num lines to skip, after that first line is col names
    
    # # Lets assume always the same ordering of columns, but not take the exact names for granted
    # oldnames = df.columns.tolist()
    # newnames = ["time", "volt", "curr", "res"]
    # namedict = dict(pair for pair in zip(oldnames, newnames))
    # df.rename(columns=namedict, inplace=True)  # inplace modifies this DF
    
    
    # Lets assume always the same ordering of columns, but not take the exact names for granted
    oldnames = df.columns.tolist()
    newnames = ["time", "volt", "curr", "res"]
    namedict = dict(pair for pair in zip(oldnames, newnames))
    df.rename(columns=namedict, inplace=True)  # inplace modifies this DF
    res = df.res.values - df.res.values[0]
    resistances.append(df.res.values[0])
    time = df.time.values - df.time.values[-1]
    curr = df.curr.values
    power = res*curr**2
    col = "blue" if "8-4-a" in filename else "red"
    ax.plot(time, res, label=filename, marker=None)
    # axres.scatter(df.res.values[0], 1, color=col)



#########################
## PLOTTING NW YIELD RESISTANCES
########################
from itertools import chain
import math

datadir = "C:/Users/Sonya/Documents/My Box Files/MolT Project/160625_SDS20_Chip_9-7/AllYields/"
filenames = snp.txtfilenames(datadir)
filenames.sort()

ax, fig = snp.newfig("lr")
snp.labs(x="device x-location", y="device y-location", t="All Starting Resistances")
axres, figres = snp.newfig("ur")
snp.labs(x="device number", y="$R_{initial}$ ($\Omega$)", t="All Starting Resistances")

ndevbottom = 17  # To calculate where "device #1" should fall on the circle

resistances = []
for filename in filenames:
    datafile = datadir + filename
    print(filename)
    expdata = snp.captureheads(datafile, 1)  # Capture header for future reference
    df = pd.read_csv(datafile, header=1, sep=",")  # header is the num lines to skip, after that first line is col names

    oldnames = df.columns.tolist()
    newnames = ["devport", "res"]
    namedict = dict(pair for pair in zip(oldnames, newnames))
    df.rename(columns=namedict, inplace=True)  # inplace modifies this DF

    resistances.append(df.res.values[:])


initres = list(chain.from_iterable(resistances))
initres[11] = 650
# For plotting in a circle
deltatheta = 360/len(initres)
angles = [-((ndevbottom-1)/2)*deltatheta-90+deltatheta*index for index, value in enumerate(initres)]
radians = np.array(angles)*(2*math.pi/360)
xs = np.cos(radians)
ys = np.sin(radians)
initres = np.array(initres)

# Create masks for plotting DOA (dead on arrival) and non-DOA devices
mask_doa = (initres >= 10000)
mask_alive = (initres < 10000)

# markers = 6*["s"]+8*["x"]+6*["v"]+8*["o"]

ax.scatter(xs[mask_alive], ys[mask_alive], c=np.log(initres[mask_alive]), cmap="gray", s=100)
ax.scatter(xs[mask_doa], ys[mask_doa], color="red", s=50, marker="x")

res_alive = initres[mask_alive]
index = list(range(0,len(res_alive)))
axres.scatter(index, res_alive)


## To try and plot different marker types... don't know how to make that work with grayscale cmap
# m = np.array(markers)
# unique_markers = set(m)  # or yo can use: np.unique(m)
# for um in unique_markers:
#     mask = m == um
#     # mask is now an array of booleans that van be used for indexing
#     plt.scatter(xs[mask], ys[mask], marker=um)
#

for label, angle, x, y in zip(initres, radians, xs, ys):
    ax.annotate(
        str("%.0f" % label),
        fontsize=10,
        xy = (x, y), xytext = (angle, 1.6),
        textcoords = 'polar',
        ha = 'center', va = 'center',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
    )



# ########################
# # TRYING TO USE PLOTLY
# # ########################
# import plotly.plotly as py
# import plotly.graph_objs as go
#
# trace1 = go.Scatter(
#     x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
#     y=[0, 3, 6, 4, 5, 2, 3, 5, 4]
# )
# trace2 = go.Scatter(
#     x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
#     y=[0, 4, 7, 8, 3, 6, 3, 3, 4]
# )
# data = [trace1, trace2]
# layout = go.Layout(
#     showlegend=True,
#     legend=dict(
#         x=100,
#         y=1
#     )
# )
# fig = go.Figure(data=data, layout=layout)
# plot_url = py.plot(fig, filename='legend-outside')
#
#



# ########################
# # PLOTTING CHIP 4-7 YIELD DATA (NO TIMESTAMPS)
# # ########################
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline
import snips as snp  # my useful snippets module
# Change default aesthetics for matplotlib and turn on interactive backend
snp.prettyplot(mpl)
mpl.interactive(True)


datadir = os.path.normpath(snp.getboxroot()+"MolT Project/160714_SDS20_Chip_4-7/Yields/Initial_and_Final_Yields")+"//"
filenames = snp.txtfilenames(datadir)
filenames.sort()


def getyields(nametype=""):
    for fname in filenames:
        if nametype in fname:
            datafile = datadir + fname
            yieldname = fname.replace(".txt", "")
            df = pd.read_csv(datafile, header=2, sep=",")  # header is the num lines to skip, then first line is col names
            df.columns = ["port", yieldname]
            try:  # Exception when object not instantiated (i.e. on first iteration of loop)
                dfcombined
            except NameError:
                dfcombined = df
            else:
                dfcombined[yieldname] = df[yieldname]  # Assume each yield file has same measurements in the same order
    dfcombined.set_index(keys="port", drop="port", inplace=True)
    return dfcombined

df_ft1 = getyields("FT1")
df_ft2 = getyields("FT2")
df_ft3 = getyields("FT3")

ax, fig = snp.newfig("lr")
snp.labs(x="Time (Arbitrary)", y="$R (\Omega$)", t="Resistance Over Time")


# Use this function to get names of DFs as strings and use that for labeling lines on the plot
def name_of_object(arg):
    for name, value in globals().items():
        if value is arg and not name.startswith('_'):
            return name

def plotrows(df):
    # iterrows is a generator which yield both index and row
    for index, row in df.iterrows():
        rowarr = np.array(row)
        # Check whether it eventually "died" or not, and set marker/color accordingly
        if abs(rowarr[-1]) > 5000:
            mark = "x"
            col = "red"
        else:
            mark = "o"
            col = "blue"
        # Check whether it's starting resistance was "normal", if not don't plot it
        if abs(rowarr[0]) < 1550:
            rowarr = rowarr - rowarr[0]
            # Convert all "dead" resistances (> 10 kohm) to the value 3000
            rowarr[abs(rowarr) >= 5000] = 3000
            mask_alive = (abs(rowarr) <= 5000)  # Only plot up to the point of "death" to avoid huge R values in plot
            linelabel = name_of_object(df) + "_" + str(index)
            ax.plot(rowarr, label=linelabel, marker=mark, color=col, markersize=6, linestyle=alltargets[linelabel])




plotrows(df_ft1)
plotrows(df_ft2)
plotrows(df_ft3)

line = [line for line in ax.lines if line.get_label() == "df_ft3_911"][0]
ax.lines.remove(line)

# ########################
# # FIGURING OUT WHAT THE TARGET EXIT RES WERE FOR CHIP 4-7 DEVICES
# # ########################
datadir = os.path.normpath(snp.getboxroot()+"MolT Project/160714_SDS20_Chip_4-7/EM")+"//"
filenames = snp.txtfilenames(datadir)
filenames.sort()

alltargets = {}
for fname in filenames:
    datafile = datadir + fname
    head = snp.captureheads(datafile,1)
    items = head[0].split(" ; ")
    target = [item for item in items if "TARGET RESISTANCE (Ohms)" in item]
    namechunks = fname.replace(".txt", "").split("_")
    if namechunks[2][0] == "0":
        namechunks[2] = namechunks[2][1:]
    if namechunks[1] == "FT1":
        namechunks[1] = "df_ft1_"
    elif namechunks[1] == "FT2":
        namechunks[1] = "df_ft2_"
    else:
        namechunks[1] = "df_ft3_"

    name = namechunks[1] + namechunks[2]
    if "200" in target[0]:
        alltargets[name] = "dashed"
    else:
        alltargets[name] = "solid"
    # alltargets[name] = target

print("debug")