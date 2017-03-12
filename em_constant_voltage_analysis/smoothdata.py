""" Smooths the junction resistance, computes residuals and smooths them.
"""
__author__ = 'Sonya'
__email__ = "sdsawtelle@gmail.com"
__version__ = "3.5"

import pandas as pd
import numpy as np
import statsmodels.api as sm
# lowess(ydata, xdata, (frac of total points to use as smoothing span),...
# (num of iterations to smooth), (return data sorted by x-vals), (ignore points already within delta of neighbors)
# lowess returns a numpy array, so we need to pull out the smoothed y-y.
lowess = sm.nonparametric.lowess  # rename module method for easier access


def smoothdata(df):
    # Apply naive smoothing to data and look at residuals to find the "window" unsuitable for smoothing
    # points, trashdf = findwindow(df.logrj, cutoff=0.05)
    # Recall Series objects are mutable so inside smoothseries it will make a fresh copy of these guys before smoothing
    smdf = df.copy()
    smdf["smlogrj"] = smoothseries(smdf.logrj, 0.05)
    # smdf["smcurr"] = smoothseries(smdf.i, 0.00003)
    return smdf


# modularizing the code, this function will execute window smoothing on a list of y and corresponding x.
def windowsmoothing(y, x, nsmooths, windowstep, spanstep, direction):
    for n in range(nsmooths,-1,-1):  # starts at nsmooths goes to 0
        if direction == "forward":
            endpt = len(y) - windowstep * n
            tempy = y[0:endpt]
            tempx = x[0:endpt]
            smspan = 0.03 + n*spanstep  # starts at 0.058 goes to 0.02
            y[0:endpt] = lowess(tempy, tempx, smspan, 2, is_sorted=False, delta=0.001)[:, 1]
        elif direction == "reverse":
            startpt = windowstep * n
            tempy = y[startpt:-1]
            tempx = x[startpt:-1]
            smspan = 0.03 + n*spanstep  # starts at 0.058 goes to 0.02
            y[startpt:-1] = lowess(tempy, tempx, smspan, 2, is_sorted=False, delta=0.001)[:, 1]
    return y


def findwindow(vals, indx, cutoff=0.015):
    # Apply naive smoothing to data...
    sm_vals = lowess(vals, indx, 0.04, 5, is_sorted=False)[:, 1]

    # calculate residuals between smoothed data and original data
    abs_residuals = abs(vals - sm_vals)

    # smooothing the residuals themselves!
    sm_abs_residuals = lowess(abs_residuals, indx, 0.015, 1, is_sorted=False)[:, 1]

    # create a list that is has boolean y indicating whether inside the not-to-be-smoothed window
    window = sm_abs_residuals > cutoff
    df = pd.DataFrame(data={"window": window}, index=indx)

    # to identify the start / end points of the window do an XOR operation with the window column and a copy of it that
    # is shifted up by one.
    df["shiftwindow"] = df.window.shift(-1)
    df = df[:-1]  # remove last row to get rid of NaN
    df["xorwindow"] = df.shiftwindow != df.window  # This is an XOR operation!

    # make a new data frame that selects only rows where xorwindow is True, then turn it's index into a list
    points = df[df.xorwindow].index.tolist()
    # take the first two points which are the end of the ramp and start of the plateau
    points = points[0:2]
    print("Smoothing will exclude the range from index = ", points[0], " to index = ", points[1])
    return points, df


def smoothseries(ser, cutoff=0.015):
    # Apply naive smoothing to data and look at residuals to find the "window" unsuitable for smoothing
    points, trashdf = findwindow(ser.tolist(), ser.index.tolist(), cutoff)

    # create a new object to hold smoothed values (series are mutable, don't want to change what was passed in!)
    sm_ser = ser.copy()
    
    # smooth a region on the RIGHT of the non-smoothing window
    ramp = sm_ser.loc[:points[0]].tolist()
    ramptime = sm_ser.loc[:points[0]].index.tolist()
    sm_ser.loc[:points[0]] = windowsmoothing(ramp, ramptime, nsmooths=15, windowstep=15, spanstep=0.0025, direction="forward")
    
    # smooth a region on the RIGHT of the non-smoothing window
    plateau = sm_ser.loc[points[1]:].tolist()
    plateautime = sm_ser.loc[points[1]:].index.tolist()
    sm_ser.ix[points[1]:] = windowsmoothing(plateau, plateautime, nsmooths=10, windowstep=5, spanstep=0.02, direction="reverse")
    
    # glue the smoothed left and right to the unsmoothed window
    sm_ser[:] = lowess(sm_ser, sm_ser.index.tolist(), 0.02, 1, is_sorted=False, delta=0.02)[:, 1]

    return sm_ser


