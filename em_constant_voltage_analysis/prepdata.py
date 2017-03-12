""" Takes in the pure data file and spits out a data frame ready for smoothing
"""
__author__ = 'Sonya'
__email__ = "sdsawtelle@gmail.com"
__version__ = "3.5"

import snips as snp
import pandas as pd
import numpy as np


def prepdata(datafile):
    expdata = snp.captureheads(datafile, 1)

    # Reading in the data, renaming columns
    df = pd.read_csv(datafile, header=1, sep=",")  # header is the num lines to skip, after that first line is col names

    # lets assume always the same ordering of columns, but not take the exact names for granted
    oldnames = df.columns.tolist()
    newnames = ["time", "v", "i", "r"]
    namedict = dict(pair for pair in zip(oldnames, newnames))
    df.rename(columns=namedict, inplace=True)  # inplace modifies this DF

    # Old Approach, only looking at dwell voltage
    # dwellidx = df.v.idxmax()  # find index where EM is triggered and voltages stops ramping up and is held constant
    # rseries = df.loc[:dwellidx, "r"].mean()  # all points before that point are used to get an estimate of series R
    #
    # # Now that I have an estimate of the series resistance I want to make two new columns in the DF - one will be $R_j =
    # # R - R_L$ and the other will be $P_j = I^2*R_j$.
    # df["rj"] = df.r-rseries
    #
    # # Lets also make an explicit column that is the "time" in arbitrary units. It can be the index minus the first value
    # # of the index so that EM is considered to start at t = 0.
    # # df["time"] = df.index.tolist()
    # # df.time = df.time - df.time.min()  # the minimum of this new time column will be the value of the smallest row index
    #
    # # Since the resistance is spanning an order of magnitude, let's trying looking at log(R).
    # df["logrj"] = np.log10(df.rj)
    #
    # dfcut = df.loc[dwellidx:]  # only keep points after voltage dwelling begins, this is the true EM.
    # return dfcut

    # New Approach:
    return 0