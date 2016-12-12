"""  This script will make plots of I vs. V and R vs. V from standard EM traces that were acquired with C++ molT master
program.
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


# Move to the place where this data is stored
path = "C:/Users/Sonya/Documents/My Box Files/molT Project/161212_cpp_debug/"
os.chdir(path)
save = False

###############################################################################################
################### PLOT IV ##########################
###############################################################################################
for fname in snp.txtfilenames(path):
    df = pd.read_csv(fname, header=1)
    df.columns = [col.strip(" ") for col in df.columns]  # Some headers have extra spaces

    fig, ax = snp.newfig()
    snp.labs("Voltage (V)", "Current (I)", "EM Trace - %s" %(fname,))
    ax.plot(df["voltage"], df["current"])
    if(save):
        plt.savefig("EM_IV_Trace_%s.png" %(fname,))
plt.close("all")

###############################################################################################
################### PLOT RV ##########################
###############################################################################################
for fname in snp.txtfilenames(path):
    df = pd.read_csv(fname, header=1)
    df.columns = [col.strip(" ") for col in df.columns]  # Some headers have extra spaces
    fig, ax = snp.newfig()
    snp.labs("Voltage (V)", "Resistance ($\Omega$)", "EM Trace - %s" %(fname,))
    ax.plot(df["voltage"], df["resistance"])
    if(save):
        plt.savefig("EM_RV_Trace_%s.png" %(fname,))
plt.close("all")