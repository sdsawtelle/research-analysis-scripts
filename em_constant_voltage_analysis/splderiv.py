""" Splines and then computes analytical first derivative of the spline and evaluates it to create a new array. Returns the spline tuple
"""

__author__ = 'Sonya'
__email__ = "sdsawtelle@gmail.com"
__version__ = "3.5"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline


def splderiv(ser, smoothing):
    # Series are mutable, create list copies of the values and indexes to work with
    vals = ser.tolist()
    indx = ser.index.tolist()
    # get a spline fit to the values, splfit is a spline object
    splfit = UnivariateSpline(indx, vals, k=5, s=smoothing)  # k is order of polynomial, s is smoothness
    # evaluate the spline object at the x-values (the index values of the series)
    spl = splfit(indx)
    # get the analtical derivative of the spline fit object
    splderivfit = splfit.derivative(n=1)  # n is order of derivative
    # evaluate the spline derivative at the x-values (the index values of the series)
    spl_deriv = splderivfit(indx)
    return spl, spl_deriv
