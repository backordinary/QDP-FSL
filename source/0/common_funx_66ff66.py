# https://github.com/MozammilQ/cross_calib/blob/611f1392cc3a7e3bab46a146b54f1a5c4fc941d0/experiments/common_funx.py
import math
import numpy as np
from qiskit import pulse
from qiskit.circuit import Parameter
import time
from scipy.optimize import curve_fit

KHz=1.0E+3
MHz=1.0E+6
GHz=1.0E+9
us=1.0E-6
ns=1.0E-9
scale_fact=1.0E-14
wait_time=45

def x_16(x):
    return int(x+8)-(int(x+8)%16)

def fit_fnx(x_val, y_val, fnx, init_params):
    fit_params, conv=curve_fit(fnx, x_val, y_val, init_params)
    y_fit=fnx(x_val, *fit_params)
    return fit_params, y_fit


