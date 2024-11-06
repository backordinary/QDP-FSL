# https://github.com/mauget/qiskit-sdk-py/blob/a863e2df4e59ae6a74962cfb4fb4693b2af37cfc/hello2.py
# Checking the version of PYTHON; we only support > 3.5
import sys

if sys.version_info < (3, 5):
    raise Exception('Please use Python version 3.5 or greater.')

# useful additional packages
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
from pprint import pprint

# importing the QISKit
from qiskit import QuantumProgram
import Qconfig

# import basic plot tools
from qiskit.tools.visualization import plot_histogram