# https://github.com/marcelcech/Thermodynamics-of-quantum-jump-trajectories-on-a-quantum-computer/blob/163dbd82b10d007b108f0100b5cb7e44c152b5dc/code_files/src/package_requirements.py
import numpy as np
import pandas as pd
import qutip as qt
import qiskit
import qiskit.quantum_info
import qiskit.extensions.unitary
import qiskit.visualization
import qiskit.tools
import qiskit.result
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import copy, deepcopy
from typing import Union
from functools import reduce  # forward compatibility
import operator
import h5py

h5str = h5py.special_dtype(vlen=str)
