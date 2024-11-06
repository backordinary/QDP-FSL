# https://github.com/omersajid9/FermiAdsCFT/blob/65d0fd50d6a4301427588b1503797f0d1e18d7b0/utils/import_list.py
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, execute, transpile
# from qiskit.tools.jupyter import *
from qiskit.visualization import *
# from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer import *
from qiskit import *
# Loading your IBM Quantum account(s)
# provider = IBMQ.save_account()


import numpy as np
from IPython.display import Image
import random as rn
import copy
import functools

from qiskit.circuit.library import FourierChecking
from qiskit.quantum_info import *
from qiskit.circuit import *
from qiskit.circuit.library import *
import qiskit.circuit.gate
import qiskit.extensions as ext
import qiskit.opflow as opflow
import qiskit.opflow.expectations as expec
import qiskit.opflow.converters as conver
#from qiskit.QuantumCircuit import save_expectation_value
import qiskit.utils as utils
import qiskit.opflow.primitive_ops.matrix_op as mo
import qiskit.quantum_info as qInfo
import qiskit.providers.ibmq as ibmq
# import tenpy.linalg.random_matrix as tnp
# import pylatexenc
# utils.algorithm_globals.massive = True
import time
from multiprocessing import Process, Value, Manager
from threading import Thread
from qiskit.providers.aer.noise import NoiseModel
import pickle
import shutil
import os

import itertools

from utils.Algorithm import Algorithm
from utils.Simulator import Simulator
from utils.Circuit import Circuit
from utils.HyperbolicStructure import HyperbolicStructure
from utils.Wilson import Wilson