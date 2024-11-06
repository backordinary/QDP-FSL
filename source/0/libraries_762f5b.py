# https://github.com/mahsakarimii/The-power-of-one-clean-qubit-in-supervised-machine-learning./blob/8d1392235e020439c80410afe4e5c6d6e4eb79ae/libraries.py
# Import Qiskit classes
import qiskit
from qiskit import *
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit import BasicAer

# Tomography functions
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
import qiskit.ignis.mitigation.measurement as mc


from qiskit import QuantumCircuit, BasicAer, transpile


# example dataset
from qiskit_machine_learning.datasets import ad_hoc_data
from sklearn_datasets import make_moons
from sklearn_datasets import make_circles



import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, datasets
from sklearn.svm import SVC

import time





