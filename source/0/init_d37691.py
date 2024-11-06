# https://github.com/epelaaez/SelfLearningDistributions/blob/8cdee487a2655f38f8cf0fdcc7f7a073f6020f3c/selflearning/__init__.py
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms import FasterAmplitudeEstimation, EstimationProblem
from itertools import product

import numpy as np
import random

from .sampler import Sampler