# https://github.com/KristianWold/Master-Thesis/blob/fd9a65f52793fe7016d40d0b3a6220ed48a12014/src/ansatzes.py
import numpy as np
import qiskit as qk
from copy import deepcopy
from tqdm.notebook import tqdm
from optimizers import Adam, GD
from data_encoders import *
from samplers import *
from utils import *


class Ansatz():
    def __init__(self, blocks=["entangle", "ry"], reps=1):
        self.blocks = blocks
        self.reps = reps

    def __call__(self, circuit, data_register, weight):

        idx_start = idx_end = 0
        for i in range(self.reps):
            for block in self.blocks:
                if block == "rx":
                    idx_end += self.n_qubits
                    for j, w in enumerate(weight[idx_start:idx_end]):
                        circuit.rx(w, data_register[j])
                    idx_start = idx_end

                if block == "ry":
                    idx_end += self.n_qubits
                    for j, w in enumerate(weight[idx_start:idx_end]):
                        circuit.ry(w, data_register[j])
                    idx_start = idx_end

                if block == "rz":
                    idx_end += self.n_qubits
                    for j, w in enumerate(weight[idx_start:idx_end]):
                        circuit.rz(w, data_register[j])
                    idx_start = idx_end

                if block == "entangle":
                    for j in range(self.n_qubits - 1):
                        circuit.cx(data_register[j], data_register[j + 1])

        return circuit

    def calculate_n_weights(self, n_qubits):
        self.n_qubits = n_qubits
        self.n_weights_per_target = 0
        for block in self.blocks:
            if block in ["rx", "ry", "rz"]:
                self.n_weights_per_target += self.reps * self.n_qubits


class Ansatz2():
    def __call__(self, circuit, data_register, weight):
        n_qubits = data_register.size

        for i, w in enumerate(weight[:n_qubits // 2]):
            circuit.cx(data_register[2 * i], data_register[2 * i + 1])
            circuit.rz(w, data_register[i])
            circuit.cx(data_register[2 * i], data_register[2 * i + 1])

        for i, w in enumerate(weight[n_qubits // 2:]):
            circuit.cx(data_register[2 * i + 1], data_register[2 * i + 2])
            circuit.rz(w, data_register[i])
            circuit.cx(data_register[2 * i + 1], data_register[2 * i + 2])

        return circuit
