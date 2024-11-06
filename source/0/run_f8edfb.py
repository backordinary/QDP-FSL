# https://github.com/Robinbux/AI-Projects/blob/edde9e02ad21263ad21ad0d3bff64e78c556d587/quantum_nn/run.py
# Custom Libraries
from qml import QMLQuantumCircuit, NeuralNet, CircuitParams, generate_thetas, QMLWrapper
from dataset_util import TorchDataset, load_dataset

# Numpy
import numpy as np
from numpy import pi

#Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

#Qiskit
from qiskit import Aer, IBMQ, QuantumRegister, ClassicalRegister, QuantumCircuit

# Other
from matplotlib import pyplot as plt
import logging
from datetime import datetime

#IBMQ.save_account('f7d9e28527aac1a83634450f117d3af7d02bfcea1c995bad5a6bcde3a69baa92bac9148ed73db91349b8c50c07390ac8bbabcdac9066f56e34b169c177f064c6', overwrite=True)
#IBMQ.enable_account("f7d9e28527aac1a83634450f117d3af7d02bfcea1c995bad5a6bcde3a69baa92bac9148ed73db91349b8c50c07390ac8bbabcdac9066f56e34b169c177f064c6")
#provider = IBMQ.get_provider(hub='ibm-q-fraunhofer', group='fhg-all', project='ticket')
#print(f"Backends: {provider.backends()}")
#print(f"Backends2: {provider2.backends()}")
#backend = provider.get_backend('ibmq_rome')

backend = Aer.get_backend('qasm_simulator')

PLOT_RESULTS = False

NUM_QUBITS = 1
NUM_THETAS = 2
NUM_SHOTS = 100
SHIFT= np.pi/4
LEARNING_RATE = 0.001
EPOCHS = 2
BACKEND = backend
DATASET = TorchDataset.MNIST

DATASET_NAME = DATASET.name

# Run on Real QPU
# provider = IBMQ.load_account()
# backend_name = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits == NUM_QUBITS and not x.configuration().simulator and x.status().operational == True))
# print(f"Least busy backend: {backend_name}")
# BACKEND = provider.get_backend(backend_name)

DATA_PATH = '/Users/robinbux/Desktop/quantum-machine-learning/qiskit-pytorch/library/data'

circuit_params = CircuitParams(
    num_qubits    = NUM_QUBITS,
    num_thetas    = NUM_THETAS,
    num_shots     = NUM_SHOTS,
    shift         = SHIFT,
    learning_rate = LEARNING_RATE,
    backend       = BACKEND,
    epochs        = EPOCHS,
    dataset       = DATASET
)


class CustomCircuit(QMLQuantumCircuit):
    def __init__old(self, qubits, rotations_per_qubit, circuit_params):
        self.thetas = generate_thetas(qubits * rotations_per_qubit)
        
        qreg_q = QuantumRegister(qubits, 'q')
        creg_c = ClassicalRegister(qubits, 'c')
        self.circuit = QuantumCircuit(qreg_q, creg_c)
        
        
        
    
    def __init__(self, circuit_params):
        self.thetas = generate_thetas(circuit_params.num_thetas)
        
        qreg_q = QuantumRegister(1, 'q')
        creg_c = ClassicalRegister(1, 'c')
        self.circuit = QuantumCircuit(qreg_q, creg_c)

        self.circuit.ry(self.thetas[0], qreg_q[0]) # First Theta
        self.circuit.rz(self.thetas[1], qreg_q[0]) # First Theta

        self.circuit.measure(qreg_q[0], creg_c[0])
        
        qreg_q = QuantumRegister(3, 'q')
        creg_c = ClassicalRegister(3, 'c')
        self.circuit = QuantumCircuit(qreg_q, creg_c)
        
        self.circuit.ry(self.thetas[0], qreg_q[0])
        self.circuit.ry(self.thetas[1], qreg_q[1])
        self.circuit.ry(self.thetas[2], qreg_q[2])
        self.circuit.rz(self.thetas[3], qreg_q[0])
        self.circuit.rz(self.thetas[4], qreg_q[1])
        self.circuit.rz(self.thetas[5], qreg_q[2])
        self.circuit.cx(qreg_q[0], qreg_q[1])
        self.circuit.cx(qreg_q[0], qreg_q[2])
        self.circuit.cx(qreg_q[1], qreg_q[2])
        self.circuit.ry(self.thetas[6], qreg_q[0])
        self.circuit.ry(self.thetas[7], qreg_q[1])
        self.circuit.ry(self.thetas[8], qreg_q[2])
        self.circuit.rz(self.thetas[9], qreg_q[0])
        self.circuit.rz(self.thetas[10], qreg_q[1])
        self.circuit.rz(self.thetas[11], qreg_q[2])
        self.circuit.cx(qreg_q[0], qreg_q[1])
        self.circuit.cx(qreg_q[0], qreg_q[2])
        self.circuit.cx(qreg_q[1], qreg_q[2])
        self.circuit.ry(self.thetas[12], qreg_q[0])
        self.circuit.ry(self.thetas[13], qreg_q[1])
        self.circuit.ry(self.thetas[14], qreg_q[2])
        self.circuit.rz(self.thetas[15], qreg_q[0])
        self.circuit.rz(self.thetas[16], qreg_q[1])
        self.circuit.rz(self.thetas[17], qreg_q[2])
        self.circuit.measure(qreg_q[0], creg_c[0])
        self.circuit.measure(qreg_q[1], creg_c[1])
        self.circuit.measure(qreg_q[2], creg_c[2])
        
        super(CustomCircuit, self).__init__(self, circuit_params)

circuit = CustomCircuit(circuit_params)

loader = load_dataset(DATASET, 2, DATA_PATH)

qml_wrapper = QMLWrapper(circuit, loader.train_loader, loader.test_loader, plot_results = PLOT_RESULTS)

qml_wrapper.train()
qml_wrapper.plot_loss()
qml_wrapper.get_accuracy()
qml_wrapper.save_circuit()
qml_wrapper.plot_sample_predictions()
