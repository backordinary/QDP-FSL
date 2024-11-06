# https://github.com/katepiotrowsk/QCNN/blob/f97165f85e00d7332af8aa19f8a47e034f753464/ConvolutionalQuantumNetwork/QuantumEncoder.py
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.extensions import XGate, UnitaryGate
from unitary_data_encoder import EncodeUnitaryMatrix
import time
import matplotlib.pyplot as plt


class QuantumEncoder:
    def __init__(self):
        self.data = None
        self.states = None
        self.qubits = None
        self.uninit_layers = []

    def __init__(self, quantum_matrix, states):
        # make sure that the quantum_matrix is properly unitary
        # assert np.isclose(np.matmul(quantum_matrix, np.transpose(quantum_matrix)), np.identity(len(quantum_matrix)))
        self.mat = quantum_matrix # quantum data that is already unitary
        self.set_dims() # loads self.qubits with appropriate value
        self.states = states
        self.inp = None
        self.circ = None
        self.c_reg = None
        self.uninit_layers = []
        self.circ = None # circuit
        self.c = 1  # one classical bit
        self.inp = None # array of input qubits
        self.c_reg = None
        self.make_circuit() # initializes the circuit
        self.q_encode() # after setting up, finally, encode


    def set_dims(self):
        """sets dimensions of input qubit amount for the circuit. length(inp) == qubits"""
        dims = len(self.mat) # dimensionality of data
        # added len of random states, plus one ancilla for swap-tets
        in_q = int(np.log2(dims))
        self.qubits = in_q*2 + 1 # Number of qubits = log2(Dimensionality of data) (could be a decimal number)

    def make_circuit(self):
        """creates quantum and classical register  & circuit w/ appropriate qubits"""
        self.inp = QuantumRegister(self.qubits, "in_qbit")
        self.circ = QuantumCircuit(self.inp) #
        self.c_reg = ClassicalRegister(self.c, "reg")
        self.circ.add_register(self.c_reg)

    def RY(self,states):
        """performs RY on all initial states"""
        for i in range(1, len(states) + 1):
            self.circ.ry(states[i-1], self.inp[i])

    def CRY(self,states):
        """performs controlled ry chain on all qubits, creates cycle.
            i.e. cry from qubit 5->1, 1->2, etc."""
        inp = self.inp
        for i in range(1, len(states) + 1):
            if i == len(states): # if last state, conrol first qubit, ignore ancilla
                self.circ.cry(states[i-1], inp[i], inp[1])
                continue
            self.circ.cry(states[i-1], inp[i], inp[i+1])

    def swap_test(self):
        """performs cswap on each"""
        inp = self.inp
        stretch = self.qubits//2
        self.circ.h(inp[0])
        for i in range(1, stretch+1):
            self.circ.cswap(inp[i], inp[i+stretch], inp[0])
        self.circ.h(inp[0])
        self.circ.measure(0,0)

    def addLayer(self,layer_style):
        if layer_style == "RY" or "CRY":
            self.uninit_layers.append(layer_style)
        else:
            raise("Not a valid layer")

    def initializeCircuit(self):
        for index,layer in enumerate(self.uninit_layers):
            if layer == "RY":
                self.RY(self.states[index])
            if layer == "CRY":
                self.CRY(self.states[index])

    def q_encode(self):
        """performs RY, controlled RY and then RY on each state, followed by the swap_test"""
        #print(self.qubits, ' qubits')
        self.addLayer("RY")
        self.addLayer("CRY")
        self.addLayer("RY")
        self.initializeCircuit()
        self.circ.append(UnitaryGate(self.mat, label="Input"), self.inp[5:9])
        self.swap_test()
#       return self.circ, self.qubits

    def update_states(self,new_state):
        assert type(self.states==new_state,"Not the same type of data")
        self.states = new_state

    def get_states(self):
        return self.states


class QuantumEncoderBank():
    def __init__(self,unitary_matrix,states,simulator=None):
        self.unitaries = np.array(unitary_matrix)
        self.circuits = np.empty((unitary_matrix.shape[0],unitary_matrix.shape[1]),dtype=object)
        self.states = states
        self.convert_to_circuits()
        self.probability_matrix = np.zeros((unitary_matrix.shape[0],unitary_matrix.shape[1]))

    def convert_to_circuits(self):
        for i in range(self.unitaries.shape[0]):
            for j in range(self.unitaries.shape[1]):
                self.circuits[i,j] = QuantumEncoder(self.unitaries[i,j],self.states)

    def get_circuits(self):
        return self.circuits

    def induceLayer(self,simulator,counts=7500):
        for i in range(self.circuits.shape[0]):
            for j in range(self.circuits.shape[1]):
                circuit = self.circuits[i,j].circ
                job = execute(circuit, simulator, shots=counts)
                results = job.result().get_counts(circuit)
                try:
                    prob = results['0'] / (results['1'] + results['0'])
                    prob = (prob - 0.5)
                    if prob <= 0.005:
                        prob = 0.005
                    else:
                        prob = prob * 2
                except:
                    prob = 1
                self.probability_matrix[i,j] = prob
        return self.probability_matrix
