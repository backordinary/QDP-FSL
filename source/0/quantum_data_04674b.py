# https://github.com/moswil/Qiskit-Hackathon/blob/1b1b1cf553c9c93ec0c88df9b76a47b1e44bc404/quantum_data.py
import os
import pickle

import numpy as np
import math
import matplotlib.pyplot as plt 
from qiskit.visualization import plot_histogram, plot_bloch_multivector

from qiskit.aqua.components.initial_states import Custom
from qiskit import ClassicalRegister, Aer, execute, QuantumCircuit, QuantumRegister

from q_backend import QBitsLoader


class QuantumData:
    def __init__(self, dir_, device=None, online=False):
        self.dir_ = dir_
        self.device = device
        self.length = 0
        self.online = online
    def choose_device(self, name):
        self.device = name
    def transform(self, save=False):
        qbit_loader = QBitsLoader(self.online)
        lower_dimensional_data = qbit_loader.pca_recommendation(self.dir_, self.device)
        quantum_data = []
        for image,feature in zip(os.listdir(self.dir_),lower_dimensional_data):
            vector = np.array(feature)
            self.length = vector.shape[0]
            if int(math.log2(self.length)) != self.length:
                pad_length = 2**(int(math.log2(self.length))+1)
                vector = np.pad(vector, (0, pad_length - self.length), mode='constant')
            self.length = vector.shape[0]

            cr, qbits = self.circuit(vector)

            ## For plotting purposes, uncomment this
        #     print(list(range(qbits)))
        #     cr.add_register(ClassicalRegister(qbits))
        #     cr.measure(list(range(qbits)), list(range(qbits)))

        #     simulator = Aer.get_backend('qasm_simulator')
        #     counts = execute(cr, backend=simulator, shots=1024).result().get_counts()
        #     plot_histogram(counts)
        #     simulator = Aer.get_backend('statevector_simulator')
        #     result = execute(cr, backend = simulator).result()
        #     statevector = result.get_statevector()
        #     plot_bloch_multivector(statevector)
        #     cr.draw(output = 'mpl')
            quantum_data.append(cr)

        #     plt.show()

            # 
            # print(dir(cr))
        if save:
            pickle.dump(quantum_data, open(self.dir_+'_quantum_representation.pkl', 'wb'))
        return quantum_data
        
    def circuit(self, vec):
        qbits =  int(math.log2(self.length))
        cct = Custom(qbits, state_vector=vec)
        return cct.construct_circuit('circuit'), qbits