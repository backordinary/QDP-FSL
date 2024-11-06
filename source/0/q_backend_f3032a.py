# https://github.com/moswil/Qiskit-Hackathon/blob/1b1b1cf553c9c93ec0c88df9b76a47b1e44bc404/q_backend.py

# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *

import settings

import math
import os

from preprocessing import resize_images, pca


class QBitsLoader():
    
    def __init__(self, online=False):
        self.online = online

        if self.online:
            TOKEN = os.getenv('TOKEN')
            IBMQ.save_account(TOKEN, overwrite=True)
            provider = IBMQ.load_account()
            self.backends = provider.backends()
            self.backends.append(Aer.backends())
        else:
            print('[INFO] Offline mode: only simulators can be accessed')
            self.backends = Aer.backends()

    def get_backends(self):
        """
            Returns a list of tuples with the available backends and their number of qubits
        """
        return [backend.name() for backend in self.backends]

    def get_backends_with_qubits(self):
        """
            Returns a list of tuples with the available backends and their number of qubits
        """
        return [(backend.name(), backend.configuration().n_qubits,) for backend in self.backends]
    
    def find_backend(self, name):
        """
            Finds a backend by its name
        """
        for backend in self.backends:
            if name == backend.name():
                return backend
        raise ValueError(f"Backend '{name}' not found")
    
    def get_backend_qubits(self, name):
        """
            Returns the number of qubits a certain quantum device can handle
        """
        return self.find_backend(name).configuration().n_qubits
    
    def pca_recommendation(self, dir_, name, size=128):
        qubits = self.get_backend_qubits(name)
        FEATURE_LIMIT = 256
        components = min(2**qubits, len(os.listdir(dir_)), FEATURE_LIMIT)

        
        images = resize_images(dir_, size)
        lower_dimensional_data, variance = pca(images, components)
        print('[INFO] Using ' + str(components) + ' number of components for PCA')
        print('[INFO] This preserves ' + str(variance*100) + '% of the original variance in the images')
        return lower_dimensional_data

if __name__ == "__main__":
    q_bits_loader = QBitsLoader(True)
    
    print(q_bits_loader.get_backends())
    print(q_bits_loader.get_backends_with_qubits())
    print(q_bits_loader.call_backend('unitary_simulator'))