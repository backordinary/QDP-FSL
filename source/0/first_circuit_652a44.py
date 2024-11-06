# https://github.com/jannikobenhoff/quantum_computation/blob/3e38c8b4bc4b7882f7b7e8985aecbb5d36dec6d5/first_circuit.py
import numpy as np
from qiskit import *
from qiskit.quantum_info import Statevector
from qiskit.visualization import array_to_latex

import matplotlib.pyplot as plt
from pylatexenc import *
from PIL import Image
from helpers import *

if __name__ == "__main__":
    circ = QuantumCircuit(3)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(0, 2)
    #circ.draw('mpl', filename="__output/first_circuit.png")
    print(circ)
    # open_image("__output/first_circuit.png")

    state = Statevector.from_int(0, 2 ** 3)

    # Evolve the state by the quantum circuit
    state = state.evolve(circ)

    # Alternative way of representing in latex
    array_to_latex(state)



