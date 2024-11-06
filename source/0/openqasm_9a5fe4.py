# https://github.com/hflash/profiling/blob/0498dff8c1901591d4428c3149eb3aadf80ac483/openqasm.py
from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def main():
    filename = './openqasm/adder.qasm'
    circuit = QuantumCircuit.from_qasm_file(filename)
    circuit.draw()
    plt.show()


if __name__ == '__main__':
    main()