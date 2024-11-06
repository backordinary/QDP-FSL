# https://github.com/nrmanohar/graph_state_generator/blob/ff32ac730908fa4b0c77811dae0acc68b8018793/graph_state_generator/graph_state.py
"""Provide the primary functions."""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
import matplotlib
import matplotlib.pyplot as plt

def grapher(edgelist=[[0,1],[1,2],[2,0]]):
    num=0
    for i in range(len(edgelist)):
        for j in range(len(edgelist[i])):
            if edgelist[i][j]>num:
                num = edgelist[i][j]
    circuit = QuantumCircuit(num+1, num+1)
    for i in range(num+1):
        circuit.h(i)
    for i in range(len(edgelist)):
        circuit.cz(edgelist[i][0],edgelist[i][1])
    print(circuit)

def stabilizer(edgelist=[[0,1],[1,2],[2,0]]):
    num=0
    for i in range(len(edgelist)):
        for j in range(len(edgelist[i])):
            if edgelist[i][j]>num:
                num = edgelist[i][j]
    circuit = QuantumCircuit(2*(num+1), num+1)
    for i in range(num+1):
        circuit.h(i)
    for i in range(len(edgelist)):
        circuit.cz(edgelist[i][0],edgelist[i][1])
    print(circuit)


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    inp = input("Edgelist: ")
    inp=inp.lstrip('[')
    inp=inp.rstrip(']')
    edges = inp.split(';')

    for i in range(len(edges)):
        edges[i] = edges[i].split(',')
    for i in range(len(edges)):
        for j in range(len(edges[i])):
            edges[i][j] = int(edges[i][j])
    print(grapher(edges))