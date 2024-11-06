# https://github.com/Mirkesx/quantum_programming/blob/f7674cf833035a8115442a7f7ad49fef9f4c85ed/Exercises/quantum_circuits.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:50:52 2021

@author: mc
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from qiskit import QuantumRegister, ClassicalRegister, transpile
from qiskit.providers.aer import QasmSimulator
import os
import time


# raise up the quality of the inline plots
dpi = 200
mpl.rcParams['figure.dpi']= dpi
#mpl.rcParams["figure.figsize"] = (32,9)
#plt.rcParams["figure.figsize"] = (32,9)
mpl.rc("savefig", dpi=dpi)

def draw_circuit(qc):
    now = int(time.time())
    filepath = '{}.png'.format(now)
    figure = qc.draw(output="mpl")
    figure.savefig( filepath )
    img = mpimg.imread( filepath )
    plt.axis('off')
    plt.grid(b=None)
    plt.imshow(img)
    os.remove( filepath )
    
# returns a list of qbits from a register
def get_qbits(list_registers):
    list_qbits = []
    for register in list_registers:
        list_qbits.extend([qbit for qbit in register] if type(register) in [QuantumRegister,ClassicalRegister] else [])
    return list_qbits

#percentages
def reformat_counts(counts, n, t=0):
    keys = [key for key in counts.keys()]
    keys.sort()
    new_counts = {
        key: round(counts[key]/n * 100, 2) for key in keys if counts[key]/n > t
    }
    return new_counts

#list of sorted counts (default desc):
def sort_counts(counts, rev=True):
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=rev))
    return sorted_counts
    
    
def print_counts(counts):
    print("COUNTS:")
    for key in counts.keys():
        print("- {}: {}".format(key, counts[key]))


def simulate(qc, shots=1000):
    simulator = QasmSimulator()
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    #qcl.draw_circuit(qc)
    counts = result.get_counts(compiled_circuit)
    #new_counts = qcl.reformat_counts(counts, shots)
    #return new_counts
    return counts
    return int(list(counts.keys())[0])