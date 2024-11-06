# https://github.com/jerry910525/2022_intern/blob/ac3c3f74103015e234fed3141b99b29cec36170c/intern_codes/quantum_computation/basic_encoding.py
import numpy as np
from numpy import pi
# importing Qiskit
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.circuit import parameter
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import matplotlib.pyplot as plt
from qiskit.opflow import Z,I

"""
this code implenment the basic encoding.
"""

def basis_encoding(ds):
    l = []

    for dp  in ds:
        
        qc = QuantumCircuit(len(dp))
        qc.clear()
        for i in range(len(dp)):
            if dp[i] =='1':
                qc.x(i)
        l.append(qc)
    return l
    
if __name__ =='__main__':
    #prepare data
    ds = ['1100','1000']
    qds = basis_encoding(ds)
    print(qds)



    

