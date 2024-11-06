# https://github.com/hmy98213/Fault-Simulation/blob/e96bcde84f27f0470c94a6438761063c7e9bc1aa/Circuit_Generators/shor_generator.py
import math
import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Shor
from qiskit import transpile

if __name__ == "__main__":
    N = 15
    a = 7
    backend = Aer.get_backend('aer_simulator')
    quantum_instance = QuantumInstance(backend, shots=1024)
    shor = Shor(quantum_instance=quantum_instance)
    # path = "../Benchmarks/Shor/"
    path = ""
    file_name = "shor_%d_%d.qasm"%(N, a)
    with open(path+file_name, 'w') as f:
        cir = shor.construct_circuit(N, a)
        cir = transpile(cir, basis_gates=['cu', 'u3'])
        f.write(cir.qasm)
        
        