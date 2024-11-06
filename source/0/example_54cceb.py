# https://github.com/MightyGoldenOctopus/QCOMP/blob/5f06fec4bab01be17eade0a7d5455c8ee0cd87bb/Workshop/WP2/givenFile/example.py
import math 
import numpy as np 
from qiskit import QuantumCircuit, execute, Aer

def example():
    circuit = QuantumCircuit(1)
    pass

circuit = example()
sim = Aer.get_backend('statevector_simulator')
job = execute(circuit, sim)

arr = job.result().get_statevector(circuit)

print(circuit.draw())
assert(np.isclose(arr,[1/math.sqrt(2), 1/math.sqrt(2) ]).all())    
circuit.measure_all()
simulator = Aer.get_backend('qasm_simulator')
# Execute the circuit on the qasm simulator
job = execute(circuit, simulator, shots=1000)
# Grab results from the job
result = job.result()
# Returns counts 
counts = result.get_counts(circuit) 
print(counts)

