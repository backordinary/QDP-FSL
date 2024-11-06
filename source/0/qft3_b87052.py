# https://github.com/MightyGoldenOctopus/QCOMP/blob/5f06fec4bab01be17eade0a7d5455c8ee0cd87bb/Workshop/WP2/correction/QFT3.py
import math 
import numpy as np 
from numpy import pi
from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info.operators import Operator

def qft():
    circuit = QuantumCircuit(3)
    circuit.h(2)
    circuit.cu1(pi/4,0,2)
    circuit.cu1(pi/2,1,2)
    circuit.h(1)
    circuit.cu1(pi/2,0,1)
    circuit.h(0)
    circuit.swap(0,2)
    return circuit

circuit = qft()

#Lets print the circuit
print(circuit.draw())


#Lets verify the state vector, with |101> as input
qftCircuit = circuit.to_gate()
circuitForTest = QuantumCircuit(3)
circuitForTest.x(0)
circuitForTest.x(1)
circuitForTest.append(qftCircuit,[0,1,2])
sim = Aer.get_backend('statevector_simulator')
job = execute(circuitForTest, sim)
arr = job.result().get_statevector(circuitForTest) 
tab = [ 3.53553391e-01-8.65956056e-17j, -2.50000000e-01+2.50000000e-01j,
 -1.08244507e-16-3.53553391e-01j,  2.50000000e-01+2.50000000e-01j,
 -3.53553391e-01+8.65956056e-17j,  2.50000000e-01-2.50000000e-01j,
  1.08244507e-16+3.53553391e-01j, -2.50000000e-01-2.50000000e-01j]
if(np.isclose(arr,tab).all()):
    print("\nYou are good at computing quantum Fourier transformation!\n")    


#Lets execute our circuit
circuit.measure_all()
simulator = Aer.get_backend('qasm_simulator')
job = execute(circuit, simulator, shots=1000)
result = job.result()
counts = result.get_counts(circuit) 
print(counts)

