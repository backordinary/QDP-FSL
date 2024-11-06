# https://github.com/MightyGoldenOctopus/QCOMP/blob/5f06fec4bab01be17eade0a7d5455c8ee0cd87bb/Workshop/WP2/givenFile/QFTN.py
import numpy as np
from numpy import pi
# importing Qiskit
from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, execute, Aer

# import basic plot tools
from qiskit.visualization import plot_histogram


def qft(n):
    pass

circuit = qft(4)

#Lets print the circuit for 4 qubits
print("QFT for 4 qubits:\n",circuit.draw())

circuit = qft(5)

#Lets print the circuit for 5 qubits
print("QFT for 5 qubits:",circuit.draw())


#Lets verify the state vector
qftCircuit = circuit.to_gate()
circuitForTest = QuantumCircuit(5)
circuitForTest.x(0)
circuitForTest.x(1)
circuitForTest.x(3)
circuitForTest.append(qftCircuit,[0,1,2,3,4])
sim = Aer.get_backend('statevector_simulator')
job = execute(circuitForTest, sim)
arr = job.result().get_statevector(circuitForTest) 
tab = [ 1.76776695e-01-6.49467042e-17j, -9.82118698e-02+1.46984450e-01j,
 -6.76495125e-02-1.63320371e-01j,  1.73379981e-01+3.44874224e-02j,
 -1.25000000e-01+1.25000000e-01j, -3.44874224e-02-1.73379981e-01j,
  1.63320371e-01+6.76495125e-02j, -1.46984450e-01+9.82118698e-02j,
 -7.57711549e-17-1.76776695e-01j,  1.46984450e-01+9.82118698e-02j,
 -1.63320371e-01+6.76495125e-02j,  3.44874224e-02-1.73379981e-01j,
  1.25000000e-01+1.25000000e-01j, -1.73379981e-01+3.44874224e-02j,
  6.76495125e-02-1.63320371e-01j,  9.82118698e-02+1.46984450e-01j,
 -1.76776695e-01+6.49467042e-17j,  9.82118698e-02-1.46984450e-01j,
  6.76495125e-02+1.63320371e-01j, -1.73379981e-01-3.44874224e-02j,
  1.25000000e-01-1.25000000e-01j,  3.44874224e-02+1.73379981e-01j,
 -1.63320371e-01-6.76495125e-02j,  1.46984450e-01-9.82118698e-02j,
  7.57711549e-17+1.76776695e-01j, -1.46984450e-01-9.82118698e-02j,
  1.63320371e-01-6.76495125e-02j, -3.44874224e-02+1.73379981e-01j,
 -1.25000000e-01-1.25000000e-01j,  1.73379981e-01-3.44874224e-02j,
 -6.76495125e-02+1.63320371e-01j, -9.82118698e-02-1.46984450e-01j] 
if(np.isclose(arr,tab).all()):
    print("""You handle quantum Fourier transfomation
          like a boss""")    


#Lets execute our circuit
circuit.measure_all()
simulator = Aer.get_backend('qasm_simulator')
job = execute(circuit, simulator, shots=1000)
result = job.result()
counts = result.get_counts(circuit) 
#print(counts)