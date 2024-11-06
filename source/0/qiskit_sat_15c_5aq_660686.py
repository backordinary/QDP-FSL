# https://github.com/DiegoAlonso/sat_qcore/blob/384dbc2638f651c7f952ec8e20dbc6b71eab686b/tests/qiskitCode_gen3/qiskit_sat_15c_5aQ.py
import numpy as np

# Importing standard Qiskit libraries
from qiskit import *
from qiskit.circuit import *
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit.library.standard_gates import XGate,ZGate,HGate 


# Loading your IBM Quantum account(s)
#provider = IBMQ.load_account()

q_reg = QuantumRegister(20, 'q')
c_reg = ClassicalRegister(5, 'c')
circuit = QuantumCircuit(q_reg, c_reg)


# Qslice 0
circuit.append(HGate(),[q_reg[0]])
circuit.append(HGate(),[q_reg[1]])
circuit.append(HGate(),[q_reg[2]])
circuit.append(HGate(),[q_reg[3]])
circuit.append(HGate(),[q_reg[4]])
circuit.barrier(q_reg)
# Qslice 1
circuit.append(XGate(),[q_reg[0]])
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[2]])
# Qslice 2
circuit.append(XGate().control(3),[q_reg[0], q_reg[1], q_reg[2],  q_reg[5]])
# Qslice 3
circuit.append(XGate(),[q_reg[0]])
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[5]])
circuit.barrier(q_reg)
# Qslice 4
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[2]])
# Qslice 5
circuit.append(XGate().control(2),[q_reg[1], q_reg[2],  q_reg[6]])
# Qslice 6
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[6]])
circuit.barrier(q_reg)
# Qslice 7
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[4]])
# Qslice 8
circuit.append(XGate().control(3),[q_reg[2], q_reg[3], q_reg[4],  q_reg[7]])
# Qslice 9
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[4]])
circuit.append(XGate(),[q_reg[7]])
circuit.barrier(q_reg)
# Qslice 10
circuit.append(XGate(),[q_reg[0]])
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[4]])
# Qslice 11
circuit.append(XGate().control(3),[q_reg[0], q_reg[2], q_reg[4],  q_reg[8]])
# Qslice 12
circuit.append(XGate(),[q_reg[0]])
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[4]])
circuit.append(XGate(),[q_reg[8]])
circuit.barrier(q_reg)
# Qslice 13
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[3]])
# Qslice 14
circuit.append(XGate().control(2),[q_reg[1], q_reg[3],  q_reg[9]])
# Qslice 15
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[9]])
circuit.barrier(q_reg)
# Qslice 16
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[4]])
# Qslice 17
circuit.append(XGate().control(5),[q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4],  q_reg[10]])
# Qslice 18
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[4]])
circuit.append(XGate(),[q_reg[10]])
circuit.barrier(q_reg)
# Qslice 19
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[3]])
# Qslice 20
circuit.append(XGate().control(3),[q_reg[1], q_reg[2], q_reg[3],  q_reg[11]])
# Qslice 21
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[11]])
circuit.barrier(q_reg)
# Qslice 22
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
# Qslice 23
circuit.append(XGate().control(3),[q_reg[2], q_reg[3], q_reg[4],  q_reg[12]])
# Qslice 24
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[12]])
circuit.barrier(q_reg)
# Qslice 25
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[4]])
# Qslice 26
circuit.append(XGate().control(4),[q_reg[0], q_reg[2], q_reg[3], q_reg[4],  q_reg[13]])
# Qslice 27
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[4]])
circuit.append(XGate(),[q_reg[13]])
circuit.barrier(q_reg)
# Qslice 28
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[4]])
# Qslice 29
circuit.append(XGate().control(3),[q_reg[1], q_reg[3], q_reg[4],  q_reg[14]])
# Qslice 30
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[4]])
circuit.append(XGate(),[q_reg[14]])
circuit.barrier(q_reg)
# Qslice 31
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[4]])
# Qslice 32
circuit.append(XGate().control(4),[q_reg[0], q_reg[1], q_reg[2], q_reg[4],  q_reg[15]])
# Qslice 33
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[4]])
circuit.append(XGate(),[q_reg[15]])
circuit.barrier(q_reg)
# Qslice 34
circuit.append(XGate(),[q_reg[1]])
# Qslice 35
circuit.append(XGate().control(3),[q_reg[1], q_reg[2], q_reg[3],  q_reg[16]])
# Qslice 36
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[16]])
circuit.barrier(q_reg)
# Qslice 37
circuit.append(XGate(),[q_reg[4]])
# Qslice 38
circuit.append(XGate().control(2),[q_reg[2], q_reg[4],  q_reg[17]])
# Qslice 39
circuit.append(XGate(),[q_reg[4]])
circuit.append(XGate(),[q_reg[17]])
circuit.barrier(q_reg)
# Qslice 40
circuit.append(XGate(),[q_reg[0]])
circuit.append(XGate(),[q_reg[1]])
# Qslice 41
circuit.append(XGate().control(5),[q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4],  q_reg[18]])
# Qslice 42
circuit.append(XGate(),[q_reg[0]])
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[18]])
circuit.barrier(q_reg)
# Qslice 43
circuit.append(XGate(),[q_reg[3]])
# Qslice 44
circuit.append(XGate().control(4),[q_reg[1], q_reg[2], q_reg[3], q_reg[4],  q_reg[19]])
# Qslice 45
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[19]])
circuit.barrier(q_reg)
# Qslice 46
circuit.append(ZGate().control(14),[q_reg[5], q_reg[6], q_reg[7], q_reg[8], q_reg[9], q_reg[10], q_reg[11], q_reg[12], q_reg[13], q_reg[14], q_reg[15], q_reg[16], q_reg[17], q_reg[18],  q_reg[19]])
circuit.barrier(q_reg)
# Qslice 45
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[19]])
# Qslice 44
circuit.append(XGate().control(4),[q_reg[1], q_reg[2], q_reg[3], q_reg[4],  q_reg[19]])
# Qslice 43
circuit.append(XGate(),[q_reg[3]])
circuit.barrier(q_reg)
# Qslice 42
circuit.append(XGate(),[q_reg[0]])
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[18]])
# Qslice 41
circuit.append(XGate().control(5),[q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4],  q_reg[18]])
# Qslice 40
circuit.append(XGate(),[q_reg[0]])
circuit.append(XGate(),[q_reg[1]])
circuit.barrier(q_reg)
# Qslice 39
circuit.append(XGate(),[q_reg[4]])
circuit.append(XGate(),[q_reg[17]])
# Qslice 38
circuit.append(XGate().control(2),[q_reg[2], q_reg[4],  q_reg[17]])
# Qslice 37
circuit.append(XGate(),[q_reg[4]])
circuit.barrier(q_reg)
# Qslice 36
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[16]])
# Qslice 35
circuit.append(XGate().control(3),[q_reg[1], q_reg[2], q_reg[3],  q_reg[16]])
# Qslice 34
circuit.append(XGate(),[q_reg[1]])
circuit.barrier(q_reg)
# Qslice 33
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[4]])
circuit.append(XGate(),[q_reg[15]])
# Qslice 32
circuit.append(XGate().control(4),[q_reg[0], q_reg[1], q_reg[2], q_reg[4],  q_reg[15]])
# Qslice 31
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[4]])
circuit.barrier(q_reg)
# Qslice 30
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[4]])
circuit.append(XGate(),[q_reg[14]])
# Qslice 29
circuit.append(XGate().control(3),[q_reg[1], q_reg[3], q_reg[4],  q_reg[14]])
# Qslice 28
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[4]])
circuit.barrier(q_reg)
# Qslice 27
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[4]])
circuit.append(XGate(),[q_reg[13]])
# Qslice 26
circuit.append(XGate().control(4),[q_reg[0], q_reg[2], q_reg[3], q_reg[4],  q_reg[13]])
# Qslice 25
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[4]])
circuit.barrier(q_reg)
# Qslice 24
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[12]])
# Qslice 23
circuit.append(XGate().control(3),[q_reg[2], q_reg[3], q_reg[4],  q_reg[12]])
# Qslice 22
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.barrier(q_reg)
# Qslice 21
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[11]])
# Qslice 20
circuit.append(XGate().control(3),[q_reg[1], q_reg[2], q_reg[3],  q_reg[11]])
# Qslice 19
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[3]])
circuit.barrier(q_reg)
# Qslice 18
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[4]])
circuit.append(XGate(),[q_reg[10]])
# Qslice 17
circuit.append(XGate().control(5),[q_reg[0], q_reg[1], q_reg[2], q_reg[3], q_reg[4],  q_reg[10]])
# Qslice 16
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[4]])
circuit.barrier(q_reg)
# Qslice 15
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[9]])
# Qslice 14
circuit.append(XGate().control(2),[q_reg[1], q_reg[3],  q_reg[9]])
# Qslice 13
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[3]])
circuit.barrier(q_reg)
# Qslice 12
circuit.append(XGate(),[q_reg[0]])
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[4]])
circuit.append(XGate(),[q_reg[8]])
# Qslice 11
circuit.append(XGate().control(3),[q_reg[0], q_reg[2], q_reg[4],  q_reg[8]])
# Qslice 10
circuit.append(XGate(),[q_reg[0]])
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[4]])
circuit.barrier(q_reg)
# Qslice 9
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[4]])
circuit.append(XGate(),[q_reg[7]])
# Qslice 8
circuit.append(XGate().control(3),[q_reg[2], q_reg[3], q_reg[4],  q_reg[7]])
# Qslice 7
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[4]])
circuit.barrier(q_reg)
# Qslice 6
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[6]])
# Qslice 5
circuit.append(XGate().control(2),[q_reg[1], q_reg[2],  q_reg[6]])
# Qslice 4
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[2]])
circuit.barrier(q_reg)
# Qslice 3
circuit.append(XGate(),[q_reg[0]])
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[5]])
# Qslice 2
circuit.append(XGate().control(3),[q_reg[0], q_reg[1], q_reg[2],  q_reg[5]])
# Qslice 1
circuit.append(XGate(),[q_reg[0]])
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[2]])
circuit.barrier(q_reg)
# Qslice 47
circuit.append(HGate(),[q_reg[0]])
circuit.append(HGate(),[q_reg[1]])
circuit.append(HGate(),[q_reg[2]])
circuit.append(HGate(),[q_reg[3]])
circuit.append(HGate(),[q_reg[4]])
# Qslice 48
circuit.append(XGate(),[q_reg[0]])
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[4]])
# Qslice 49
circuit.append(ZGate().control(4),[q_reg[0], q_reg[1], q_reg[2], q_reg[3],  q_reg[4]])
# Qslice 48
circuit.append(XGate(),[q_reg[0]])
circuit.append(XGate(),[q_reg[1]])
circuit.append(XGate(),[q_reg[2]])
circuit.append(XGate(),[q_reg[3]])
circuit.append(XGate(),[q_reg[4]])
# Qslice 51
circuit.append(HGate(),[q_reg[0]])
circuit.append(HGate(),[q_reg[1]])
circuit.append(HGate(),[q_reg[2]])
circuit.append(HGate(),[q_reg[3]])
circuit.append(HGate(),[q_reg[4]])
circuit.barrier(q_reg)
# Qslice 52
circuit.measure(q_reg[0], c_reg[0])
circuit.measure(q_reg[1], c_reg[1])
circuit.measure(q_reg[2], c_reg[2])
circuit.measure(q_reg[3], c_reg[3])
circuit.measure(q_reg[4], c_reg[4])

circuit.draw('mpl')


from qiskit import Aer
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram

# Use Aer's qasm_simulator
backend_sim = Aer.get_backend('qasm_simulator')

# Execute the circuit on the qasm simulator.
# We've set the number of repeats of the circuit
# to be 1024, which is the default.
job_sim = backend_sim.run(transpile(circuit, backend_sim), shots=1024)

# Grab the results from the job.
result_sim = job_sim.result()
counts = result_sim.get_counts(circuit)
plot_histogram(counts)


