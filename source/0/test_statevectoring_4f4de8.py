# https://github.com/KennethGrace/intro_to_qiskit/blob/7d67933c4ed53548df8fa9976b0183b1665199aa/test_StateVectoring.py
# via Qiskit we will asses the affects of certain statevector
# %% import qiskit as well as ancillary modules
import math
from qiskit import *
from qiskit.tools.visualization import plot_bloch_multivector, plot_histogram
# %% simulator settings
sim = 'qasm_simulator'
# %% create a VERY basic quantum circuit
circuit = QuantumCircuit(1, 1)
# %% assign an initial rotation to q0, and draw
circuit.h(0)
# %% measure the value in q0 to c0 if QASM
if 'qasm' in sim:
    circuit.measure(0, 0)
# %% draw the circuit
circuit.draw()
# %% simulate in Aer, because IBMQ cant output state vector qubits
simulator = Aer.get_backend(sim)
result = execute(circuit, backend=simulator, shots=2048).result()
# %% print to a blochsphere of q0
if 'statevector' in sim:
    statevector = result.get_statevector()
    plot_bloch_multivector(statevector)
# %% print to a histogram of q0
if 'qasm' in sim:
    counts = result.get_counts()
    plot_histogram(counts)
# %% print to console the value of counts
if 'qasm' in sim:
    print(counts)


# %%
