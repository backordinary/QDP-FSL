# https://github.com/oimichiu/quantumGateModel/blob/fa1cb5ed751edbebe512ee299d5484949c856340/IBMQX/qiskit-tutorials/coduriCareNUcompileaza/tutorial_programming_IBM/localSimulator/1/qft.py
import sys
import os
# Checking the version of PYTHO; we only support > 3.5
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')

# import qiskit from qiskit-sdk-py folder 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'qiskit-sdk-py'))
from qiskit import QuantumProgram
import qiskit.tools.qi as qi
import math

# Define methods for making QFT circuits
def input_state(circ, q, n):
    """n-qubit input state for QFT that produces output 1."""
    for j in range(n):
        circ.h(q[j])
        circ.u1(math.pi/float(2**(j)), q[j]).inverse()

def qft(circ, q, n):
    """n-qubit QFT on q in circ."""
    for j in range(n):
        for k in range(j):
            circ.cu1(math.pi/float(2**(j-k)), q[j], q[k])
        circ.h(q[j])

# Creating the circuit
qp = QuantumProgram()
q = qp.create_quantum_register("q", 3)
c = qp.create_classical_register("c", 3)
qft3 = qp.create_circuit("qft3", [q], [c])
input_state(qft3, q, 3)
qft(qft3, q, 3)
qft3.measure(q[0], c[0])
qft3.measure(q[1], c[1])
qft3.measure(q[2], c[2])
print(qft3.qasm())

# execute circuit on local simulator
result = qp.execute(["qft3"], backend="local_qasm_simulator", shots=1024)
result.get_counts("qft3")
print(result.get_ran_qasm("qft3"))

# Coupling map for local simulator
coupling_map = {0: [1, 2],
                1: [2],
                2: [],
                3: [2, 4],
                4: [2]}

# Place the qubits on a triangle in the bow-tie
initial_layout = {("q", 0): ("q", 2), ("q", 1): ("q", 3), ("q", 2): ("q", 4)}
result2 = qp.execute(["qft3"], backend="local_qasm_simulator", coupling_map=coupling_map, initial_layout=initial_layout)
result2.get_counts("qft3")
print(result2.get_ran_qasm("qft3"))