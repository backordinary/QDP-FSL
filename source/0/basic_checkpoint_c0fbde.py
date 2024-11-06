# https://github.com/IgnacioRiveraGonzalez/aws_qiskit_notebooks/blob/250b1bed38707af39fd5bb83752c68007c6c552c/.ipynb_checkpoints/basic-checkpoint.py
import numpy as np
from qiskit.providers.aer import AerSimulator
from qiskit import transpile, execute, QuantumCircuit
from qiskit.circuit.library import QuantumVolume


circ = QuantumCircuit(8)

def initialize_s(qc, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    for q in qubits:
        qc.h(q)
    return qc
circ = initialize_s(circ, [0,1,2,3,4,5,6,7])

# Add a H gate on qubit 0, putting this qubit in superposition.
#circ.h(0)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
circ.cx(0, 1)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 2, putting
# the qubits in a GHZ state.
circ.cx(0, 2)
circ.cx(0, 3)
circ.cx(0, 4)
circ.cx(0, 5)
circ.cx(0, 6)
circ.cx(0, 7)
#circ.cx(0, 8)
#circ.cx(0, 9)
#circ.cx(0, 10)
#circ.cx(0, 11)

circ.x(0)

circ.mct([0,1,2,3,4,5],6)

circ.z(6)

circ.measure_all()

sim = AerSimulator(method="statevector")

#qc_compiled = transpile(circ, backend)
#job = backend.run(qc_compiled, shots=10, blocking_enable=True, blocking_qubits=1)
#result = job.result()

circ = transpile(circ)

result = execute(circ, sim, shots=14, blocking_enable=True, blocking_qubits=2).result()


counts = result.get_counts(circ)

print(counts)
#print(result)