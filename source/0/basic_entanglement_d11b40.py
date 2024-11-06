# https://github.com/sybarits/quantum-computing/blob/6d5c872ed64cc969aa5611e3a291811561f6415c/q-play/basic-entanglement.py
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(2)
qc.h(1)
qc.h(0)
qc.z(0)
# This calculates what the state vector of our qubits would be
# after passing through the circuit 'qc'
ket = Statevector(qc)

# The code below writes down the state vector.
# Since it's the last line in the cell, the cell will display it as output
print(ket)





sim = AerSimulator()  # make new simulator object

# Create quantum circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0,1)  # CNOT controlled by qubit 0 and targeting qubit 1


qc.measure([0,1], [0,1])
print(qc.draw())     # display a drawing of the circuit



job = sim.run(qc)      # run the experiment
result = job.result()  # get the results
# interpret the results as a "counts" dictionary
print("Result: ", result.get_counts())

