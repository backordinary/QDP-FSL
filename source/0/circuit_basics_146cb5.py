# https://github.com/kosycoder/qiskit/blob/910fde68802a94f3745b7f534ce5c46fd725390b/circuit_basics.py
import numpy as np
from qiskit import QuantumCircuit
# Create a Quantum Circuit acting on a quantum register of three qubits
circ = QuantumCircuit(3)
# Add a H gate on qubit 0, putting this qubit in superposition.
circ.h(0)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
circ.cx(0, 1)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 2, putting
# the qubits in a GHZ state.
circ.cx(0, 2)
circ.draw('mpl')

from qiskit.quantum_info import Statevector
# Set the intial state of the simulator to the ground state using from_int
state = Statevector.from_int(0, 2**3)
# Evolve the state by the quantum circuit
state = state.evolve(circ)
#draw using latex
state.draw('latex')
from qiskit.visualization import array_to_latex
#Alternative way of representing in latex
array_to_latex(state)
state.draw('qsphere')
state.draw('hinton')

from qiskit.quantum_info import Operator
U = Operator(circ)
# Show the results
U.data

# Create a Quantum Circuit
meas = QuantumCircuit(3, 3)
meas.barrier(range(3))
# map the quantum measurement to the classical bits
meas.measure(range(3), range(3))
# The Qiskit circuit object supports composition.
# Here the meas has to be first and front=True (putting it before)
# as compose must put a smaller circuit into a larger one.
qc = meas.compose(circ, range(3), front=True)
#drawing the circuit
qc.draw('mpl')

# Adding the transpiler to reduce the circuit to QASM instructions
# supported by the backend
from qiskit import transpile

# Use Aer's qasm_simulator
from qiskit.providers.aer import QasmSimulator
backend = QasmSimulator()
# First we have to transpile the quantum circuit
# to the low-level QASM instructions used by the
# backend
qc_compiled = transpile(qc, backend)
# Execute the circuit on the qasm simulator.
# We've set the number of repeats of the circuit
# to be 1024, which is the default.
job_sim = backend.run(qc_compiled, shots=1024)
# Grab the results from the job.
result_sim = job_sim.result()
counts = result_sim.get_counts(qc_compiled)
print(counts)
from qiskit.visualization import plot_histogram
plot_histogram(counts)
