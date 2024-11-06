# https://github.com/pablogalaviz/qiskit_demo/blob/59e1e6e610c7fed8bd48b64f6485e9195dc3790b/src/1_getting_started_with_qiskit.py
import numpy as np
from qiskit import *
import matplotlib.pyplot as plt
from qiskit.visualization import plot_state_city
from qiskit import Aer
from qiskit.visualization import plot_histogram

# Create a Quantum Circuit acting on a quantum register of three qubits
circ = QuantumCircuit(3)

# %%
# Add a H gate on qubit 0, putting this qubit in superposition.
circ.h(0)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
circ.cx(0, 1)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 2, putting
# the qubits in a GHZ state.
circ.cx(0, 2)

# %%
# Draw teh circuit
circ.draw('mpl')

plt.show()

# %%

# Run the quantum circuit on a statevector simulator backend
backend = Aer.get_backend('statevector_simulator')

# Create a Quantum Program for execution
job = backend.run(circ)

# %%
result = job.result()
output_state = result.get_statevector(circ, decimals=3)

# %%
plot_state_city(output_state)
plt.show()

# %%
# Run the quantum circuit on a unitary simulator backend
backend = Aer.get_backend('unitary_simulator')
job = backend.run(circ)
result = job.result()

# Show the results
print(result.get_unitary(circ, decimals=3))

# %%
# Create a Quantum Circuit
meas = QuantumCircuit(3, 3)
meas.barrier(range(3))
# map the quantum measurement to the classical bits
meas.measure(range(3), range(3))

# The Qiskit circuit object supports composition using
# the addition operator.
circ.add_register(meas.cregs[0])
qc = circ.compose(meas)

# %%
# drawing the circuit
qc.draw('mpl')
plt.show()

# %%
# Use Aer's qasm_simulator
backend_sim = Aer.get_backend('qasm_simulator')

# Execute the circuit on the qasm simulator.
# We've set the number of repeats of the circuit
# to be 1024, which is the default.
job_sim = backend_sim.run(transpile(qc, backend_sim), shots=1024)

# Grab the results from the job.
result_sim = job_sim.result()

# %%
counts = result_sim.get_counts(qc)
print(counts)

# %%
plot_histogram(counts)
plt.show()



