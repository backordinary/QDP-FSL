# https://github.com/pablogalaviz/qiskit_demo/blob/a87c49ce52695cedbd96a7f899f50b90d7b1813e/src/gates.py
# Useful additional packages
import matplotlib.pyplot as plt
from numpy import pi
from qiskit import Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.quantum_info import Pauli
from qiskit.visualization import plot_histogram, plot_state_qsphere, plot_state_paulivec

# %%

backend = Aer.get_backend('statevector_simulator')

n_bits = 1
quantum_registers = QuantumRegister(n_bits, name='q')
classical_registers = ClassicalRegister(n_bits, name='c')

X_gate = Pauli(label='X' * quantum_registers.size)

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(quantum_registers, classical_registers)
# In Qiskit we give you access to the general unitary using the 𝑢3 gate
#circuit.h(quantum_registers)
circuit.rx(pi / 3, quantum_registers)
# circuit.append(X_gate, quantum_registers)
circuit.measure(quantum_registers, classical_registers)

circuit.draw('mpl')
plt.show()

# %%
job = backend.run(transpile(circuit, backend), shots=1024)
result = job.result()

# %%
count = result.get_counts()
# Plot a histogram
plot_histogram(count)
plt.show()

print("\ncount:\n", count)

# %%

psi = result.get_statevector(circuit)

plot_state_qsphere(psi)

plt.show()

# %%
plot_state_paulivec(psi)
plt.show()
