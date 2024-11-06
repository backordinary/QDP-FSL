# https://github.com/CxrlosKenobi/qxq-qiskit-backups/blob/4f43a502916a89a8aba332106e231bb059aafd33/QCircuits-15.py
#%%
from numpy import pi
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit import (
    QuantumRegister,
    ClassicalRegister,
    QuantumCircuit,
    execute,
    Aer,
    IBMQ
)

#TOKEN = 'x'
#IBMQ.save_account(TOKEN, overwrite=True)

# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

## For questions 1 -4 
qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(3, 'c')
# qc briefly refers to Quantum Circuit
qc = QuantumCircuit(qreg_q, creg_c)

qc.h(qreg_q[0])
qc.h(qreg_q[1])
qc.h(qreg_q[2])
qc.measure(qreg_q[0], creg_c[0])
qc.measure(qreg_q[1], creg_c[1])
qc.measure(qreg_q[2], creg_c[2])

# Execute the circuit on the qasm simulator
job = execute(qc, simulator, shots=8000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(qc)
print("\nTotal count for 00 and 11 are:",counts)

# Draw the circuit
print(qc.draw())

# Plot a histogram
plt.figure(figsize=(10,7))
plt.hist(counts, alpha=.7, color='blue')
plt.ylabel('Measurement probability (%)')
plt.xlabel('Computational basis states')

## For questions 5 - 8
qreg_q = QuantumRegister(2, 'q')
creg_c = ClassicalRegister(2, 'c')
qc2 = QuantumCircuit(qreg_q, creg_c)

qc2.x(qreg_q[0])
qc2.h(qreg_q[0])
qc2.cx(qreg_q[0], qreg_q[1])
qc2.measure(qreg_q[0], creg_c[0])
qc2.measure(qreg_q[1], creg_c[1])

# Execute the circuit on the qasm simulator
job = execute(qc2, simulator, shots=8000)

# Grab results from the job
result = job.result()

# Returns counts
counts2 = result.get_counts(qc2)
print("\nTotal count for 00 and 11 are:", counts2)

# Draw the circuit
print(qc2.draw())

# Plot a histogram
plt.hist(counts2, alpha=.7, color='blue')
plt.title('Quantum simulation')

plt.style.use('ggplot')
plt.show()