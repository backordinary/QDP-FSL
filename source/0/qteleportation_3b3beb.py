# https://github.com/arecibokck/Moonlight/blob/f0c4dd82002d8d79c9391376cb7c1a5be0a634a9/Qiskit/QTeleportation.py
import qiskit as qk
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from matplotlib import gridspec

# Define the Quantum and Classical Registers
q0 = qk.QuantumRegister(1, 'q0')
q1 = qk.QuantumRegister(1, 'q1')
q2 = qk.QuantumRegister(1, 'q2')
c0 = qk.ClassicalRegister(1, 'c0')
c1 = qk.ClassicalRegister(1, 'c1')
c2 = qk.ClassicalRegister(1, 'c2')

# Build the circuit
qcircuit = qk.QuantumCircuit(q0, q1, q2, c0, c1, c2)

# Prepare an initial state with first qubit using a single unitary
#qcircuit.u1(np.pi, q0) # U1 Gate [[1 0] [0 e^(i*lambda)]]
qcircuit.u2(np.pi*0.5, np.pi*0.5, q0) # U2 Gate [[1 -e^(i*lambda)] [e^(i*phi) e^(i*(phi+lambda))]]

# Prepare a Bell State with the remaining 2 qubits
qcircuit.h(q1)
qcircuit.cx(q1, q2)

# Perform a CNOT between first (control) qubit and second (target) qubit
qcircuit.cx(q0, q1)
# Measure second qubit in the computational basis
qcircuit.measure(q1, c1)

# Measure first qubit in the + - basis
qcircuit.h(q0)
qcircuit.measure(q0, c0)

# Bit Flip Correction with X gate
qcircuit.x(q2).c_if(c1, 1)

# Phase Flip Correction with Z gate
qcircuit.z(q2).c_if(c0, 1)

# Reverse Unitary and measure to check for successful teleportation
#qcircuit.u1(-np.pi, q2) # U1 Gate Adjoint

qcircuit.z(q2) #U2 Gate Adjoint
qcircuit.u2(-np.pi*0.5, -np.pi*0.5, q2)
qcircuit.z(q2)
qcircuit.measure(q2, c2)

# Execute the circuit
backend_sim = qk.BasicAer.get_backend('qasm_simulator')
shots_num = 1000
job = qk.execute(qcircuit,backend_sim,shots=shots_num)
result = job.result()
data = result.get_counts(qcircuit)

# Print the result
print(data)

#Draw and Save the circuit
diagram = qcircuit.draw(output = "mpl")
diagram.savefig('QTeleportationCircuit.png', dpi = 100)


fig = plt.figure(figsize = (15,5))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

#Show Circuit
a = fig.add_subplot(gs[0])
a.set_title('Quantum Circuit')
a.set_xticks([])
a.set_yticks([])
img = mpimg.imread('./QTeleportationCircuit.png')
imgplot = plt.imshow(img)

#Plot Histogram
a = fig.add_subplot(gs[1])
a.set_title('Simulation Result')
#plt.xlabel('States', fontsize=11)
plt.ylabel('Probability', fontsize=11)
dk = list(data.keys())
dv = list(data.values())
dv = [x / shots_num for x in dv]
index = np.arange(len(dk))
plt.xticks(index, dk, fontsize=11, rotation=30)
plt.bar(index, dv)
for i in range(len(dk)):
    plt.text(x = index[i]-0.25 , y = dv[i]+0.001, s = str(dv[i]), size = 11)
plt.show()
