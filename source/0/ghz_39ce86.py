# https://github.com/arecibokck/Moonlight/blob/f0c4dd82002d8d79c9391376cb7c1a5be0a634a9/Qiskit/GHZ.py
import qiskit as qk
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from matplotlib import gridspec


# Define the Quantum and Classical Registers
q = qk.QuantumRegister(3)
c = qk.ClassicalRegister(3)

# Build the circuit
qcircuit = qk.QuantumCircuit(q, c)

qcircuit.h(q[0])
qcircuit.h(q[1])
qcircuit.x(q[2])
qcircuit.cx(q[1], q[2])
qcircuit.cx(q[0], q[2])
qcircuit.h(q[0])
qcircuit.h(q[1])
qcircuit.h(q[2])

qcircuit.measure(q, c)

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
diagram.savefig('GHZState.png', dpi = 100)


fig = plt.figure(figsize = (15,5))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

#Show Circuit
a = fig.add_subplot(gs[0])
a.set_title('Quantum Circuit')
a.set_xticks([])
a.set_yticks([])
img = mpimg.imread('./GHZState.png')
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
    plt.text(x = index[i]-0.15 , y = dv[i]+0.005, s = str(dv[i]), size = 11)
plt.show()
fig.savefig('Sim_Result.png')
