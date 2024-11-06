# https://github.com/MaximBolduc/S8Quantic/blob/93436490c718a6354eb25e74315142d374daa263/AtomsOfComputation.py
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt

sim = Aer.get_backend('aer_simulator')  # this is the simulator we'll use

""" n = 8
n_q = n
n_b = n
qc_output = QuantumCircuit(n_q,n_b)

for j in range(n):
    qc_output.measure(j,j)

qc_output.draw(output='mpl')

sim = Aer.get_backend('aer_simulator')  # this is the simulator we'll use
qobj = assemble(qc_output)  # this turns the circuit into an object our backend can run
result = sim.run(qobj).result()  # we run the experiment and get the result from that experiment
# from the results, we get a dictionary containing the number of times (counts)
# each result appeared
counts = result.get_counts()
# and display it on a histogram
plot_histogram(counts)




qc_encode = QuantumCircuit(n)
qc_encode.x(7)
qc_encode.draw()
qc = qc_encode + qc_output
qc.draw(output='mpl')

qobj = assemble(qc)
counts = sim.run(qobj).result().get_counts()
plot_histogram(counts)

qc_encode = QuantumCircuit(n)
qc_encode.x(1)
qc_encode.x(5)

qc_encode.draw(output='mpl') """

""" #CNOT display
qc_cnot = QuantumCircuit(2)
qc_cnot.cx(0,1)
qc_cnot.draw(output='mpl')

#CNOT try
qc = QuantumCircuit(2,2)
qc.x(0)
qc.cx(0,1)
qc.measure(0,0)
qc.measure(1,1)
qc.draw(output='mpl')

 """


#Half adder
qc_ha = QuantumCircuit(4,2)
# encode inputs in qubits 0 and 1
qc_ha.x(0) # For a=0, remove the this line. For a=1, leave it.
qc_ha.x(1) # For b=0, remove the this line. For b=1, leave it.
qc_ha.barrier()
# use cnots to write the XOR of the inputs on qubit 2
qc_ha.cx(0,2)
qc_ha.cx(1,2)
# use ccx to write the AND of the inputs on qubit 3
qc_ha.ccx(0,1,3)
qc_ha.barrier()
# extract outputs
qc_ha.measure(2,0) # extract XOR value
qc_ha.measure(3,1) # extract AND value

qc_ha.draw(output='mpl')

qobj = assemble(qc_ha)
counts = sim.run(qobj).result().get_counts()
plot_histogram(counts)
print(counts)





plt.show()
