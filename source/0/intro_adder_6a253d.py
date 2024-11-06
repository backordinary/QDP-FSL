# https://github.com/Stephen-Campbell-UTD/NM_Project_Quantum_Computing/blob/8a45f599c8c6d705d4435e253096ee18b1612c19/Qiskit/tutorials/intro_adder.py
#%%
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram

# %%
#arg is num of qubits 
# all qubits initialized to 0
qc_output = QuantumCircuit(8)
qc_output.measure_all()

# %%
#make quantum circuit plot
qc_output.draw(initial_state=True)
# %%
sim = Aer.get_backend('aer_simulator')
results = sim.run(qc_output).result()
counts = results.get_counts()
# print(counts)

# %%

qc_encode = QuantumCircuit(8)
qc_encode.x(7)
qc_encode.measure_all()
qc_encode.draw(initial_state=True)
# %%
results = sim.run(qc_encode).result()
print(results.get_counts())
# %%
## Using CNOT for half adder (XOR)
# takes # of qubits and optionally # of cbits

# input is q0, q1
# output is   S: q2 ,C: q3,

qc_ha = QuantumCircuit(4,2)

#set the input to 1, 0
qc_ha.x(0)
qc_ha.x(0)

# S = a XOR b
#control -> control, target -> target XOR control
qc_ha.cx(0,2)
qc_ha.cx(1,2)
# C = a AND b
qc_ha.ccx(0,1,3)

# Measurement
qc_ha.measure(3,1) # Carry
qc_ha.measure(2,0) # Sum
qc_ha.draw(initial_state=True)
# %%
#Measure Half Adder wiht input 1 0 
results = sim.run(qc_ha).result()
print(results.get_counts())

# %%

# %%