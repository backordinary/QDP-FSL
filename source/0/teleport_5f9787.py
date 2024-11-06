# https://github.com/ArdaGurcan/quantum-teleportation/blob/ba09ef1d0f9691661c80dd2453a649a810b1bd78/teleport.py
from matplotlib import pyplot as plt
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, IBMQ, transpile
from qiskit.providers.ibmq import least_busy
from qiskit.quantum_info import random_statevector, partial_trace
from qiskit.visualization import plot_bloch_multivector


qreg_anne = QuantumRegister(2, 'anne') # entangler
qreg_bob = QuantumRegister(2, 'bob') # sender
creg_bob = ClassicalRegister(2, 'c') # classical channel
qreg_clara = QuantumRegister(1, 'clara') # receiver
circuit = QuantumCircuit(qreg_anne, qreg_bob, qreg_clara, creg_bob)

# initialize info_qubit with random value and send it to bob
info_qubit = random_statevector(2)

plot_bloch_multivector(info_qubit)
plt.savefig("initial_qubit.svg")

circuit.initialize(info_qubit, [qreg_bob[0]])
circuit.barrier()

# create ERP pair
circuit.h(qreg_anne[0]) 
circuit.cx(qreg_anne[0], qreg_anne[1])
circuit.barrier()

# send ERP pair to bob and clara
circuit.swap(qreg_anne[0], qreg_bob[1])
circuit.swap(qreg_anne[1], qreg_clara[0])

# measure Bell state qubit and info qubit
circuit.cx(qreg_bob[0], qreg_bob[1])
circuit.h(qreg_bob[0]) 
circuit.barrier()

# send results through classical channel
circuit.measure(qreg_bob[1], creg_bob[1]) 
circuit.measure(qreg_bob[0], creg_bob[0])
circuit.barrier()

# apply gates if the depending on registers
circuit.x(qreg_clara[0]).c_if(creg_bob[1], 1)
circuit.z(qreg_clara[0]).c_if(creg_bob[0], 1)

# circuit.cx(qreg_clara[0], qreg_bob[1])
# circuit.cz(qreg_clara[0], qreg_bob[0])

circuit.draw()
plt.savefig("circuit.svg")

# run simulation
aer_sim = Aer.get_backend('aer_simulator')
circuit.save_statevector()
result = aer_sim.run(circuit).result()
counts = result.get_statevector()

### real functionality not supported yet
# # run circuit on actual quantum computer
# IBMQ.load_account()

# # get the least busy backend
# provider = IBMQ.get_provider(hub='ibm-q')
# backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 5 and not x.configuration().simulator and x.status().operational))
# print("Running on least busy backend:", backend)

# # run circuit
# transpiled_circuit = transpile(circuit, backend, optimization_level=3)
# job = backend.run(transpiled_circuit)
# result = job.result()
# counts = result.get_statevector()

plot_bloch_multivector(partial_trace(counts, [0,1,2,3]))
plt.savefig("final_qubit.svg")

plt.show()