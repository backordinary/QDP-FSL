# https://github.com/ArdaGurcan/quantum-superdense-coding/blob/f8e7b639e2d08ffdb3865151bf39e834db2c510a/superdense.py
from matplotlib import pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, IBMQ, transpile
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram

qreg_anne = QuantumRegister(2, 'anne') # entangler
qreg_bob = QuantumRegister(1, 'bob') # sender
qreg_clara = QuantumRegister(2, 'clara') # receiver
creg_clara = ClassicalRegister(2, 'received message')
circuit = QuantumCircuit(qreg_anne, qreg_bob, qreg_clara, creg_clara)

# create ERP pair
circuit.h(qreg_anne[0]) 
circuit.cx(qreg_anne[0], qreg_anne[1])
circuit.barrier()

# send one qubit to bob 
circuit.swap(qreg_anne[0], qreg_bob[0])

# send one qubit to clara 
circuit.swap(qreg_anne[1], qreg_clara[0])
circuit.barrier()

# bob's message two bit integer to send to clara
bob_message = '10'

# bob encodes her bits
if bob_message == '00':
    # if message is 0 add identity gate
    circuit.id(qreg_bob[0])

elif bob_message == '01':
    # if message is 1 add x gate
    circuit.x(qreg_bob[0])

elif bob_message == '10':
    # if message is 2 add z gate
    circuit.z(qreg_bob[0])

elif bob_message == '11':
    # if message is 3 add z * x gates 
    circuit.z(qreg_bob[0])
    circuit.x(qreg_bob[0])

circuit.barrier()

# send encoded qubit to clara
circuit.swap(qreg_bob[0], qreg_clara[1])
circuit.barrier()

# clara decodes qubits
circuit.cx(qreg_clara[0], qreg_clara[1])
circuit.h(qreg_clara[0])
circuit.barrier()

# clara measures his qubits
circuit.measure(qreg_clara[0], creg_clara[0])
circuit.measure(qreg_clara[1], creg_clara[1])

circuit.draw('mpl')
plt.savefig("circuit.svg")
## simulate circuit
aer_sim = Aer.get_backend('aer_simulator')
result = aer_sim.run(circuit).result()
counts = result.get_counts(circuit)


## run circuit on actual quantum computer
# IBMQ.load_account()

# # get the least busy backend
# provider = IBMQ.get_provider(hub='ibm-q')
# backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 5 and not x.configuration().simulator and x.status().operational))
# print("Running on least busy backend:", backend)

# # run circuit
# transpiled_circuit = transpile(circuit, backend, optimization_level=3)
# job = backend.run(transpiled_circuit)
# result = job.result()
# print(f"Bob's Message: {max(counts, key=counts.get)}")

plot_histogram(result.get_counts(circuit))
plt.show()