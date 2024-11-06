# https://github.com/imaybeniki/qiskit/blob/6bf43d9f986f0fb4387291af4a5d5f52ec64213b/Playground/qiskit_playground.py
from qiskit import QuantumProgram

# Create a QuantumProgram object instance.
qp = QuantumProgram()

# Create a Quantum Register called "qr" with 2 qubits.
qr = qp.create_quantum_register('qr',2)

# Create a Classical Register called "cr" with 2 bits.
cr = qp.create_classical_register('cr',2)

# Create a Quantum Circuit called "qc" involving qr and cr.
qc = qp.create_circuit('HelloFratBoiCircuit', [qr],[cr])

backend = 'ibmqx5'

# This is just a playground, so I am going to just hard code this here
# If you want your own token just copy it from `My Account` https://quantum-computing.ibm.com/
token = '8213e86064371c63ba86d3ea3624c01f46b885b8e2527439fc73a2365ae6afda5750d6999d3e374efd70750745ce54399134c78096a6dfeabbc78b03d1ef9483'

qp.set_api(token,url='https://quantumexperience.ng.bluemix.net/api')

# Add the H gate in the Qubit 1, putting this qubit in superposition.
qc.h(qr[1])

# Add the CX gate on control qubit 1 and target qubit 0, putting the qubits in a Bell state i.e entanglement
qc.cx(qr[1], qr[0])

# Compile and execute the Quantum Program in the ibmqx5
results = qp.execute(['HelloFratBoiCircuit'] ,backend ,timeout=2400)
print(results.get_counts('HelloFratBoiCircuit'))

# So that we have time to see the output
input('Press ENTER to exit')