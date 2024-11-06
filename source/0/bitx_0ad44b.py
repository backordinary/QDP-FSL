# https://github.com/zillerium/shoro/blob/d1d4d7e0959921b800aa48107503cddbf91348b2/bitx.py
# Import the QISKit SDK
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, register

# Set your API Token.
# You can get it from https://quantumexperience.ng.bluemix.net/qx/account,
# looking for "Personal Access Token" section.
QX_TOKEN = "MY_Token"
QX_URL = "https://quantumexperience.ng.bluemix.net/api"

# Authenticate with the IBM Q API in order to use online devices.
# You need the API Token and the QX URL.
register(QX_TOKEN, QX_URL)

# Create a Quantum Register with 2 qubits.
q = QuantumRegister(2)
# Create a Classical Register with 2 bits.
c = ClassicalRegister(2)
# Create a Quantum Circuit
qc = QuantumCircuit(q, c)

# Add a H gate on qubit 0, putting this qubit in superposition.
qc.h(q[0])
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
qc.cx(q[0], q[1])
# Add a Measure gate to see the state.
qc.measure(q, c)

# Compile and run the Quantum Program on a real device backend
job_exp = execute(qc, 'ibmqx4', shots=1024, max_credits=10)
result = job_exp.result()

# Show the results
print(result)
print(result.get_data())
