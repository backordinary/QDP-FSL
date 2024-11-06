# https://github.com/ankurbhambri/Quantum-Computing/blob/73ad25643b5f806de788d9d19cdc328d237b6b5a/test.py
from numpy import quantile
from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
import json


key_data = json.load(open('key.json'))

access_key = key_data.get('IBM_KEY', None)

# (2, 2) means first register 2 quibits and second 2 clasical bits
circuit = QuantumCircuit(2, 2)

# same thing above written
# quantum_register = QuantumRegister(2)
# classical_register = ClassicalRegister(2)
# circuit = QuantumCircuit(quantum_register, classical_register)

# circuit.draw(output='mpl')

circuit.h(0)
circuit.cx(0, 1)  # 0-> cntrl qubit, 1-> target quibit
circuit.measure([0, 1], [0, 1])
print(circuit.draw())

# for backend in Aer.backends():
#     print(backend.name())
simulator = Aer.get_backend('qasm_simulator')
res = execute(circuit, backend=simulator).result()
print(res.get_counts())

# IBMQ.save_account(access_key)

IBMQ.load_account()

providers = IBMQ.get_provider("ibm-q")

# for backend in providers.backends():
#     print(backend.properties(), backend.name())

quantum_computer = providers.get_backend("ibmq_lima")

job = execute(circuit, backend=quantum_computer)
print(job_monitor(job))
res = job.result()
print(res.get_counts(circuit))
