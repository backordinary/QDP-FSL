# https://github.com/D-J-Harris/QuantumComputing/blob/49e4b2b5197887735373672017385a42900e634f/tutorials/qiskit-vscode-test.py
from qiskit import QuantumCircuit
from qiskit import Aer, execute


qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0,1], [0,1])

qc.draw(output='mpl')

backend = Aer.get_backend('qasm_simulator')
result = execute(qc, backend, shots = 1000).result()
counts = result.get_counts()

print(counts)
