# https://github.com/brower/QuLAT/blob/30e53a6220d5b9ad08779ae755c1efce5167fcda/Hiroki/test/trotterization_test.py
'''
Created on Sep 27, 2019

@author: kwibu
'''
import numpy as np
from scipy.linalg import expm, eig
from qiskit import QuantumCircuit, QuantumRegister, BasicAer, execute
from quantum_circuit.trotterization import trotter_circuit
from operators.pauli_hamiltonian import PauliOperator, PauliHamiltonian
import matplotlib.pyplot as plt

T = 4
# 2-site Heisenberg
H = PauliHamiltonian([1., -0.5, -0.5], [{0: "Z", 1: "Z"}, {0: "X"}, {1: "X"}], n_sites=2)
unitaryH = expm(-1j*H.matrix_form()*T)
spectrum, _ = eig(unitaryH)

tr_spectra = []
backend = BasicAer.get_backend('unitary_simulator')
qr = QuantumRegister(2, 'qr')
circ = QuantumCircuit(qr)
circ = trotter_circuit(circ, qr, H, T, 0)
circ.draw(filename='unitary.jpg', output='mpl')
job = execute(circ, backend)
eigval, eigvec = eig(job.result().get_unitary(circ, decimals=3))
tr_spectra.append(eigval)
backend = BasicAer.get_backend('unitary_simulator')
qr = QuantumRegister(2, 'qr')
circ = QuantumCircuit(qr)
circ = trotter_circuit(circ, qr, H, T, 1)
circ.draw(filename='unitary.jpg', output='mpl')
job = execute(circ, backend)
eigval, eigvec = eig(job.result().get_unitary(circ, decimals=3))
tr_spectra.append(eigval)
backend = BasicAer.get_backend('unitary_simulator')
qr = QuantumRegister(2, 'qr')
circ = QuantumCircuit(qr)
circ = trotter_circuit(circ, qr, H, T, 2)
circ.draw(filename='unitary.jpg', output='mpl')
job = execute(circ, backend)
eigval, eigvec = eig(job.result().get_unitary(circ, decimals=3))
tr_spectra.append(eigval)
backend = BasicAer.get_backend('unitary_simulator')
qr = QuantumRegister(2, 'qr')
circ = QuantumCircuit(qr)
circ = trotter_circuit(circ, qr, H, T, 3)
circ.draw(filename='unitary.jpg', output='mpl')
job = execute(circ, backend)
eigval, eigvec = eig(job.result().get_unitary(circ, decimals=3))
tr_spectra.append(eigval)
backend = BasicAer.get_backend('unitary_simulator')
qr = QuantumRegister(2, 'qr')
circ = QuantumCircuit(qr)
circ = trotter_circuit(circ, qr, H, T, 4)
circ.draw(filename='unitary.jpg', output='mpl')
job = execute(circ, backend)
eigval, eigvec = eig(job.result().get_unitary(circ, decimals=3))
tr_spectra.append(eigval)
backend = BasicAer.get_backend('unitary_simulator')
qr = QuantumRegister(2, 'qr')
circ = QuantumCircuit(qr)
circ = trotter_circuit(circ, qr, H, T, 5)
circ.draw(filename='unitary.jpg', output='mpl')
job = execute(circ, backend)
eigval, eigvec = eig(job.result().get_unitary(circ, decimals=3))
tr_spectra.append(eigval)
backend = BasicAer.get_backend('unitary_simulator')
qr = QuantumRegister(2, 'qr')
circ = QuantumCircuit(qr)
circ = trotter_circuit(circ, qr, H, T, 6)
circ.draw(filename='unitary.jpg', output='mpl')
job = execute(circ, backend)
eigval, eigvec = eig(job.result().get_unitary(circ, decimals=3))
tr_spectra.append(eigval)
backend = BasicAer.get_backend('unitary_simulator')
qr = QuantumRegister(2, 'qr')
circ = QuantumCircuit(qr)
circ = trotter_circuit(circ, qr, H, T, 7)
circ.draw(filename='unitary.jpg', output='mpl')
job = execute(circ, backend)
eigval, eigvec = eig(job.result().get_unitary(circ, decimals=3))
tr_spectra.append(eigval)
backend = BasicAer.get_backend('unitary_simulator')
qr = QuantumRegister(2, 'qr')
circ = QuantumCircuit(qr)
circ = trotter_circuit(circ, qr, H, T, 8)
circ.draw(filename='unitary.jpg', output='mpl')
job = execute(circ, backend)
eigval, eigvec = eig(job.result().get_unitary(circ, decimals=3))
tr_spectra.append(eigval)
backend = BasicAer.get_backend('unitary_simulator')
qr = QuantumRegister(2, 'qr')
circ = QuantumCircuit(qr)
circ = trotter_circuit(circ, qr, H, T, 9)
circ.draw(filename='unitary.jpg', output='mpl')
job = execute(circ, backend)
eigval, eigvec = eig(job.result().get_unitary(circ, decimals=3))
tr_spectra.append(eigval)
plt.plot(range(10), tr_spectra, label='Trotterization')
plt.plot(range(10), np.tile(np.reshape(spectrum, [1, -1]), (10, 1)), linestyle='--', label='scipy result')
plt.legend()
plt.xlabel('# Steps')
plt.ylabel('Spectrum')
plt.show()

'''
Created on Sep 27, 2019

@author: kwibu
'''
import numpy as np
from scipy.linalg import expm, eig
from qiskit import QuantumCircuit, QuantumRegister, BasicAer, execute
from quantum_circuit.trotterization import trotter_circuit
from operators.pauli_hamiltonian import PauliOperator, PauliHamiltonian

# 2-site Heisenberg
H = PauliHamiltonian([1., -0.5, -0.5], [{0: "Z", 1: "Z"}, {0: "X"}, {1: "X"}], n_sites=2)
#H = PauliHamiltonian([1.], [{0: "Z", 1: "X"}], n_sites=2)
backend = BasicAer.get_backend('unitary_simulator')
qr = QuantumRegister(2, 'qr')
circ = QuantumCircuit(qr)
circ = trotter_circuit(circ, qr, H, 0.1, 10)
circ.draw(filename='unitary.jpg', output='mpl')
job = execute(circ, backend)
print(expm(-1j*H.matrix_form()*0.1))
print(job.result().get_unitary(circ, decimals=3))
