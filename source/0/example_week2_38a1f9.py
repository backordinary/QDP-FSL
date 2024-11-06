# https://github.com/kaykhos/clifford_project/blob/574255e8e72d4b718ee18d5af8d3c6fc7656d3a5/week2/example_week2.py
# qiskit                0.23.0
# qiskit-aer            0.7.0
# qiskit-aqua           0.8.0
# qiskit-ibmq-provider  0.11.0
# qiskit-ignis          0.5.0
# qiskit-terra          0.16.0
from qiskit.quantum_info import Clifford
from qiskit.aqua.operators import PauliExpectation, CircuitSampler, StateFn
from qiskit.aqua import QuantumInstance
from qiskit import Aer
import numpy as np
import qiskit
import random
from docplex.mp.model import Model
from qiskit.optimization.problems import QuadraticProgram
from qiskit.optimization.converters import QuadraticProgramToIsing
from qiskit import QuantumCircuit
from qiskit.aqua.operators import StateFn

qiskit.__version__
# Prepare the Ising Hamiltonian
n = 3  # number of qubits
a = 1.0
k = 2
t = range(1, n + 1)

mdl = Model()  # build model with docplex
x = [mdl.binary_var() for i in range(n)]
objective = a * (k - mdl.sum(t[i] * x[i] for i in range(n)))**2
mdl.minimize(objective)

qp = QuadraticProgram()  # convert to Qiskit's quadratic program
qp.from_docplex(mdl)

qp2ising = QuadraticProgramToIsing()  # convert to Ising Hamiltonian
H, offset = qp2ising.encode(qp)
op = H.to_pauli_op()
help(H)
print(H)

# Prepare the state
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cy(1, 2)
psi = StateFn(qc)  # wrap it into a statefunction
print(psi)

# Calculate the expectation value for Ising Hamiltonian
print('expectation_value:', psi.adjoint().compose(op).compose(psi).eval().real)

# %% another way of calculating expectation


# define your backend or quantum instance (where you can add settings)
backend = Aer.get_backend('qasm_simulator')
q_instance = QuantumInstance(backend, shots=1024)

# define the state to sample
measurable_expression = StateFn(op, is_measurement=True).compose(psi)

# convert to expectation value
expectation = PauliExpectation().convert(measurable_expression)

# get state sampler (you can also pass the backend directly)
sampler = CircuitSampler(q_instance).convert(expectation)

# evaluate
print('Sampled:', sampler.eval().real)

# %%

print(qc)
print('Hamiltonian of Ising model:')
print(H.print_details())

# %% Randomly sample Clifford circuits and calculate <H> for each one.
# (E.g. only change single qubit gates, not their locations in the circuit)

# import random
# import quimb.tensor as qtn
# import quimb as qu
# %config InlineBackend.figure_formats = ['svg']
#
# cliff_circ = qtn.Circuit(N=4, tags='PSI0')
#
# # initial layer of hadamards
# for i in range(4):
#     cliff_circ.apply_gate('H', i, gate_round=0)
#
# # # 8 rounds of entangling gates
# for r in range(1, 3):
#     #
#     # even pairs
#     for i in range(0, 4, 2):
#         cliff_circ.apply_gate('CNOT', i, i + 1, gate_round=r)
#     #
#     # Y-rotations
#     for i in range(4):
#         cliff_circ.apply_gate('RY', 1.234, i, gate_round=r)
# #
#     # odd pairs
#     for i in range(1, 3, 2):
#         cliff_circ.apply_gate('CZ', i, i + 1, gate_round=r)
#
# # final layer of hadamards
# for i in range(4):
#     cliff_circ.apply_gate('H', i, gate_round=r + 1)


qc = QuantumCircuit(4)

param = random.randint(0, 5)
print(param)
if param == 0:
    qc.id(0)
elif param == 1:
    qc.H(0)
elif param == 2:
    qc.s(0)
elif param == 3:
    qc.x(0)
elif param == 4:
    qc.y(0)
else:
    qc.z(0)

qc.cx(0, 1)
qc.swap(1, 3)

cliff = Clifford(qc)
cliff_circ = cliff.to_circuit()
cliff_sf = StateFn(cliff_circ)
print(cliff_circ)
# help(cliff)
# cliff_op = qiskit.quantum_info.random_clifford(4)
# print(cliff_op)
#
# # cliff_circ = cliff_op.to_circuit()
# cliff_dense= cliff_circ.to_dense()  # representing circuit in the form of ndarray
# cliff_operator = qiskit.aqua.operators.legacy.MatrixOperator(cliff_dense)
# cliff_detail= cliff_operator.print_details()

# cliff_sf = StateFn(cliff_operator)


# Print the Clifford


#
# Prepare the Ising Hamiltonian
n = 4  # number of qubits
a = 1.0
k = 2
t = range(1, n + 1)

mdl = Model()  # build model with docplex
x = [mdl.binary_var() for i in range(n)]
objective = a * (k - mdl.sum(t[i] * x[i] for i in range(n)))**2
mdl.minimize(objective)

qp = QuadraticProgram()  # convert to Qiskit's quadratic program
qp.from_docplex(mdl)

qp2ising = QuadraticProgramToIsing()  # convert to Ising Hamiltonian
H, offset = qp2ising.encode(qp)
op = H.to_pauli_op()

print('expectation_value:', cliff_sf.adjoint().compose(
    op).compose(cliff_sf).eval().real)
