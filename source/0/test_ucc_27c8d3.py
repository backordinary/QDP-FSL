# https://github.com/heineaabo/QuantumAlgorithms/blob/9ef038b5b2b721efada8360e76382b055a1fd38b/quantum_algorithms/ansatz/test_ucc.py
import qiskit as qk
import numpy as np

from vqe import VQE
from ucc import UnitaryCoupledCluster
from qiskit.chemistry.components.variational_forms import UCCSD as qkUCC

import sys
sys.path.append('../../QuantumCircuitOptimizer')
from quantum_circuit import SecondQuantizedHamiltonian as sqh


l = 4     # number of spin orbitals / number of qubits
n = 2     # Number of occupied spin orbitals
delta = 1 # Level spacing
g = 1     # Interaction strength
#g /= 4

# Matrix elements
h_pq = np.identity(l)
for p in range(l):
	h_pq[p,p] *= delta*(p - (p%2))/2
	
h_pqrs = np.zeros((l,l,l,l))
for p in range(0,l-1,2):
	for r in range(0,l-1,2):
		h_pqrs[p,p+1,r,r+1] = -0.5*g


pairing = sqh(n,l)
pairing.set_integrals(h_pq,h_pqrs)
pairing.get_circuit()
circuit_list = pairing.to_circuit_list(ptype='vqe')


my_ucc = UnitaryCoupledCluster(n,l,'D',1)
my_ucc.new_parameters()
qk_ucc = qkUCC(num_qubits=l,
               depth=1,
               num_orbitals=l,
               num_particles=n,
               two_qubit_reduction=False,
               qubit_mapping='jordan_wigner',
               excitation_type='d')

theta = [5.829889373194686]

my_qb = qk.QuantumRegister(l)
my_cb = qk.ClassicalRegister(l)
my_qc = qk.QuantumCircuit(my_qb,my_cb)

qk_qb = qk.QuantumRegister(l)
qk_cb = qk.ClassicalRegister(l)
qk_qc = qk.QuantumCircuit(qk_qb,qk_cb)

my_qc = my_ucc(theta,my_qc,my_qb)
qk_qc = qk_ucc.construct_circuit(theta,qk_qb)


def qk_ucc_func(theta,qc,qb):
    return qk_ucc.construct_circuit(theta,q=qk_qb)



my_vqe = VQE(n_qubits = l,
             n_fermi = n,
             ansatz = my_ucc,
             circuit_list = circuit_list,
             shots = 1000,
             seed=1,
             count_states=True)

qk_vqe = VQE(n_qubits = l,
             n_fermi = n,
             ansatz = qk_ucc_func,
             circuit_list = circuit_list,
             shots = 1000,
             seed=1,
             count_states=True)

my_E = my_vqe.expval(theta)
qk_E = qk_vqe.expval(theta)


print('Circuit depth of UCCD on pairing model with {} particles and {} orbitals'.format(n,l))
print('    My    Qiskit ')
print(' ────────────────')
print(' {:^8}{:^8}'.format(my_E,qk_E))

