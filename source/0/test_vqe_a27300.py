# https://github.com/abid1214/mbl-vvqe-bba/blob/dbed18ce592b979687acc6c0e26a096a256dc20f/src/vqe/tests/test_vqe.py
from .context import variational_form as vf, hamiltonian as ham

import numpy as np
import numpy.linalg as nla

from qiskit import Aer
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.aqua.algorithms import VQE


def test_antiferromagnetic_field_5qubits_with_vqe(seed = '999999'):
	#Sets up a simple antiferromagnetic background field and checks if VQE can create the proper pattern
	num_qubits = 5
	H = ham.magnetic_fields([-1, 1, -1, 1, -1]) #With this field the ground state should be ^v^v^
	ansatz = vf.sz_conserved_ansatz(num_qubits, entanglement='full', spindn_cluster = 'random', seed = seed)
	#ansatz.draw('mpl')

	optimizer = SLSQP(maxiter = 30)
	vqe_h5 = VQE(H, ansatz, optimizer)

	backend = Aer.get_backend("statevector_simulator")
	vqe_h5_results = vqe_h5.run(backend)

	assert(np.absolute(vqe_h5_results['eigenvalue'] + 2.5) < 0.01) #Ground state energy should just be -5/2
