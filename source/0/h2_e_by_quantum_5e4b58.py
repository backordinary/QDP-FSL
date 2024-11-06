# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/h2_lh/h2_E_by_quantum.py
import numpy as np

from qiskit import Aer
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD

from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import ExactEigensolver, VQE
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.chemistry.drivers import PySCFDriver, UnitsType

'''
H_2 Hamilton integral with  1.43 UnitsType.BOHR = 0.7414 A
'''
driver = PySCFDriver(atom='H 0.0 0.0 0.0; H 0.0 0.0 1.41968', unit=UnitsType.BOHR,
                     charge=0, spin=0, basis='sto3g')
molecule = driver.run()

print(molecule.one_body_integrals)
print(molecule.two_body_integrals)

ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)

qubitOp = ferOp.mapping(map_type='parity', threshold=0.00000001)

# Obliczenie klasyczne
exact_eigensolver = ExactEigensolver(qubitOp, k=1)
ret = exact_eigensolver.run()
print('The classically computed energy is: {:.12f}'.format(ret['eigvals'][0].real))

backend = Aer.get_backend('statevector_simulator')

# setup COBYLA optimizer
max_eval = 200
cobyla = COBYLA(maxiter=max_eval)

num_spin_orbitals = 4
num_particles = 2
map_type = 'parity'
qubit_reduction = False
# setup HartreeFock state
HF_state = HartreeFock(qubitOp.num_qubits, num_spin_orbitals, num_particles, map_type, qubit_reduction)


# setup UCCSD variational form
var_form = UCCSD(qubitOp.num_qubits, depth=1,
                   num_orbitals=num_spin_orbitals, num_particles=num_particles,
                   #active_occupied=[0], active_unoccupied=[0, 1],
                   initial_state=HF_state, qubit_mapping=map_type,
                   two_qubit_reduction=qubit_reduction, num_time_slices=1)

# setup VQE
vqe = VQE(qubitOp, var_form, cobyla, 'matrix')
quantum_instance = QuantumInstance(backend=backend)

results = vqe.run(quantum_instance)
print('The q. computed ground state energy is: {:.12f}'.format(results['eigvals'][0]))
#print('The total ground state energy is: {:.12f}'.format(results['eigvals'][0] + energy_shift + nuclear_repulsion_energy))
print("Parameters: {}".format(results['opt_params']))