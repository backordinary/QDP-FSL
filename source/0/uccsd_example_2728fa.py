# https://github.com/dfedorov1988/qiskit_dev_examples/blob/29844903d684d8ac0ae8ce8c0988616e36b0ec19/uccsd_example.py
import itertools
import logging
import time
import numpy as np

from qiskit.circuit.library import EfficientSU2
from qiskit import Aer
# from qiskit.validation.base import BaseModel, BaseSchema, ObjSchema, bind_schema, Obj
from qiskit.aqua import QuantumInstance
# from qiskit.aqua.operators import Z2Symmetries, WeightedPauliOperator
from qiskit.aqua.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.aqua.components.optimizers import COBYLA, L_BFGS_B, SLSQP
from qiskit.chemistry.fermionic_operator import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
# from qiskit import IBMQ
# from qiskit.aqua.operators import (OperatorBase, ExpectationBase, ExpectationFactory, StateFn,
#                                    CircuitStateFn, LegacyBaseOperator, ListOp, I, CircuitSampler)
# from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
# from qiskit.providers.aer.noise import NoiseModel
# from qiskit.transpiler import Layout
# from qiskit.circuit import QuantumRegister
# from qiskit.quantum_info import Statevector
# log = logging.getLogger('')
# log.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# using driver to get fermionic Hamiltonian
distance = 1.5
driver = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 {}'.format(distance), unit=UnitsType.ANGSTROM,
                     charge=0, spin=0, basis='sto3g')
molecule = driver.run()
e_nr = molecule.nuclear_repulsion_energy

core = Hamiltonian(transformation=TransformationType.FULL, qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                   two_qubit_reduction=False, freeze_core=False)
qubit_op, aux_ops = core.run(molecule)
print("Originally requires {} qubits".format(qubit_op.num_qubits))

fer_op = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)

print(fer_op.total_angular_momentum())

ee = NumPyMinimumEigensolver(qubit_op)
result = core.process_algorithm_result(ee.run())
exact_e = result['computed_electronic_energy']
print("exact_e", exact_e)
# for pauli in aux_ops[1].paulis:
#     print(pauli[0]*8, pauli[1])

init_state = HartreeFock(num_orbitals=core._molecule_info['num_orbitals'],
                    qubit_mapping=core._qubit_mapping, two_qubit_reduction=core._two_qubit_reduction,
                    num_particles=core._molecule_info['num_particles'])

# var_form = EfficientSU2(num_qubits=4, su2_gates=None, entanglement='full', reps=3, skip_unentangled_qubits=False,
#                         skip_final_rotation_layer=False, parameter_prefix='θ', insert_barriers=False, initial_state=init_state)

var_form = UCCSD(num_orbitals=core._molecule_info['num_orbitals'],
                   num_particles=core._molecule_info['num_particles'],
                   active_occupied=None, active_unoccupied=None, initial_state=init_state,
                   qubit_mapping=core._qubit_mapping, two_qubit_reduction=core._two_qubit_reduction,
                   # Jamming in filtered amplitudes
                   num_time_slices=1,
                   single_exc_op_list=[[1, 2], [7, 8], [0, 2], [6, 8], [1, 5], [7, 11], [0, 5], [6, 11]],
                   double_exc_op_list=[[0, 2, 7, 11], [0, 5, 7, 8], [1, 2, 6, 11], [1, 5, 6, 8], [1, 2, 7, 8], [0, 2, 7, 8], [1, 2, 6, 8],
                   [1, 2, 7, 11], [1, 5, 7, 8], [0, 2, 6, 8], [0, 2, 6, 11], [0, 5, 6, 8], [1, 3, 7, 9], [0, 3, 7, 9], 
                   [1, 3, 6, 9], [0, 3, 6, 9], [1, 4, 7, 10], [0, 4, 7, 10], [1, 4, 6, 10], [0, 4, 6, 10], [1, 5, 7, 11], [0, 5, 7, 11], [1, 5, 6, 11], [0, 5, 6, 11]],
                   triple_exc_op_list=[[0, 2, 7, 8, 7, 11], [0, 2, 7, 11, 7, 8], [0, 5, 7, 8, 7, 8], [1, 2, 6, 8, 7, 11], [1, 2, 6, 11, 7, 8], [1, 5, 6, 8, 7, 8]],
                #    [1, 2, 7, 8, 6, 11], [1, 2, 7, 11, 6, 8], [1, 5, 7, 8, 6, 8], [0, 2, 7, 9, 7, 9], [0, 3, 7, 8, 7, 9], [0, 3, 7, 9, 7, 8], [1, 2, 6, 9, 7, 9],
                #    [1, 3, 6, 8, 7, 9], [1, 3, 6, 9, 7, 8], [1, 2, 7, 9, 6, 9], [1, 3, 7, 8, 6, 9], [1, 3, 7, 9, 6, 8], [0, 3, 7, 9, 7, 11], [0, 3, 7, 11, 7, 9],
                #    [0, 5, 7, 9, 7, 9], [1, 3, 6, 9, 7, 11], [1, 3, 6, 11, 7, 9], [1, 5, 6, 9, 7, 9], [1, 3, 7, 9, 6, 11], [1, 3, 7, 11, 6, 9], [1, 5, 7, 9, 6, 9],
                #     [0, 2, 7, 10, 7, 10], [0, 4, 7, 8, 7, 10], [0, 4, 7, 10, 7, 8], [1, 2, 6, 10, 7, 10], [1, 4, 6, 8, 7, 10], [1, 4, 6, 10, 7, 8], [1, 2, 7, 10, 6, 10], 
                #     [1, 4, 7, 8, 6, 10], [1, 4, 7, 10, 6, 8], [0, 4, 7, 10, 7, 11], [0, 4, 7, 11, 7, 10], [0, 5, 7, 10, 7, 10], [1, 4, 6, 10, 7, 11], [1, 4, 6, 11, 7, 10], 
                #     [1, 5, 6, 10, 7, 10], [1, 4, 7, 10, 6, 11], [1, 4, 7, 11, 6, 10], [1, 5, 7, 10, 6, 10], [0, 2, 7, 11, 7, 11], [0, 5, 7, 8, 7, 11], [0, 5, 7, 11, 7, 8], 
                #     [1, 2, 6, 11, 7, 11], [1, 5, 6, 8, 7, 11], [1, 5, 6, 11, 7, 8], [1, 2, 7, 11, 6, 11], [1, 5, 7, 8, 6, 11], [1, 5, 7, 11, 6, 8]]
                   )
double_exc_op_list=[[0, 2, 7, 11], [0, 5, 7, 8], [1, 2, 6, 11], [1, 5, 6, 8], [1, 2, 7, 8], [0, 2, 7, 8], [1, 2, 6, 8],
                   [1, 2, 7, 11], [1, 5, 7, 8], [0, 2, 6, 8], [0, 2, 6, 11], [0, 5, 6, 8], [1, 3, 7, 9], [0, 3, 7, 9], 
                   [1, 3, 6, 9], [0, 3, 6, 9], [1, 4, 7, 10], [0, 4, 7, 10], [1, 4, 6, 10], [0, 4, 6, 10], [1, 5, 7, 11], [0, 5, 7, 11], [1, 5, 6, 11], [0, 5, 6, 11]]
print("\nh2 coefficients")
for l in double_exc_op_list:
    print(molecule.two_body_integrals[l[0], l[1], l[2], l[3]])
# optimizer = L_BFGS_B()
# optimizer = COBYLA()
optimizer = SLSQP()
vqe = VQE(qubit_op,  var_form=var_form, initial_point=None, optimizer=optimizer) 

print("List of single excitations\n", var_form._single_excitations)
print("List of double excitations\n", var_form._double_excitations)
print("Number of orbitals", core._molecule_info['num_orbitals'])
print("Number of electrons", core._molecule_info['num_particles'])

# backend_type = 'qasm_simulator'
# backend_type = 'statevector_simulator'
# backend = Aer.get_backend(backend_type)
# quantum_instance = QuantumInstance(backend=backend, shots=10000)
# ret = vqe.run(quantum_instance)
# print('Final total energy', ret['optimal_value'] + e_nr)
# print("\nResults:")
# print(ret)
    #  print(print(pauli[0]), pauli[1].to_label())
# for aux_op in aux_ops:
#     print(aux_op)
# print(core.total_angular_momentum)
# for pauli in qubit_op.paulis:
#     print(print(pauli[0]), pauli[1].to_label())
