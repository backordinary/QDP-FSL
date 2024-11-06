# https://github.com/dfedorov1988/qiskit_dev_examples/blob/29844903d684d8ac0ae8ce8c0988616e36b0ec19/basic_test.py
import numpy as np
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, Molecule
from qiskit.chemistry.transformations import FermionicTransformation, FermionicQubitMappingType
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.chemistry.algorithms.ground_state_solvers.minimum_eigensolver_factories import VQEUCCSDFactory
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.aqua.algorithms import VQE
from qiskit.chemistry.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit.aqua.operators import I, X, Z
from qiskit.chemistry.circuit.library.initial_states import HartreeFock

import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 

# def construct_jastrow_factor(num_qubits : int,
#                              alpha: np.array,
#                              beta: np.array):
#     """Construction of Jastrow operator as in arXiv:2101.09316v1, equation 9"""
    
#     # construct jastrow factor
#     jastrow_factor = (1*I)
#     for i in range(num_qubits-1):
#         jastrow_factor = jastrow_factor^I
    
#     # First term construction
#     for i in range(num_qubits):
#         for k in range(num_qubits):
#             if k == 0:
#                 if i==k:
#                     p_string = -alpha[i] * Z
#                 else:
#                     p_string = -alpha[i] * I
#             else:
#                 if i == k:
#                     p_string = p_string^Z
#                 else:
#                     p_string = p_string^I
#         jastrow_factor += p_string

#     # Second term construction                
#     for i in range(num_qubits):
#         for j in range(i+1, num_qubits):
#             for k in range(num_qubits):
#                 if k == 0:
#                     if i == 0:
#                         p_string = -beta[i, j] * Z
#                     else:
#                         p_string = -beta[i, j] * I
#                 else:
#                     if i == k or j == k:
#                         p_string = p_string^Z
#                     else:
#                         p_string = p_string^I
            
#             jastrow_factor += p_string
#     return jastrow_factor

# print("DEBUG!!!")
molecule = Molecule(geometry=[['Li', [0., 0., 0.]],
                              ['H', [0., 0., 1.5]]],
                     charge=0, multiplicity=1)
driver = PySCFDriver(molecule = molecule, unit=UnitsType.ANGSTROM, basis='sto-3g')
transformation = FermionicTransformation(qubit_mapping=FermionicQubitMappingType.JORDAN_WIGNER, two_qubit_reduction=False)
# print("DEBUG!!")
qubit_ham = transformation.transform(driver)
# numpy_solver = NumPyMinimumEigensolver()
# calc = GroundStateEigensolver(transformation, numpy_solver)
# res = calc.solve(driver)
# print(res)
# print("DEBUG")
num_orbitals = transformation.molecule_info['num_orbitals']
num_particles = transformation.molecule_info['num_particles']
qubit_mapping = transformation.qubit_mapping
two_qubit_reduction = transformation.molecule_info['two_qubit_reduction']
z2_symmetries = transformation.molecule_info['z2_symmetries']

initial_state = HartreeFock(num_orbitals, num_particles, qubit_mapping,
                                    two_qubit_reduction, z2_symmetries.sq_list)

var_form = UCCSD(num_orbitals=num_orbitals,
                 num_particles=num_particles,
                 initial_state=initial_state,
                 qubit_mapping=qubit_mapping,
                 two_qubit_reduction=two_qubit_reduction,
                 z2_symmetries=z2_symmetries)

print("List of single excitations\n", var_form._single_excitations)
print("List of double excitations\n", var_form._double_excitations)
print("Number of orbitals", num_orbitals)
print("Number of electrons", num_particles)
vqe = VQE(var_form=var_form, quantum_instance=QuantumInstance(Aer.get_backend('qasm_simulator')))
calc = GroundStateEigensolver(transformation, vqe)
res = calc.solve(driver)
print(res)

# num_parameters = vqe.var_form.num_parameters
# print("Number of parameters:", num_parameters)

# seed = np.random.seed(43)
# parameters = np.random.rand(num_parameters)

# circuit = vqe.var_form.construct_circuit(parameters)
# print("Parameters", parameters)
# print("Circuit\n", circuit.decompose())

# # This is the qubit Hamiltonian consisting of Pauli strings with coefficients in front
# qubit_ham = calc.transformation.transform(driver)
# num_qubits = qubit_ham[0].num_qubits
# print("Number of qubits is {}".format(num_qubits))
# print("Qubit Hamiltonian")
# print(qubit_ham[0])

# # Constructing Jastrow factor
# alpha = np.random.rand(num_qubits)
# beta = np.random.rand(num_qubits, num_qubits)        
# jastrow_factor = construct_jastrow_factor(num_qubits, alpha, beta)
# print("\njastrow\n", jastrow_factor)
