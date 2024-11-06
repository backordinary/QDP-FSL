# https://github.com/paulineollitrault/COBRA/blob/069d869ed73739e4169dcab43d5db82874a5d877/qEOM.py
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.aqua.components.optimizers.cobyla import COBYLA
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit.aqua.algorithms.adaptive.vqe.vqe import VQE
from qiskit.chemistry import FermionicOperator
from qiskit import Aer, IBMQ
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit import execute
from qiskit.aqua import QuantumInstance

import sys
sys.path.append('./EOM_subroutines')
from EOM_subroutines.q_equation_of_motion import QEquationOfMotion 

#from qiskit.chemistry.aqua_extensions.algorithms.q_equation_of_motion.q_equation_of_motion import QEquationOfMotion 
from qiskit.providers.aer import noise
from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.quantum_info import Pauli
import numpy as np
import os
import logging
import copy

qasm_simulator = Aer.get_backend('qasm_simulator')
sv_simulator = Aer.get_backend('statevector_simulator')

def spin_operator(num_qubits):
    pauli_list = []
    for i in range(num_qubits):
        if i < num_qubits:
            c = 1/2
        else:
            c = -1/2
        a_z = np.asarray([0] * i + [1] + [0] * (num_qubits - i - 1), dtype=np.bool)
        a_x = np.asarray([0] * i + [0] + [0] * (num_qubits - i - 1), dtype=np.bool)
        pauli_list.append([c, Pauli(a_z, a_x)])
    op = WeightedPauliOperator(pauli_list)
    return op

def number_operator(num_qubits):
    h1 = np.identity(num_qubits)
    op = FermionicOperator(h1)
    num_op = op.mapping('jordan_wigner')
    return num_op

def exact_eigenstates(hamiltonian, num_particles, num_spin):
    num_qubits = hamiltonian.num_qubits
    exact_eigensolver = ExactEigensolver(hamiltonian, k=1<<num_qubits)
    exact_results = exact_eigensolver.run()
    
    results = [[],[]]
    
    number_op = number_operator(num_qubits)
    spin_op = spin_operator(num_qubits)
    
    for i in range(len(exact_results['eigvals'])):
        particle = round(number_op.evaluate_with_statevector(exact_results['eigvecs'][i])[0],1)
        spin = round(spin_op.evaluate_with_statevector(exact_results['eigvecs'][i])[0],1)
        if particle != num_particles or spin != num_spin:
            continue
        results[0].append(exact_results['eigvals'][i])
        results[1].append(exact_results['eigvecs'][i])
    return results

two_qubit_reduction = False
qubit_mapping = 'jordan_wigner'
distance = 0.75

pyscf_driver = PySCFDriver(atom='H .0 .0 {}; H .0 .0 .0'.format(distance),
                               unit=UnitsType.ANGSTROM, charge=0, spin=0, basis='sto3g')
molecule = pyscf_driver.run()

core = Hamiltonian(transformation=TransformationType.PH,
                   qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                   two_qubit_reduction=two_qubit_reduction,
                   freeze_core=False,
                   orbital_reduction=[])
algo_input = core.run(molecule)
hamiltonian = algo_input[0]

num_orbitals = core.molecule_info['num_orbitals']
num_particles = core.molecule_info['num_particles']

init_state = HartreeFock(hamiltonian.num_qubits, num_orbitals, num_particles,
                             qubit_mapping=qubit_mapping, two_qubit_reduction=two_qubit_reduction)

depth = 1
var_form = UCCSD(hamiltonian.num_qubits, depth, num_orbitals, num_particles, initial_state=init_state,
                 qubit_mapping=qubit_mapping, two_qubit_reduction=two_qubit_reduction)

optimizer = COBYLA(maxiter = 5000)

algo = VQE(hamiltonian, var_form, optimizer)

results = algo.run(sv_simulator)
print(results)
energy = results['energy']
opt_params = results['opt_params']
ground_state = results['min_vector']

eom = QEquationOfMotion(hamiltonian, num_orbitals, num_particles, qubit_mapping=qubit_mapping,
                                    two_qubit_reduction=two_qubit_reduction)

excitation_energies, eom_matrices = eom.calculate_excited_states(ground_state)

print(excitation_energies)

qeom_energies = [energy]
for gap_i in excitation_energies:
    qeom_energies.append(energy+gap_i)

reference = exact_eigenstates(hamiltonian, 2, 0) #returns only the states with 2 electrons and singlet spin state.
exact_energies = []

tmp = 1000
for i in range(len(reference[0])):
    if np.abs(reference[0][i]-tmp)>1e-5:
        exact_energies.append(np.real(reference[0][i]))
        tmp = reference[0][i]
print('State {} -> exact energy={} , qeom energy={}'.format(0, exact_energies[0], qeom_energies[0]))
print('State {} -> exact energy={} , qeom energy={}'.format(1, exact_energies[1], qeom_energies[1]))
print('State {} -> exact energy={} , qeom energy={}'.format(2, exact_energies[2], qeom_energies[2]))
print('State {} -> exact energy={} , qeom energy={}'.format(3, exact_energies[3], qeom_energies[3]))



'''
device = provider_zrl.get_backend('ibmq_boeblingen')
properties = device.properties()
coupling_map = device.configuration().coupling_map
noise_model = noise.device.basic_device_noise_model(properties)
basis_gates = noise_model.basis_gates
shots = 10000

quantum_instance = QuantumInstance(qasm_simulator, shots=shots, basis_gates=basis_gates, 
                                   coupling_map=coupling_map, noise_model=noise_model)

wave_function = var_form.construct_circuit(opt_params)


excitation_energies, eom_matrices = eom.calculate_excited_states(wave_function, quantum_instance = quantum_instance)

qeom_energies_noisy = [energy]
for gap_i in excitation_energies:
    qeom_energies_noisy.append(energy+gap_i)

for i in range(4):
    print('State {} -> exact energy={} , qeom energy={}'.format(i, exact_energies[i], qeom_energies_noisy[i]))
'''
