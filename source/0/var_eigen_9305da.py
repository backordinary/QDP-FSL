# https://github.com/Aarun2/Quantum_Repo/blob/d1faf997eda4dd9fa3618ee6c0f2f0d8b1c1d5b4/Qiskit_Tutorials/Var_Eigen.py
# Every time we add a particle to the system
# The computational cost grows exponentially
# Computational space doubles with each qubit
# educated guess of wavefunction which is molecule
# vary wavefunction till we get minimum value of ground
# state energy given the hamiltonian
# hamiltonian total energy of the system
# vqe is a hybrid algo where quantum part computes the energy
# classical optimizes variational param

# Calculate interatomic distance of LiH
# distance with lowest energy is the interatomic distance

# Ansatz: Educated guess of the wavefunction
# Mapping: Process of encoding ansatz into qubits of a Qc
# calculate energy with quantum computer and converge it

import numpy as np
import pylab
import copy
from qiskit import BasicAer
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.drivers import PySCFDriver
from qiskit.chemistry.core import Hamiltonian, QubitMappingType

molecule = 'H .0 .0 -{0}; Li .0 .0 {0}' # 3 d distances, curly braces varying
distances = np.arange(0.5, 4.25, 0.25) # vary distance between lithium hydride in the Z dir
# distances to calculate energy at
vqe_energies = [] # ground state by vqe
hf_energies = [] # initial guess
exact_energies = [] # numpy will calculate classical solver

for i,d in enumerate(distances): # compute VQE
    print('step', i)
    
    # experiment setup
    # sto3g how basis will represent orbitals
    driver = PySCFDriver(molecule.format(d/2), basis= 'sto3g')
    qmolecule = driver.run()
    # mapping type parity, two qubit reduction speeds up
    # other options speed up calculations
    operator = Hamiltonian(qubit_mapping=QubitMappingType.PARITY, 
                           two_qubit_reduction=True, freeze_core=True,
                          orbital_reduction=[-3, -2])
    qubit_op, aux_ops = operator.run(qmolecule)
    
    # exact classical result
    exact_result = NumPyMinimumEigensolver(qubit_op, aux_operators=aux_ops).run()
    exact_result = operator.process_algorithm_result(exact_result)

    #VQE
    optimizer = SLSQP(maxiter=1000) # 1000 tries before converging
    # what the molecule looks like, 
    initial_state = HartreeFock(operator.molecule_info['num_orbitals'],
                               operator.molecule_info['num_particles'],
                               qubit_mapping=operator._qubit_mapping,
                               two_qubit_reduction=operator._two_qubit_reduction)
    # variations to find minimum energy
    var_form = UCCSD(num_orbitals = operator.molecule_info['num_orbitals'],
                     num_particles=operator.molecule_info['num_particles'],
                     qubit_mapping=operator._qubit_mapping,
                     two_qubit_reduction=operator._two_qubit_reduction)
    algo = VQE(qubit_op, var_form, optimizer, aux_operators=aux_ops)
    
    vqe_result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
    vqe_result = operator.process_algorithm_result(vqe_result)
    
    exact_energies.append(exact_result.energy)
    vqe_energies.append(vqe_result.energy)
    hf_energies.append(vqe_result.hartree_fock_energy)

pylab.plot(distances, hf_energies, label='Hartree-fock')
pylab.plot(distances, np.array(vqe_energies), 'o',label='VQE')
pylab.plot(distances, exact_energies, 'x',label='Exact')
pylab.xlabel('Interatomic Distance')
pylab.ylabel('Energy')
pylab.title('LiH Ground State Energy')
pylab.legend(loc='upper right')







 
