# https://github.com/henrik-dreyer/ParallelBond/blob/0007ce7b902025e1d9829acf27bc2bc0626b6866/test_multichip.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:17:39 2019

@author: henrikdreyer

This function computes the ground state energy for molecular Hydrogen close
to the optimal bond length using parallel VQE on different backends

"""

from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.aqua.components.optimizers import COBYLA
from qiskit import BasicAer
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit.aqua import QuantumInstance

opt_iter=20
dist=0.735
driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(dist), unit=UnitsType.ANGSTROM, 
                     charge=0, spin=0, basis='sto3g')
molecule = driver.run()
repulsion_energy = molecule.nuclear_repulsion_energy
num_spin_orbitals=molecule.num_orbitals*2
num_particles=molecule.num_alpha+molecule.num_beta

map_type='jordan_wigner'
ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
qubitOp = ferOp.mapping(map_type=map_type, threshold=0.00000001)


exact_solution = ExactEigensolver(qubitOp).run()['energy']
exact_solution=exact_solution+repulsion_energy

optimizer = COBYLA(maxiter=opt_iter)
HF_state=HartreeFock(qubitOp.num_qubits, num_spin_orbitals, num_particles, map_type)
var_form = UCCSD(qubitOp.num_qubits, depth=1, num_orbitals=num_spin_orbitals, num_particles=num_particles,
                 active_occupied=None, active_unoccupied=None, initial_state=HF_state,
                 qubit_mapping='jordan_wigner', two_qubit_reduction=False, num_time_slices=1,
                 shallow_circuit_concat=True, z2_symmetries=None)

n_shots=200
n_instances=5
instances=[]
for i in range(n_instances):
    instances.append(QuantumInstance(backend=BasicAer.get_backend("qasm_simulator"), shots=n_shots))

print('Molecule H2 at bond length ' + str(dist))
print('Running VQE with ' + str(n_instances) + ' different quantum instances. This may take a few minutes:')
vqe_instance = VQE(qubitOp, var_form, optimizer=optimizer,threads=len(instances))
result=vqe_instance.run(instances)
energy=result['energy']+repulsion_energy 

print('Optimal energy is ' + str(energy) +'. Exact result is ' + str(exact_solution))

