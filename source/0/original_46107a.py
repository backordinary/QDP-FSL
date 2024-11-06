# https://github.com/jason-jk-kang/QIS-qchem/blob/6f296a4272d842a136892e33f8edd77d2051fc60/Qiskit/Original%20(No%20Propogation)/noiseless/original.py
import os
import time
import matplotlib.pyplot as plt
import numpy as np

from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.circuit.library import EfficientSU2
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP
from qiskit.aqua.operators import Z2Symmetries
from qiskit import IBMQ, BasicAer, Aer
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit import IBMQ
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQE, NumPyEigensolver, NumPyMinimumEigensolver
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel
from qiskit.chemistry.applications import MolecularGroundStateEnergy
from qiskit.chemistry.core import QubitMappingType




# Settings
n_points = 40
distances = [0.2 + d * 3.0/n_points for d in range(n_points)]
exact_energies = []
vqe_energies = []

basis_string = 'sto-3g'
charge = 0
spin = 1

no_qubits = 6
optimizer = SPSA(maxiter=100)
optimizer = SLSQP(maxiter=500)
backend = Aer.get_backend('qasm_simulator')

# vqe Solver
def vqe_create_solver(num_particles, num_orbitals, qubit_mapping,
                      two_qubit_reduction, z2_symmetries):

    initial_state = HartreeFock(num_orbitals, num_particles, qubit_mapping,
                                two_qubit_reduction, z2_symmetries.sq_list)

    var_form = UCCSD(num_orbitals=num_orbitals,
                     num_particles=num_particles,
                     initial_state=initial_state,
                     qubit_mapping=qubit_mapping,
                     two_qubit_reduction=two_qubit_reduction,
                     z2_symmetries=z2_symmetries)

    vqe = VQE(var_form=var_form, optimizer=SLSQP(maxiter=500), include_custom=True)
    vqe.quantum_instance = backend
    return vqe


for dist in distances:

    molecule_string = "H .0 .0 .0; H .0 .0 " + str(dist) + "; H .0 .0 3.3"
    driver = PySCFDriver(atom=molecule_string,
                         unit=UnitsType.ANGSTROM, charge=charge, spin=spin, basis=basis_string)


    # # # # # # # # # # # # # # # # # # # # # #
    # vqe Result
    # # # # # # # # # # # # # # # # # # # # # #
    mgse = MolecularGroundStateEnergy(driver, qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                                      two_qubit_reduction=False, freeze_core=False,
                                      z2symmetry_reduction=None)
    vqe_result = mgse.compute_energy(vqe_create_solver)
    vqe_energies.append(vqe_result.energy)


    # # # # # # # # # # # # # # # # # # # # # #
    # Exact Result
    # # # # # # # # # # # # # # # # # # # # # #
    mgse = MolecularGroundStateEnergy(driver, NumPyMinimumEigensolver(),
                                      qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                                      two_qubit_reduction=False, freeze_core=False,
                                      z2symmetry_reduction=None)
    exact_result = mgse.compute_energy()
    exact_energies.append(exact_result.energy)



    print("Interatomic Distance:", np.round(dist, 2),
          "vqe Result:", vqe_energies[-1],
          "Exact Energy:", exact_energies[-1])


print("All energies have been calculated")


# plot energies
f1 = plt.figure(0)
plt.plot(distances, vqe_energies, '-', color = 'orange', label = "vqe")
plt.plot(distances, exact_energies, ':', color = 'blue', label = "exact")
plt.ylabel('Energy in Hartree')
plt.xlabel('Position of h2 atom in angstrom')
plt.legend()
plt.tight_layout()

plt.savefig("qiskit-Original-energy-graph", dpi=400, orientation='portrait')

energy_delta = [b - a for a, b in zip(exact_energies, exact_energies[1:])]
length_delta = [b - a for a, b in zip(distances, distances[1:])]

force = [-a/b for a, b in zip(energy_delta, length_delta)]

distances = [a + 1/2*(b - a) for a, b in zip(distances, distances[1:])]

f2 = plt.figure(1)
plt.plot(distances, force, '-')
plt.ylabel('Force in Hartree / angstrom')
plt.xlabel('Position of h2 atom in angstrom')
plt.tight_layout()

plt.savefig("qiskit-Original-force-graph", dpi=400, orientation='portrait')
