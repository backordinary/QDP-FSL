# https://github.com/jason-jk-kang/QIS-qchem/blob/6f296a4272d842a136892e33f8edd77d2051fc60/Qiskit/functions.py
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
H5PY_DEFAULT_READONLY=1


def run_simulation (system, indx, commandprinter = False, noise = False):

    def vqe_create_solver(num_particles, num_orbitals, qubit_mapping,
                          two_qubit_reduction, z2_symmetries,
                          initial_point = system.opt_amplitudes,
                          noise = noise):

        initial_state = HartreeFock(num_orbitals, num_particles, qubit_mapping,
                                    two_qubit_reduction, z2_symmetries.sq_list)

        var_form = UCCSD(num_orbitals=num_orbitals,
                         num_particles=num_particles,
                         initial_state=initial_state,
                         qubit_mapping=qubit_mapping,
                         two_qubit_reduction=two_qubit_reduction,
                         z2_symmetries=z2_symmetries)

        if noise:
            var_form = EfficientSU2(num_qubits = no_qubits, entanglement="linear")
        else:
            var_form = UCCSD(num_orbitals=num_orbitals,
                             num_particles=num_particles,
                             initial_state=initial_state,
                             qubit_mapping=qubit_mapping,
                             two_qubit_reduction=two_qubit_reduction,
                             z2_symmetries=z2_symmetries)

        vqe = VQE(var_form=var_form, optimizer=SLSQP(maxiter=500),
                  include_custom=True, initial_point = initial_point)
        vqe.quantum_instance = backend
        return vqe

    basis_string = 'sto-3g'
    charge = 0
    spin = 1

    no_qubits = 6


    if noise:
        optimizer = SPSA(maxiter=100)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        qasm = Aer.get_backend("qasm_simulator")
        device = provider.get_backend("ibmq_16_melbourne")
        coupling_map = device.configuration().coupling_map
        noise_model = NoiseModel.from_backend(device.properties())
        backend = QuantumInstance(backend=qasm,
                                  shots=10000,
                                  noise_model=noise_model,
                                  coupling_map=coupling_map,
                                  measurement_error_mitigation_cls=CompleteMeasFitter,
                                  cals_matrix_refresh_period=30)
    else:
        optimizer = SLSQP(maxiter=500)
        backend = Aer.get_backend('qasm_simulator')

    ########################################################################

    # Begin Running Simulation, Convert distance_counter to angstroms
    geometry = ['H 0. 0. ' + str(system.atoms[0].position[-1] * 0.529177249),
                'H 0. 0. ' + str(system.atoms[1].position[-1] * 0.529177249),
                'H 0. 0. ' + str(system.atoms[2].position[-1] * 0.529177249)]

    if indx is not None:
        geometry[indx] = 'H 0. 0. ' + str(system.atoms[indx].stand_by_position * 0.529177249)

    print(geometry)

    driver = PySCFDriver(atom=geometry,
                         unit=UnitsType.ANGSTROM, charge=charge, spin=spin, basis=basis_string)


    # # # # # # # # # # # # # # # # # # # # # #
    # VQE Result
    # # # # # # # # # # # # # # # # # # # # # #
    mgse = MolecularGroundStateEnergy(driver,
                                      qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                                      two_qubit_reduction=False, freeze_core=False,
                                      z2symmetry_reduction=None)

    vqe_result = mgse.compute_energy(vqe_create_solver).energy
    system.opt_amplitudes = mgse.solver.optimal_params

    # # # # # # # # # # # # # # # # # # # # # #
    # Exact Result
    # # # # # # # # # # # # # # # # # # # # # #
    mgse = MolecularGroundStateEnergy(driver, NumPyMinimumEigensolver(),
                                      qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                                      two_qubit_reduction=False, freeze_core=False,
                                      z2symmetry_reduction=None)
    exact_result = mgse.compute_energy().energy

    print("VQE Result:", vqe_result,
          "Exact Energy:", exact_result)

    return ({"Exact Energy" : exact_result, "VQE Energy" : vqe_result})
