# https://github.com/aakif-akhtar/error_mitigation/blob/070f3ebc2a332b87b8fcdf5119a72dfa742a5d57/libraries/vqe_ansatz.py
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, transpile, Aer, IBMQ, execute
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization.electronic import (
    ElectronicStructureProblem,
)
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import (
    ParityMapper,
    BravyiKitaevMapper,
    JordanWignerMapper,
)
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber
from qiskit_nature.transformers.second_quantization.electronic import (
    FreezeCoreTransformer,
)
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit.providers.aer import QasmSimulator
from qiskit_nature.algorithms import VQEUCCFactory, AdaptVQE
from qiskit.circuit.library import EfficientSU2, ExcitationPreserving
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import SPSA, COBYLA, SLSQP, QNSPSA
from qiskit_nature.circuit.library.ansatzes import UCCSD
from qiskit_nature.circuit.library import HartreeFock
from qiskit.algorithms import VQE, NumPyEigensolver


class vqe:
    def __init__(self, ansatz_id, repitition=1) -> None:

        self.ansatz_id = ansatz_id
        if repitition == 1:
            self.repitition = 1
        else:
            self.repitition = repitition

        # self.bond_distance = 0.75
        self.bond_distance = 10

        # self.molecule = Molecule(geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, self.bond_distance]]], charge=0, multiplicity=1)# H2
        self.molecule = Molecule(
            geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, self.bond_distance]]],
            charge=1.0,
            multiplicity=2,
        )  # H2+
        # self.molecule = Molecule(
        #     geometry=[
        #         ["H", [0.0, 0.504977000, 0.0]],
        #         ["H", [0.437323000, -0.252489000, 0.0]],
        #         ["H", [-0.437323000, -0.252489000, 0.0]],
        #     ],
        #     charge=1.0,
        #     multiplicity=1,
        # )  # H3+

        self.driver = ElectronicStructureMoleculeDriver(
            self.molecule,
            basis="sto3g",
            driver_type=ElectronicStructureDriverType.PYSCF,
        )
        self.properties = self.driver.run()
        self.particle_number = self.properties.get_property(ParticleNumber)
        self.problem = ElectronicStructureProblem(self.driver)
        # self.problem = ElectronicStructureProblem(self.driver, transformers=[FreezeCoreTransformer()])
        self.converter = QubitConverter(JordanWignerMapper(), two_qubit_reduction=True)
        self.second_q_ops = self.problem.second_q_ops()
        self.main_op = self.second_q_ops[0]
        self.num_particles = self.problem.num_particles
        self.num_spin_orbitals = self.problem.num_spin_orbitals
        self.init_state = HartreeFock(
            self.num_spin_orbitals, self.num_particles, self.converter
        )

    def create_hamiltonian(self):
        qubit_op = self.converter.convert(
            self.main_op, num_particles=self.num_particles
        )

        return qubit_op

    def get_num_qubits(self):

        return vqe(self.ansatz_id).create_hamiltonian().num_qubits

    def get_circ_1(self, params) -> QuantumCircuit:
        # num_pars = 2*vqe(self.ansatz_id).get_num_qubits()*self.repitition
        paravec = params
        # paravec = np.random.randn(num_pars)
        # paravec = np.array([-0.08691733277146457,3.1773022318936697,0.17688210271442215,0.041291420803605365,0.18972954095723865, 0.008727814823910432,2.4995610547237423, -0.729111188626813])
        circ = QuantumCircuit(vqe(self.ansatz_id).get_num_qubits())

        arg_count = 0

        for _ in range(self.repitition):

            for i in range(circ.num_qubits):
                circ.rx(paravec[i], i)
                circ.y(i)
                arg_count += 1

            for i in range(circ.num_qubits - 1):
                circ.cx(i, i + 1)

            for i in range(circ.num_qubits):
                circ.rx(paravec[arg_count], i)
                circ.y(i)
                arg_count += 1

            circ.barrier()
            circ.compose(self.init_state, front=True, inplace=True)
        return circ

    def get_circ_2(self, params):

        # num_pars = 2*vqe(self.ansatz_id).get_num_qubits()*self.repitition + 2*(vqe(self.ansatz_id).get_num_qubits()-1)*self.repitition

        # paravec = np.random.randn(num_pars)

        paravec = params
        circ = QuantumCircuit(vqe(self.ansatz_id).get_num_qubits())

        arg_count = 0

        for _ in range(self.repitition):

            for i in range(circ.num_qubits):
                circ.rz(paravec[i], i)
                arg_count += 1
            for i in range(circ.num_qubits - 1):
                circ.rxx(paravec[arg_count], i, i + 1)
                circ.ryy(paravec[arg_count], i, i + 1)
                arg_count += 1
            for i in range(circ.num_qubits):
                circ.rz(paravec[arg_count], i)
                arg_count += 1
            circ.barrier()
            circ = circ.decompose().decompose(gates_to_decompose=["rxx", "ryy"])
            # circ.compose(self.init_state, front=True, inplace=True)
        return circ

    def get_efficient_su2(self) -> QuantumCircuit:

        circ = EfficientSU2(
            vqe(self.ansatz_id).get_num_qubits(),
            entanglement="linear",
            reps=self.repitition,
            su2_gates=["rx", "y"],
        )
        circ.compose(self.init_state, front=True, inplace=True)

        return circ.decompose()

    def get_excitation_preserving(self) -> QuantumCircuit:
        circ = ExcitationPreserving(
            self.num_spin_orbitals, mode="iswap", reps=1, entanglement="linear"
        )
        circ.compose(self.init_state, front=True, inplace=True)

        return circ.decompose()

    def get_ansatz(self):
        ansatz = {1: self.get_efficient_su2(), 2: self.get_excitation_preserving()}

        return ansatz[self.ansatz_id]
