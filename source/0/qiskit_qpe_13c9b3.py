# https://github.com/rdcunha/las-qpe/blob/da90d0c2991596a9d7810aebca9a893792763e9d/qiskit-qpe.py
import numpy as np
import logging
from argparse import ArgumentParser

# Qiskit imports
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import ElectronicStructureDriverType, ElectronicStructureMoleculeDriver, MethodType
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit.providers.aer import StatevectorSimulator
from qiskit import Aer
from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit_nature.algorithms import VQEUCCFactory, GroundStateEigensolver
from qiskit_nature.circuit.library import HartreeFock
from qiskit.algorithms import NumPyEigensolver, PhaseEstimation, PhaseEstimationScale
from qiskit.opflow import PauliTrotterEvolution,SummedOp,PauliOp,MatrixOp,PauliSumOp,StateFn

parser = ArgumentParser(description='Do QPE with an RHF reference')
parser.add_argument('--an', type=int, default=1, help='number of ancilla qubits')
parser.add_argument('--shots', type=int, default=1024, help='number of shots for the simulator')
args = parser.parse_args()

logging.basicConfig(filename='qiskit-qpe.log', level=logging.DEBUG)

# Define molecule in qiskit
qmol = Molecule(geometry=[['H', [0,0,0]],['H', [1.0,0,0]],['H', [0.2,3.9,0.1]],['H', [1.159166,4.1,-0.1]]],
                         charge=0, multiplicity=1)
#qmol = Molecule(geometry=[['H', [0,0,0]],['H', [0.7,0,0]]],
#                         charge=0, multiplicity=1)

# Set driver type and feed in molecule and basis
driver = ElectronicStructureMoleculeDriver(qmol, basis='sto3g', method=MethodType.RHF,
                                            driver_type=ElectronicStructureDriverType.PYSCF)

# Set up ESProblem object
es_problem = ElectronicStructureProblem(driver)

# The second_q_ops() function calls driver.run()
second_q_ops = es_problem.second_q_ops()
print(second_q_ops[0])

# Choose fermion-to-qubit mapping
qubit_converter = QubitConverter(mapper = ParityMapper(), two_qubit_reduction=True)
# This just outputs a qubit op corresponding to a 2nd quantized op
qubit_ops = [qubit_converter.convert(op) for op in second_q_ops]
hamiltonian = qubit_ops[0]
print(hamiltonian)

# Set the backend
quantum_instance = QuantumInstance(backend = Aer.get_backend('aer_simulator'), shots=args.shots)

np_solver = NumPyEigensolver(k=1)
ed_result = np_solver.compute_eigenvalues(hamiltonian)
print("NumPy result: ", ed_result.eigenvalues)

# Can choose a regular solver from qiskit.algorithms
qpe_solver = PhaseEstimation(num_evaluation_qubits=args.an, quantum_instance=quantum_instance)

# Create a unitary out of the Hamiltonian ops
# Lines 60-85 stolen from qiskit.algorithms.HPE
if isinstance(hamiltonian, PauliSumOp):
    hamiltonian = hamiltonian.to_pauli_op()
elif isinstance(hamiltonian, PauliOp):
    hamiltonian = SummedOp([hamiltonian])

pe_scale = PhaseEstimationScale.from_pauli_sum(hamiltonian)

# scale so that phase does not wrap.
scaled_hamiltonian = -pe_scale.scale * hamiltonian

# Default evolution: PauliTrotterEvolution
evolution = PauliTrotterEvolution()

# Create the unitary by evolving the Hamiltonian
unitary = evolution.convert(scaled_hamiltonian.exp_i())

if not isinstance(unitary, QuantumCircuit):
    unitary_circuit = unitary.to_circuit()
else:
    unitary_circuit = unitary

# Decomposing twice allows some 1Q Hamiltonians to give correct results
# when using MatrixEvolution(), that otherwise would give incorrect results.
# It does not break any others that we tested.
unitary = unitary_circuit.decompose().decompose()

# Printing this is not a good idea because the circuit is very large
#print(unitary)

# Create an HF initial state and add it to the estimate function
# For our (H2)_2 system 8 spin orbs, 2 alpha 2 beta electrons
init_state = HartreeFock(8, (2,2), qubit_converter)

# Estimate takes in a SummedPauli or a PauliOp and outputs a scaled estimate of the eigenvalue
res = qpe_solver.estimate(unitary=unitary, state_preparation=init_state)

print(res)
scaled_phases = pe_scale.scale_phases(res.filter_phases(cutoff=0.0,as_float=True))
print(scaled_phases)
