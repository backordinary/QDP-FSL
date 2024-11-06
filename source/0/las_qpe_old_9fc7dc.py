# https://github.com/rdcunha/las-qpe/blob/19d30c536974ef3e5d912494df3da1fd54fdd638/las-qpe_old.py
import numpy as np
import logging
# Qiskit imports
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import FCIDumpDriver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit.providers.aer import StatevectorSimulator
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_nature.algorithms import VQEUCCFactory, GroundStateEigensolver
from qiskit_nature.circuit.library import HartreeFock
from qiskit.algorithms import HamiltonianPhaseEstimation
from qiskit.opflow import PauliTrotterEvolution, StateFn

logging.basicConfig(filename='las-qpe.log', level=logging.DEBUG)

# Define molecule in qiskit
qmol = Molecule(geometry=[['H', [0,0,0]],['H', [1,0,0]],
                         ['H',[0.2,3.9,0.1]],['H',[1.159166,4.1,-0.1]]],
                         charge=0, multiplicity=1)

# Set driver type and feed in molecule and basis
driver = FCIDumpDriver('fcidump_las_h4')

# Set up ESProblem object
es_problem = ElectronicStructureProblem(driver)

# The second_q_ops() function calls driver.run()
second_q_ops = es_problem.second_q_ops()
print(second_q_ops[0])

# Choose fermion-to-qubit mapping
qubit_converter = QubitConverter(mapper = ParityMapper(), two_qubit_reduction=True)
# This just outputs a qubit op corresponding to a 2nd quantized op
qubit_ops = [qubit_converter.convert(op) for op in second_q_ops]
print(qubit_ops)

# Set the backend
quantum_instance = QuantumInstance(backend = Aer.get_backend('aer_simulator'), shots=1024)

# Choose the solver, VQEUCC has a special factory type
# Can choose a regular solver from qiskit.algorithms
qpe_solver = HamiltonianPhaseEstimation(num_evaluation_qubits=8, quantum_instance=quantum_instance)

# Printing this is not a good idea because the circuit is very large
#unitary = qpe_solver._get_unitary(qubit_ops[0], qpe_solver._get_scale(qubit_ops[0], bound=4.0), PauliTrotterEvolution())
#print(unitary)

# Create an HF initial state and add it to the estimate function
# For our (H2)_2 system 8 spin orbs, 2 alpha 2 beta electrons
init_state = HartreeFock(8, (2,2), qubit_converter)

# Estimate takes in a SummedPauli or a PauliOp and presents a scaled estimate of the eigenvalue
res = qpe_solver.estimate(qubit_ops[0], state_preparation=StateFn(init_state))

print(res)
