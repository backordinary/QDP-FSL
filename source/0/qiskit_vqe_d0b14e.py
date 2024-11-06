# https://github.com/rdcunha/las-qpe/blob/19d30c536974ef3e5d912494df3da1fd54fdd638/qiskit-vqe.py
import numpy as np
# Qiskit imports
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import ElectronicStructureDriverType, ElectronicStructureMoleculeDriver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit.providers.aer import StatevectorSimulator
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_nature.algorithms import VQEUCCFactory, GroundStateEigensolver
from qiskit.algorithms import HamiltonianPhaseEstimation

# Define molecule in qiskit
qmol = Molecule(geometry=[['H', [0,0,0]],['H', [1,0,0]],
                         ['H',[0.2,3.9,0.1]],['H',[1.159166,4.1,-0.1]]],
                         charge=0, multiplicity=1)

# Set driver type and feed in molecule and basis
driver = ElectronicStructureMoleculeDriver(qmol, basis='sto3g', 
                                            driver_type=ElectronicStructureDriverType.PYSCF)

# Set up ESProblem object
es_problem = ElectronicStructureProblem(driver)

# The second_q_ops() function calls driver.run()
second_q_ops = es_problem.second_q_ops()
print(second_q_ops[0])

# Choose fermion-to-qubit mapping
qubit_converter = QubitConverter(mapper=JordanWignerMapper())
# This just outputs a qubit op corresponding to a 2nd quantized op
qubit_op = qubit_converter.convert(second_q_ops[0])
print(qubit_op)

# Set the backend
quantum_instance = QuantumInstance(backend = Aer.get_backend('aer_simulator_statevector'))

# Choose the solver, VQEUCC has a special factory type
# Can choose a regular solver from qiskit.algorithms
vqe_solver = VQEUCCFactory(quantum_instance)

# This just sets up the ground state solver without feeding in the problem
calc = GroundStateEigensolver(qubit_converter, vqe_solver)

# Feed in the problem
res = calc.solve(es_problem)

print(res)
