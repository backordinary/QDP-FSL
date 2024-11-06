# https://github.com/dfedorov1988/qiskit_dev_examples/blob/29844903d684d8ac0ae8ce8c0988616e36b0ec19/dipoles.py
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, Molecule
from qiskit.chemistry.transformations import FermionicTransformation, FermionicQubitMappingType
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.chemistry.algorithms.ground_state_solvers.minimum_eigensolver_factories import VQEUCCSDFactory
from qiskit.aqua.algorithms import VQE
from qiskit.chemistry.algorithms.ground_state_solvers import GroundStateEigensolver

molecule = Molecule(geometry=[['H', [0., 0., 0.]],
                              ['H', [0., 0., 0.735]]],
                     charge=0, multiplicity=1)
driver = PySCFDriver(molecule = molecule, unit=UnitsType.ANGSTROM, basis='sto3g')
transformation = FermionicTransformation(qubit_mapping=FermionicQubitMappingType.JORDAN_WIGNER)

numpy_solver = NumPyMinimumEigensolver()
vqe_solver = VQEUCCSDFactory(QuantumInstance(BasicAer.get_backend('statevector_simulator')))

calc = GroundStateEigensolver(transformation, vqe_solver)
res = calc.solve(driver)

print(res)

calc = GroundStateEigensolver(transformation, numpy_solver)
res = calc.solve(driver)
print(res)