# https://github.com/DiracMG3/Hamiltonian-Benchmarks/blob/c047f80b395d9b14a7a2dd2b3fcf3eaf41203554/demovqe.py
from qiskit import Aer
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper,BravyiKitaevMapper
from qiskit.algorithms.optimizers import SLSQP
from qiskit.opflow import X, Z, I


slsqp = SLSQP(maxiter=1000)
H2 = [['H', [0., 0, 0]],['H', [0, 0, -1.5]]]
qubit_converter = QubitConverter(JordanWignerMapper())

molecule = Molecule(
    geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.735]]], charge=0, multiplicity=1
)
driver = ElectronicStructureMoleculeDriver(
    molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF
)

es_problem = ElectronicStructureProblem(driver)


from qiskit.providers.aer import StatevectorSimulator
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_nature.algorithms import VQEUCCFactory

quantum_instance = QuantumInstance(backend=Aer.get_backend("aer_simulator_statevector"))
vqe_solver = VQEUCCFactory(quantum_instance)

from qiskit.algorithms import VQE
from qiskit.circuit.library import TwoLocal
from qiskit_nature.algorithms import GroundStateEigensolver

tl_circuit = TwoLocal(
    rotation_blocks='ry', entanglement_blocks='cz'
)

vqe_solver = VQE(
    ansatz=tl_circuit,
    optimizer=slsqp,
    quantum_instance=QuantumInstance(Aer.get_backend("aer_simulator_statevector")),
)

calc = GroundStateEigensolver(qubit_converter, vqe_solver)
res = calc.solve(es_problem)

print(res.groundenergy)