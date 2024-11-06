# https://github.com/nelimee/sqt/blob/d1c5ab046777d03ad6bab5fb782d04e01c1ebd11/examples/simulator.py
import typing as ty
import numpy

from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator

from sqt.circuits import one_qubit_tomography_circuits
from sqt.basis.tetrahedral import TetrahedralMeasurementBasis
from sqt.basis.pauli import PauliMeasurementBasis
from sqt.basis.equidistant import EquidistantMeasurementBasis

from sqt.fit.grad import post_process_tomography_results_grad
from sqt.fit.mle import post_process_tomography_results_mle
from sqt.fit.lssr import post_process_tomography_results_lssr

METHODS = {
    "grad": post_process_tomography_results_grad,
    "mle": post_process_tomography_results_mle,
    "lssr": post_process_tomography_results_lssr,
}
simulator = AerSimulator(method="automatic")

raw_circuit = QuantumCircuit(1, 1, name="Ry(pi/8)")
raw_circuit.ry(numpy.pi / 8, 0)
raw_circuit.rz(numpy.pi / 3, 0)

qubit_number = 5
# basis = TetrahedralMeasurementBasis()
# basis = PauliMeasurementBasis()
basis = EquidistantMeasurementBasis(4)
tomography_circuits = one_qubit_tomography_circuits(
    raw_circuit, basis, qubit_number=qubit_number
)

result = simulator.run(tomography_circuits, shots=2 ** 15).result()

density_matrices: ty.Dict[str, ty.List[numpy.ndarray]] = {}
for method in METHODS:
    print(f"Method: {method}")
    density_matrices[method] = METHODS[method](
        result, raw_circuit, basis, qubit_number=qubit_number
    )

raw_circuit.save_density_matrix()
dm_result = simulator.run(raw_circuit).result()
exact_density_matrix: numpy.ndarray = dm_result.results[0].data.density_matrix

from qiskit.quantum_info.states import DensityMatrix, state_fidelity

print()
exact_state = DensityMatrix(exact_density_matrix)
for i in range(qubit_number):
    for m in METHODS:
        obtained_state = DensityMatrix(density_matrices[m][i])
        fidelity: float = state_fidelity(exact_state, obtained_state, validate=False)
        print(f"Using {m:>5}: {1 - fidelity:.4e}")
    print("=" * 80)
