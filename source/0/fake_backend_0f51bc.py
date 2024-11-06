# https://github.com/nelimee/sqt/blob/d1c5ab046777d03ad6bab5fb782d04e01c1ebd11/examples/fake_backend.py
import typing as ty
import numpy

from qiskit import QuantumCircuit, IBMQ, circuit
from qiskit.providers.aer import AerSimulator

from sqt.circuits import one_qubit_tomography_circuits
from sqt.basis.tetrahedral import TetrahedralMeasurementBasis
from sqt.fit.grad import post_process_tomography_results_grad
from sqt.fit.mle import post_process_tomography_results_mle
from sqt.fit.lssr import post_process_tomography_results_lssr

if not IBMQ.active_account():
    IBMQ.load_account()
provider = IBMQ.get_provider(hub="ibm-q-lanl", group="lanl", project="quantum-simulati")
backend = provider.get_backend("ibmq_bogota")
simulator = AerSimulator.from_backend(backend)

METHODS = {
    "grad": post_process_tomography_results_grad,
    "mle": post_process_tomography_results_mle,
    "lssr": post_process_tomography_results_lssr,
}
exact_simulator = AerSimulator(method="automatic")

raw_circuit = QuantumCircuit(1, 1, name="Ry(pi/8)")
raw_circuit.ry(numpy.pi / 8, 0)
raw_circuit.rz(numpy.pi / 3, 0)

qubit_number: int = simulator.configuration().num_qubits
basis = TetrahedralMeasurementBasis()
tomography_circuits = one_qubit_tomography_circuits(
    raw_circuit, basis, qubit_number=qubit_number
)

max_shots: int = simulator.configuration().max_shots
result = simulator.run(tomography_circuits, shots=max_shots).result()

density_matrices: ty.Dict[str, ty.List[numpy.ndarray]] = {}
for method in METHODS:
    density_matrices[method] = METHODS[method](
        result, raw_circuit, basis, qubit_number=qubit_number
    )

raw_circuit.save_density_matrix()
dm_result = exact_simulator.run(raw_circuit).result()
exact_density_matrix: numpy.ndarray = dm_result.results[0].data.density_matrix

from qiskit.quantum_info.states import DensityMatrix, state_fidelity

print()
exact_state = DensityMatrix(exact_density_matrix)
# print(exact_state)
for i in range(qubit_number):
    for m in METHODS:
        obtained_state = DensityMatrix(density_matrices[m][i])
        # print(obtained_state)
        fidelity: float = state_fidelity(exact_state, obtained_state, validate=False)
        print(f"Using {m:>5}: {1 - fidelity:.4e}")
    print("=" * 80)
# print(exact_state)

# print(raw_circuit.draw())
# for t in tomography_circuits:
#     print(t.draw())
