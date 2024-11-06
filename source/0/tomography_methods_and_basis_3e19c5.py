# https://github.com/nelimee/sqmap/blob/cdc58935174a810283ce58e5c4627564aa67c37b/benchmarks/tomography_methods_and_basis.py
import typing as ty
import numpy
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, IBMQ
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info.states import state_fidelity

from sqt.circuits import one_qubit_tomography_circuits
from sqt.basis.equidistant import EquidistantMeasurementBasis
from sqt.basis.pauli import PauliMeasurementBasis
from sqt.basis.tetrahedral import TetrahedralMeasurementBasis
from sqt.fit.grad import post_process_tomography_results_grad
from sqt.fit.mle import post_process_tomography_results_mle
from sqt.fit.lssr import post_process_tomography_results_lssr
from sqt.fit.exact import get_one_qubit_exact_density_matrix
from sqt.basis.equidistant import get_approximately_equidistant_circuits
from sqt.execution import execute

from sqmap.visualisation.flatmap import plot_over_bloch_sphere_2d

print("Recovering data from IBMQ...")
if not IBMQ.active_account():
    IBMQ.load_account()
provider = IBMQ.get_provider(hub="ibm-q-lanl", group="lanl", project="quantum-simulati")
backend = provider.get_backend("ibmq_bogota")


N = 1000
TOMOGRAPHY_METHODS = {
    "grad": post_process_tomography_results_grad,
    "mle": post_process_tomography_results_mle,
    "lssr": post_process_tomography_results_lssr,
}
BASIS = [
    TetrahedralMeasurementBasis(),
    EquidistantMeasurementBasis(5),
    EquidistantMeasurementBasis(40),
]
simulator = AerSimulator.from_backend(backend)

print("Creating circuits...")
raw_circuits: ty.List[QuantumCircuit] = get_approximately_equidistant_circuits(N)

qubit_number = backend.configuration().num_qubits

# {'basis': {'reconstruction_method': [[result[qi] for qi in range(qubit_number)] for point in points]}}
density_matrices: ty.Dict[str, ty.Dict[str, ty.List[ty.List[numpy.ndarray]]]] = dict()
points: ty.List[numpy.ndarray] = [
    numpy.array(eval(circuit.name)) for circuit in raw_circuits
]

for basis in BASIS:
    dms: ty.Dict[str, ty.List[ty.List[numpy.ndarray]]] = {
        m: [] for m in ["exact"] + list(TOMOGRAPHY_METHODS)
    }
    print(f"Creating tomography circuits for '{basis.name}'...")
    tomography_circuits: ty.List[ty.List[QuantumCircuit]] = [
        one_qubit_tomography_circuits(raw_circuit, basis, qubit_number=qubit_number)
        for raw_circuit in raw_circuits
    ]
    flattened_circuits: ty.List[QuantumCircuit] = sum(tomography_circuits, start=[])

    print("Simulating circuits...")
    result = execute(flattened_circuits, simulator, shots=2 ** 10)

    print("Computing density matrices...")
    for circuit in raw_circuits:
        exact_density_matrix: numpy.ndarray = get_one_qubit_exact_density_matrix(
            circuit
        )
        dms["exact"].append([exact_density_matrix.copy() for _ in range(qubit_number)])
        for method in TOMOGRAPHY_METHODS:
            dms[method].append(
                TOMOGRAPHY_METHODS[method](
                    result, circuit, basis, qubit_number=qubit_number
                )
            )
    density_matrices[basis.name] = dms

print("Plotting...")

fig, axes = plt.subplots(len(BASIS), len(TOMOGRAPHY_METHODS))

qubit_index = 0
for basis_index, basis in enumerate(BASIS):
    for method_index, method in enumerate(TOMOGRAPHY_METHODS):
        plot_over_bloch_sphere_2d(
            points,
            [
                density_matrices[basis.name][method][i][qubit_index]
                for i in range(len(points))
            ],
            fig=fig,
            ax=axes[basis_index][method_index],
            title=f"{basis.name} with {method}",
        )
plt.show()
