# https://github.com/nelimee/sqmap/blob/cdc58935174a810283ce58e5c4627564aa67c37b/examples/fake_backend_delay.py
import typing as ty

import numpy
import matplotlib.pyplot as plt

from qiskit import IBMQ
from qiskit.providers.aer import AerSimulator

from sqt.circuits import one_qubit_tomography_circuits
from sqt.basis.tetrahedral import TetrahedralMeasurementBasis
from sqt.fit.mle import post_process_tomography_results_mle

from sqmap.circuits import get_approximately_equidistant_circuits
from sqmap.visualisation.flatmap import plot_bloch_vector_displacement_arrow_field_2d

n: int = 100

circuits = get_approximately_equidistant_circuits(n)

for c in circuits:
    c.delay(32 * 20000, 0)

print("Recovering data from IBMQ...")
if not IBMQ.active_account():
    IBMQ.load_account()
provider = IBMQ.get_provider(hub="ibm-q-lanl", group="lanl", project="quantum-simulati")
backend = provider.get_backend("ibmq_bogota")
simulator = AerSimulator.from_backend(backend)

print("Generating tomography circuits...")
qubit_number: int = 5
basis = TetrahedralMeasurementBasis()
tomography_circuits = [
    one_qubit_tomography_circuits(circuit, basis, qubit_number) for circuit in circuits
]

print("Simulating tomography circuits...")
result = simulator.run(sum(tomography_circuits, start=[]), shots=2 ** 13).result()

print("Computing the density matrices...")
density_matrices_all_qubits: ty.List[ty.List[numpy.ndarray]] = [
    post_process_tomography_results_mle(result, circuit, basis, qubit_number)
    for circuit in circuits
]

print("Plotting...")
qubit_index = 0
points: ty.List[numpy.ndarray] = list()
density_matrices: ty.List[numpy.ndarray] = list()
for i, circuit in enumerate(circuits):
    point: numpy.ndarray = numpy.asarray(eval(circuit.name))  # UGLY
    points.append(point)
    density_matrices.append(density_matrices_all_qubits[i][qubit_index])

plot_bloch_vector_displacement_arrow_field_2d(points, density_matrices)
plt.show()
