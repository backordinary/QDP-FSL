# https://github.com/anjalia16/quantum-aia/blob/4f312d59cad70f72fbe07d3a6f0f80c18ec99fa4/QuantumAIA/assessment.py
import constants
from qiskit import Aer, transpile, execute
from qiskit.providers.aer import AerSimulator
from time import perf_counter
from qiskit.test.mock import FakeManhattan, FakeMontreal
import QiskitNEQR

simulator_manhattan = AerSimulator.from_backend(FakeManhattan())
simulator_montreal = AerSimulator.from_backend(FakeMontreal())
simulator_ideal = Aer.get_backend('aer_simulator')


def resourceAssessment(image, sim, num, op_lvl, run):
    start = perf_counter()
    circuit = QiskitNEQR.NEQR(image)
    qc = transpile(circuit, backend=sim, optimization_level=op_lvl)
    end = perf_counter()

    if run:
        simulation = execute(qc, backend=sim, shots=num)
        results = simulation.result()

    print()
    print(f"Running resource assessment on the following image for {sim} and optimization level {op_lvl}:")
    for row in image:
        print(row)
    print()
    print(f"Circuit compiled in {end - start} seconds.")
    print(f"Circuit depth: {qc.depth()}")
    print(f"Necessary number of qubits: {qc.width()}")
    if run:
        print()
        print(f"Running the circuit with {num} shots.")
        print(f"Runtime: {results.time_taken} seconds.")
    print("-------------------------------------------------------------------------------")

def runAssessments(shots, optimization_level, run_real):
    for i in constants.testimages:
        resourceAssessment(i, simulator_manhattan, shots, optimization_level, run_real)
        resourceAssessment(i, simulator_montreal, shots, optimization_level, run_real)
        resourceAssessment(i, simulator_ideal, shots, optimization_level, True)

runAssessments(1024, 1, run_real=False)