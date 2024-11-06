# https://github.com/sofferor/Quantum-Algorithms/blob/ee6678001d7d09e8edbd79c06b356dc12006c173/Deutsch-Jozsa%20Algorithm/deutchJozsa.py
from qiskit import Aer, transpile
from qiskit import QuantumCircuit, assemble
from qiskit.visualization import plot_histogram
from utils import Case


def algorithm(oracle, n):
    circuit = QuantumCircuit(n + 1, n)

    circuit.x(n)
    circuit.barrier()

    circuit.h(range(n+1))
    circuit.barrier()

    circuit.append(oracle, range(n+1))
    circuit.barrier()

    circuit.h(range(n))
    circuit.barrier()

    # Measure x (no need to measure y as well)
    circuit.measure(range(n), range(n))

    print("Deutsch-Jozsa Circuit:\n{circuit}\n\n".format(circuit=circuit.draw()))

    aer_sim = Aer.get_backend('aer_simulator')
    transpiled_dj_circuit = transpile(circuit, aer_sim)
    qobj = assemble(transpiled_dj_circuit)
    results = aer_sim.run(qobj).result()
    answer = results.get_counts()
    plot_histogram(answer).show()

    if len(answer) and '1' not in next(iter(answer)):
        return Case.constant.name
    else:
        return Case.balanced.name
