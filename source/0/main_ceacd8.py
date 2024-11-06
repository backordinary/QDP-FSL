# https://github.com/sol4ik/quantum-topological-analysis-of-stock-market-data/blob/49e992e866927794c440a2d013b7ea6b3d7706be/modules/main.py
from qiskit import IBMQ, QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram

from preprocessing import preprocess
from quantum import config_circuit, distance, persistence_homology

from itertools import combinations


if __name__ == "__main__":
    preprocess("../data/gspc/^GSPC-feb-2020-corona-daily.csv")

    with open("ibmq_token.txt", 'r') as token_file:
        token = token_file.readline()
    IBMQ.save_account(token, overwrite=True)

    # specify any other IBM Q backend you want to use
    simulator = Aer.get_backend('qasm_simulator')

    with open("results/circuit_conf.csv") as conf_file:
        n_points = len(conf_file.readlines()) - 1

    all_pairs = list(combinations(range(n_points), 2))

    n_pairs = len(all_pairs)
    n_qubits = n_points + n_pairs

    circuit = QuantumCircuit(n_qubits, n_qubits)

    # data preparation
    config_circuit(circuit, "results/circuit_conf.csv")

    c_bit = n_points  # control bits counter
    for i in range(n_pairs):
        distance(circuit, c_bit, all_pairs[i][0], all_pairs[i][1])
        c_bit += 1

    # persistence homology calculations
    persistence_homology(circuit, n_qubits)

    # run IBM Q circuit and get results
    job = execute(circuit, simulator, shots=1000)

    result = job.result()
    counts = result.get_counts(circuit)
    print("\nTotal count for states are:", counts)
    plot_histogram(counts)

    # postprocess the results
