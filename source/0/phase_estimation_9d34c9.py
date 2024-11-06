# https://github.com/yahja-dev/QuantumLabs/blob/bcde084ed481e333fe5d159b6413bda4e61ead05/QuantumPhaseEstimation/phase_estimation.py
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import operator
from methods import *

def quantum_phase_estimation(n, theta):
    """
    :param n: number of qubits
    :param theta: phase of qubits
    :return: qpe_circuit
    """
    # Create a quantum circuit on n+1 qubits (n measurement, 1 target)
    qpe_circuit = QuantumCircuit(n + 1, n)

    # Initialize the qubits
    initialize(qpe_circuit, range(n), n)

    # Apply the controlled unitary operators in sequence
    for x in range(n):
        exp = 2 ** (n - x - 1)
        exponent(qpe_circuit, x, n, theta, exp)

    # Apply the inverse quantum Fourier transform
    iqft(qpe_circuit, range(n), n)

    # Measure all qubits
    qpe_circuit.measure(range(n), range(n))

    return qpe_circuit


if __name__ == '__main__':
    n = 5
    theta = 0.5
    # Quantum Phase Estimation
    circuit = quantum_phase_estimation(n, theta)
    circuit.draw(output='mpl', filename='illustrations/qpe.png')
    # Get measurement results
    simulator = Aer.get_backend("qasm_simulator")
    counts = execute(circuit, backend=simulator, shots=1000).result().get_counts(circuit)
    # Plot results
    plot_histogram(counts, figsize=(10,10)).savefig("illustrations/results_histogram.png")
    # Get phase measurement
    highest_probability_outcome = max(counts.items(), key=operator.itemgetter(1))[0][::-1]
    measured_theta = int(highest_probability_outcome, 2) / 2 ** n
    print("Using %d qubits with theta = %.2f, measured_theta = %.2f." % (n, theta, measured_theta))
