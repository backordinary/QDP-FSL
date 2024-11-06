# https://github.com/imran-prog/python/blob/cb0d37e7d6756f526d6bbf7e85edea4cb59276a8/Algorithms/Quantum/single_qubit_measure.py
"""
Build a simple bare-minimum quantum circuit that starts with a single
qubit (by default, in state 0), runs the experiment 1000 times, and
finally prints the total count of the states finally observed.
Qiskit Docs: https://qiskit.org/documentation/getting_started.html
*
From -> https://github.com/TheAlgorithms/Python/tree/master/quantum
*
"""

import qiskit as q

def single_qubit_measure(qubits: int,classical_bits: int) -> q.result.counts.Counts:
    """
    >>> single_qubit_measure(1, 1)
    {'0': 1000}
    """
    # Use Aer's qasm_simulator
    simulator = q.Aer.get_backend("qasm_simulator")

    # Create a Quantum Circuitacting on the q register
    circuit = q.QuantumCircuit(qubits, classical_bits)

    # Map the Quantum Measurement to the classical bits
    circuit.measure([0], [0])

    # Execute the circuit on the qasm_calculator
    job = q.execute(circuit, simulator, shots=1000)

    # Return the histogram data of the results of the experiment.
    return job.result().get_counts(circuit)


if __name__ == "__main__":
    print(f"Total count for various states are: {single_qubit_measure(8, 4)}")

