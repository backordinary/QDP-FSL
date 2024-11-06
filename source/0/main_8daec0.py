# https://github.com/BHazel/hello-quantum/blob/a0578d086df28c0ee66c686a4bddc0ec285eb464/concepts/measurement-bases/qiskit/main.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute

def superposition(repeat_count, basis_measurement):
    """Creates a superposition and measures it using a specified basis a specified number of times.

    Args:
        repeat_count (int): The number of times to repeat the quantum circuit.
        basis_measurement (function): The function to perform the specified basis measurement with.

    Returns:
        qiskit.result.counts.Counts: The measurement result counts.
    """
    # Set up quantum circuit with 1 qubit and 1 bit.
    # Apply Hadamard gate to qubit 0.
    # Measure qubit 0 into bit 0 using the specified measurement basis.
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    basis_measurement(circuit, 0, 0)

    # Initialise simulator and run circuit a specified number of times.
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=repeat_count)

    # Get counts of qubit measurement results.
    result_counts = job.result().get_counts()
    return result_counts

def measure_x(quantum_circuit, qubit, bit):
    """Measures a specified qubit in the Hadamard (X) basis.

    Args:
        quantum_circuit (qiskit.circuit.quantumcircuit.QuantumCircuit): The quantum circuit to apply the measurement to.
        qubit (int): The qubit to measure.
        bit (int): The bit to save the qubit measurement result into.
    """
    # Apply Hadamard gate to the qubit.
    # Measure the qubit into the bit.
    quantum_circuit.h(qubit)
    quantum_circuit.measure(qubit, bit)

def measure_z(quantum_circuit, qubit, bit):
    """Measures a specified qubit in the Computational (Z) basis.

    Args:
        quantum_circuit (qiskit.circuit.quantumcircuit.QuantumCircuit): The quantum circuit to apply the measurement to.
        qubit (int): The qubit to measure.
        bit (int): The bit to save the qubit measurement result into.
    """
    # Measure the qubit into the bit.
    quantum_circuit.measure(qubit, bit)

print('*** Hello, Quantum! - Measurement Bases (Qiskit) ***')

repeat_count = 1000

# Create and measure a superposition in the Computational Basis (Z-Basis).
result_counts_computational = superposition(repeat_count, measure_z)

# Create and measure a superposition in the Hadamard Basis (X-Basis).
result_counts_hadamard = superposition(repeat_count, measure_x)

# Print results.
print(f"Counts for {repeat_count} repeats in Computational Basis:")
print(f"\t0: {result_counts_computational.get('0', 0)}")
print(f"\t1: {result_counts_computational.get('1', 0)}")
print(f"Counts for {repeat_count} repeats in Hadamard Basis:")
print(f"\t0: {result_counts_hadamard.get('0', 0)}")
print(f"\t1: {result_counts_hadamard.get('1', 0)}")