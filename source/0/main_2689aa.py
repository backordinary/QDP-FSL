# https://github.com/BHazel/hello-quantum/blob/a0578d086df28c0ee66c686a4bddc0ec285eb464/concepts/interference/qiskit/main.py
from qiskit import QuantumCircuit, Aer, execute

def interference(repeat_count, starting_state):
    """Creates a superposition and applies interference on a qubit in a specified state a specified number of times.

    Args:
        repeat_count (int): The number of times to repeat the quantum circuit.
        starting_state (int): The state to set the qubit to prior to running the quantum circuit.

    Returns:
        qiskit.result.counts.Counts: The measurement result counts.
    """
    # Set up quantum circuit with 1 qubit and 1 bit.
    # Set qubit to desired starting state.
    # If the starting state should be 1, apply the X gate.
    circuit = QuantumCircuit(1, 1)
    if starting_state == 1:
        circuit.x(0)
    
    # Apply Hadamard gate to create a superposition.
    circuit.h(0)

    # Apply Hadamard gate to cause interference to restore qubit to its starting state.
    circuit.h(0)

    # Measure the qubit and save the result into a classical bit.
    circuit.measure(0, 0)

    # Initialise simulator and run circuit a specified number of times.
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=repeat_count)

    # Get counts of qubit measurement results.
    result_counts = job.result().get_counts()
    return result_counts

print('*** Hello, Quantum! - Interference (Qiskit) ***')

repeat_count = 1000

# Cause interference on a qubit starting in the 0 state.
result_counts_starting_state_0 = interference(repeat_count, 0)

# Cause interference on a qubit starting in the 1 state.
result_counts_starting_state_1 = interference(repeat_count, 1)

# Print results.
print(f"Counts for {repeat_count} repeats with starting state 0:")
print(f"\t0: {result_counts_starting_state_0.get('0', 0)}")
print(f"\t1: {result_counts_starting_state_0.get('1', 0)}")
print(f"Counts for {repeat_count} repeats with starting state 1:")
print(f"\t0: {result_counts_starting_state_1.get('0', 0)}")
print(f"\t1: {result_counts_starting_state_1.get('1', 0)}")