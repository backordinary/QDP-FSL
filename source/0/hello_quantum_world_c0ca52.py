# https://github.com/dmsl/quantum/blob/69ffb7fa32852f286bcc6bc4f6fc42bf37706bac/QCLG_lvl1/hello_quantum_world.py
from qiskit import IBMQ
from qiskit import (
    QuantumCircuit,
    execute,
    Aer)


class HelloWorld:
    @classmethod
    def run(cls):
        with open('./credentials/token', 'r') as file:
            token = file.read()
        IBMQ.save_account(token, overwrite=True)
        # Use Aer's qasm_simulator
        simulator = Aer.get_backend('qasm_simulator')
        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(2, 2)
        # Add a H gate on qubit 0
        circuit.h(0)
        # Add a CX (CNOT) gate on control qubit 0 and target qubit 1
        circuit.cx(0, 1)
        # Map the quantum measurement to the classical bits
        circuit.measure([0, 1], [0, 1])
        # Execute the circuit on the qasm simulator
        job = execute(circuit, simulator, shots=1024)
        # Grab results from the job
        result = job.result()
        # Returns counts
        counts = result.get_counts(circuit)
        print("Results for the Bell State experiment.")
        print("\nTotal count for 00 and 11 are:", counts)
        # Draw the circuit
        print(circuit)
