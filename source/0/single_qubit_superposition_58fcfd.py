# https://github.com/dmsl/quantum/blob/69ffb7fa32852f286bcc6bc4f6fc42bf37706bac/QCLG_lvl1/single_qubit_superposition.py
from qiskit import execute, Aer, QuantumCircuit, IBMQ


class SingleQubitSuperposition:
    @classmethod
    def run(cls):
        # Use Aer's qasm_simulator

        simulator = Aer.get_backend('qasm_simulator')

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(1, 1)

        # Add a H gate on qubit 0
        circuit.h(0)

        # Map the quantum measurement to the classical bits
        circuit.measure(0, 0)

        # Execute the circuit on the qasm simulator
        job = execute(circuit, simulator, shots=1000)

        # Grab results from the job
        result = job.result()

        # Returns counts
        counts = result.get_counts(circuit)
        print("\nTotal count for 0 and 1 are:", counts)

        provider = IBMQ.load_account()
        backend = provider.backends.ibmq_valencia
        # Execute the circuit on a real device
        job = execute(circuit, backend=backend, shots=1000)

        # Grab results from the job
        result = job.result()

        # Returns counts
        counts = result.get_counts(circuit)
        print("\nTotal count for 0 and 1 are:", counts)

        # Draw the circuit
        print(circuit)
