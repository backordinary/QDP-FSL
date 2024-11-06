# https://github.com/cgMuro/State-of-Quantum/blob/3619ca624e75d180fb99c681e45074a4a0e3cc20/Quantum-Phase-Estimation/QPE_qiskit.py
# Quantum Phase Estimation implementation with Qiskit
# Main resource used: https://qiskit.org/textbook/ch-algorithms/quantum-phase-estimation.html

import math
import qiskit


class QuantumPhaseEstimation():
    """ Implements the Quantum Phase Estimation algorithm logic. """
    def __init__(
        self,
        n_counting_qubits: int,  # Number of qubits in the counting register
        angle: float             # Angle used to perform the controlled-U operation
    ) -> None:
        
        self.n_counting_qubits = n_counting_qubits
        self.angle = 2 * math.pi * angle

    def qft_inverse(self, circuit: qiskit.QuantumCircuit, n: int) -> qiskit.QuantumCircuit:
        """ Applies the inverse of the Quantum Fourier Transform on the first n qubits in the given circuit. """
        for qubit in range(n//2):
            circuit.swap(qubit, n-qubit-1)
        for j in range(n):
            for m in range(j):
                circuit.cp(-math.pi/float(2**(j-m)), m, j)
            circuit.h(j)

        return circuit

    def algorithm(
            self, 
            eigenstate_x = False,   # If true applies X-gate to the eigenstate
            repetitions: int = 1,   # Number of times we want to apply the controlled-U
            draw: bool = False
        ) -> qiskit.QuantumCircuit:
        """ It applies the Quantum Phase Estimation and returns the circuit. """
        # Create circuit
        circuit = qiskit.QuantumCircuit(self.n_counting_qubits+1, self.n_counting_qubits)

        # Apply Hadamard gate to counting qubits
        for qubit in range(self.n_counting_qubits):
            circuit.h(qubit)
        
        # Prepare the eigenstate |ψ⟩
        if eigenstate_x:
            circuit.x(self.n_counting_qubits)  # Applies the X-gate on the eigenstate (last qubit) if requested

        # Apply the controlled-U operation for "repetitions" times
        for counting_qubit in range(self.n_counting_qubits):
            for _ in range(repetitions):
                circuit.cp(self.angle, counting_qubit, self.n_counting_qubits)
            repetitions *= 2

        # Apply visual barrier
        circuit.barrier()

        # Apply inverse Quantum Fourier Transform
        circuit = self.qft_inverse(circuit, self.n_counting_qubits)

        # Apply visual barrier
        circuit.barrier()

        # Apply measurement
        for qubit in range(self.n_counting_qubits):
            circuit.measure(qubit, qubit)
        

        # Print and draw circuit if requested
        if draw:
            print(circuit)
            circuit.draw()

        return circuit

    def simulation(self, circuit: qiskit.QuantumCircuit, shots: int = 4096, plot: bool = False):
        """ Performs simulation on the given circuit and returns the statevector. """
        # Get simulator and serialize qobj
        qasm_simulator = qiskit.Aer.get_backend('qasm_simulator')
        transpiled_circuit = qiskit.transpile(circuit, qasm_simulator)
        qobj = qiskit.assemble(transpiled_circuit, shots=shots)

        # Get results
        results = qasm_simulator.run(qobj).result()
        counts = results.get_counts()

        # Plot histogram if requested
        if plot:
            qiskit.visualization.plot_histogram(counts)

        return counts

    def run_quantum_hardware(self, circuit: qiskit.QuantumCircuit, shots: int = 2048, plot: bool = False):
        """ Run the given circuit on the IBM quantum hardware in the cloud. """
        # Load IBMQ account and get the least busy backend device with greater than or equal to the required qubits
        qiskit.IBMQ.load_account()
        from qiskit.tools.monitor import job_monitor
        provider = qiskit.IBMQ.get_provider(hub='ibm-q')
        backend = qiskit.providers.ibmq.least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= self.n and not x.configuration().simulator and x.status().operational == True))
        print('Least busy backend:', backend)

        # Run circuit on the least busy backend and monitor the execution of the job in the queue
        transpiled_circuit = qiskit.transpile(circuit, backend, optimization_level=3)
        qobj = qiskit.assemble(transpiled_circuit, shots=shots)
        job = backend.run(qobj)
        qiskit.tools.monitor.job_monitor(job)

        # Get counts
        counts = job.result().get_counts()

        # Plot histogram if requested
        if plot:
            qiskit.visualization.plot_histogram(counts)

        return counts
