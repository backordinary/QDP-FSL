# https://github.com/cgMuro/State-of-Quantum/blob/c911f383b56dacbc60c5cb904b57071943d7ed30/Quantum-Fourier-Transform/QFT_qiskit.py
# Quantum Fourier Transform implementation with Qiskit
# Main resource used: https://qiskit.org/textbook/ch-algorithms/quantum-fourier-transform.html

import numpy as np
import qiskit
from qiskit.providers.ibmq import IBMQ, least_busy


class QuantumFourierTransform():
    """ Implements the Quantum Fourier Transform algorithm logic. """
    def __init__(
        self,
        num_qubits: int  # Number of qubits in the circuit
    ) -> None:
        
        self.n = num_qubits
        # Create quantum circuit
        self.circuit = qiskit.QuantumCircuit(self.n)

    def rotations(self, circuit: qiskit.QuantumCircuit, n: int):
        """ Performs Quantum Fourier Transform on the first n qubits in circuit. """
        # Exit function if circuit is empty
        if n == 0:
            return circuit
        
        # Reduce n because indexes start from 0
        n -= 1
        # Apply the Hadamard gate to the most significant qubit
        circuit.h(n)

        # For the less significant qubits -> apply smaller-angled controlled rotation
        for qubit in range(n):
            circuit.cp(
                np.pi/(2**(n-qubit)), 
                qubit, 
                n
            )
        
        # Call the function again with n reduced by 1 (it was done before in the function)
        circuit = self.rotations(circuit, n)

        return circuit

    def swap_registers(self, circuit: qiskit.QuantumCircuit, n: int):
        """ Swaps qubits at the end of the circuit. """
        for qubit in range(n//2):
            circuit.swap(qubit, n-qubit-1)
        return circuit
    
    def algorithm(self, draw: bool = False):
        """ Main function. It applies the Quantum Fourier Transform and returns the circuit. """
        # Rotare circuit
        circuit = self.rotations(self.circuit, self.n)
        # Swaps qubits
        circuit = self.swap_registers(circuit, self.n)

        # Print and draw circuit if requested
        if draw:
            print(circuit)
            circuit.draw()

        return circuit

    def simulation(self, circuit: qiskit.QuantumCircuit, plot: bool = False):
        """ Performs simulation on the given circuit and returns the statevector. """
        # Get simulator and serialize qobj
        sv_sim = qiskit.Aer.get_backend('statevector_simulator')
        # Get statevector
        statevector = sv_sim.run(qiskit.assemble(circuit)).result().get_statevector()
        # Plot bloch sphere if requested
        if plot:
            qiskit.visualization.plot_bloch_multivector(statevector)

        return statevector

    def run_quantum_hardware(self, circuit: qiskit.QuantumCircuit, shots: int = 2048, plot: bool = False):
        """ 
            Run the given circuit on the IBM quantum hardware in the cloud. 
            We first create the state in Fourier basis, then run QFT in reverse and finally verify that the output corresponds to the expected computational basis.
        """
        # Take the inverse of the circuit
        inverse_qft_circuit = circuit.inverse()
        # Add it to the first n qubits in existing circuit
        circuit.append(inverse_qft_circuit, circuit.qubits[:self.n])
        # Use decompose to see the individual gates
        circuit = circuit.decompose()

        # Load IBMQ account and get the least busy backend device with greater than or equal to (n+1) qubits
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= self.n and not x.configuration().simulator and x.status().operational==True))       
        print('Least busy backend:', backend)

        # Run our circuit on the least busy backend and monitor the execution of the job in the queue
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
