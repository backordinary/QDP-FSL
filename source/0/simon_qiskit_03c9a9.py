# https://github.com/cgMuro/State-of-Quantum/blob/e08a1230da6d40d5f665b9fcf3b322581403da9c/Simons-problem/Simon_qiskit.py
# Simon's algorithm implementation with Qiskit
# Main resources used: https://qiskit.org/textbook/ch-algorithms/simon.html, https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/algorithms/simon_algorithm.ipynb

import numpy as np
import qiskit
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy


class SimonProblem():
    """ Implements the Simon's algorithm logic. """
    def __init__(
        self,
        bitstring: str
    ) -> None:
        
        self.b = bitstring
        self.n = len(bitstring)

    def oracle(self, circuit: qiskit.QuantumCircuit):
        """ Builds the oracle for the circuit """
        # 1. Copy qubits in the first register to the second register
        for i in range(self.n):
            circuit.cx(i, self.n+i)

        # 2. Create 1-to-1 or 2-to-1 mapping: If b is not all-zero with j being the last 1 in the bitstring and if x_j = 0 -> Then XOR the second register with b. Otherwise, do not change the second register
        # Get the index of the last 1 in bitstring
        j = self.b.rfind('1')

        # Flip the idx-th qubit (in the second register) if b_idx is 1
        for idx, char in enumerate(self.b):
            if char == '1' and j != -1:
                circuit.cx(j, self.n+idx)

        # 3. Creating random permutation: Randomly permute and flip the qubits of the second register
        # Get random permutation of n qubits
        permutation = list(np.random.permutation(self.n))

        # Init initial position
        initial_position = list(range(self.n))
        i = 0
        while i < self.n:
            if initial_position[i] != permutation[i]:
                k = permutation.index(initial_position[i])
                # Swap qubits
                circuit.swap(self.n+i, self.n+k)
                # Mark swapped qubits
                initial_position[i], initial_position[k] = initial_position[k], initial_position[i]
            else:
                i += 1
                
        # Randomly flip the qubits
        for i in range(self.n):
            if np.random.random() > 0.5:
                circuit.x(self.n+i)

        return circuit

    def algorithm(self, draw: bool):
        """ Returns the circuit for performing the Simon's algorithm. """
        # Create circuit for Simon's algorithm
        simon_circuit = qiskit.QuantumCircuit(self.n*2, self.n)

        # Apply Hadamard gates
        simon_circuit.h(range(self.n))

        # Apply visual barrier 
        simon_circuit.barrier()

        # Add oracle
        simon_circuit = self.oracle(simon_circuit)

        # Apply visual barrier 
        simon_circuit.barrier()

        # Apply Hadamard gates to the input register
        simon_circuit.h(range(self.n))

        # Measure qubits
        simon_circuit.measure(range(self.n), range(self.n))

        if draw:
            print(simon_circuit)
            simon_circuit.draw()

        return simon_circuit

    def bdotz(self, b, z):
        """ Utility function used to calculate the dot product of the given values b and z """
        accum = 0
        for i  in range(len(b)):
            accum += int(b[i]) * int(z[i])
        return accum % 2

    def simulation(self, circuit, shots: int = 1024, plot: bool = False):
        """ Performs simulation on the given circuit. """
        # Get simulator and serialize qobj
        qasm_simulator = qiskit.Aer.get_backend('qasm_simulator')
        # transpiled_circuit = qiskit.transpile(circuit, qasm_simulator)
        qobj = qiskit.assemble(circuit, shots=shots)

        # Get results
        results = qasm_simulator.run(qobj).result()
        counts = results.get_counts()

        # Plot histogram if requested
        if plot:
            qiskit.visualization.plot_histogram(counts)

        # Print results
        for z in counts:
            print(f'{self.b}•{z} = {self.bdotz(self.b, z)} (mod 2)')        

        return counts

    def run_quantum_hardware(self, circuit, shots: int = 1024, plot: bool = False):
        """ Run the given circuit on the IBM quantum hardware in the cloud. """
        # Load IBMQ account and get the least busy backend device with greater than or equal to (n+1) qubits
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= self.n and not x.configuration().simulator and x.status().operational == True))
        print('Least busy backend:', backend)

        # Run our circuit on the least busy backend and monitor the execution of the job in the queue
        transpiled_circuit = qiskit.transpile(circuit, backend, optimization_level=3)
        qobj = qiskit.assemble(transpiled_circuit, shots=shots)
        job = backend.run(qobj)
        qiskit.tools.monitor.job_monitor(job, interval=2)

        # Get results
        results = job.result().get_counts()

        # Plot histogram if requested
        if plot:
            qiskit.visualization.plot_histogram(results)

        # Print results
        print('b = ' + self.b)
        for z in results:
            print(f'{self.b}•{z} = {self.bdotz(self.b, z)} (mod 2) ({(results[z]*100/shots):.1f}%)')

        return results
