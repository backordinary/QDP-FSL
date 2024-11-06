# https://github.com/HenningBuhl/QuantumComputing/blob/a9d64890d24c6ba24685c5b4fc6a608bfed9848b/DeutschJozsa/DeutschJozsa.py
from QuantumAlgorithm import QuantumAlgorithm
import qiskit as q
from qiskit.converters import circuit_to_gate
import numpy as np


class DeutschJozsa(QuantumAlgorithm):
    def __init__(self, args):
        super().__init__("Deutsch Jozsa", args)

    def get_circuit(self):
        n = self.args.deutsch_jozsa_n
        circuit = q.QuantumCircuit(n + 1, n)

        # Apply H-gates
        for qubit in range(n):
            circuit.h(qubit)

        # Put qubit in state |->
        circuit.x(n)
        circuit.h(n)

        # Use barrier as divider
        circuit.barrier()

        # Add oracle
        if self.args.deutsch_jozsa_oracle == 'balanced':
            oracle = self.get_balanced_oracle()
            oracle = oracle.to_gate()
            oracle.name = 'oracle'
            circuit.append(oracle, range(n+1), [])
        else:
            oracle = self.get_constant_oracle()
            oracle = oracle.to_gate()
            oracle.name = 'oracle'
            circuit.append(oracle, range(n+1), [])

        # Use barrier as divider
        circuit.barrier()

        # Repeat H-gates
        for qubit in range(n):
            circuit.h(qubit)

        # Measure
        for i in range(n):
            circuit.measure(i, i)

        return circuit

    def get_constant_oracle(self):
        n = self.args.deutsch_jozsa_n
        constant_oracle = q.QuantumCircuit(n + 1)
        output = np.random.randint(2)
        if output == 1:
            constant_oracle.x(n)
        
        return constant_oracle

    def get_balanced_oracle(self):
        n = self.args.deutsch_jozsa_n
        balanced_oracle = q.QuantumCircuit(n + 1)
        b = np.random.randint(1, 2 ** n)
        b_str = format(b, '0' + str(n) + 'b')

        # Place X-gates
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                balanced_oracle.x(qubit)

        # Controlled-NOT gates
        for qubit in range(n):
            balanced_oracle.cx(qubit, n)

        # Place X-gates
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                balanced_oracle.x(qubit)

        return balanced_oracle
