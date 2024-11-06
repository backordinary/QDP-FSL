# https://github.com/masosky/qiskit_basics/blob/a207f673078013ddbcee193f545936250bae7a8a/quantum_operator.py
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

if __name__ == '__main__':

    # Circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)

    # Operator
    operator = Operator(circuit)
    # Show the results
    print(operator.data)

