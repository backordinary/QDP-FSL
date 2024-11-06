# https://github.com/ATourkine/QOSF_QMP/blob/a00dbd783f2ab0d1cadd17f5c109b2cf54492de3/Solution1.py
import qiskit as qskt
from math import sqrt

def ccz(x1, x2, y, circuit):
    circuit.h(y)
    circuit.ccx(x1, x2, y)
    circuit.h(y)

def errorCode(input, errorQb, errorTypeQb, errorProba, bitFlipProba, circuit):
    # circuit.
    for i in range(len(input)):
        circuit.initialize([sqrt(1 - errorProba), sqrt(errorProba)], errorQb[i])
        circuit.initialize([sqrt(1 - bitFlipProba), sqrt(bitFlipProba)], errorTypeQb[i])
        circuit.ccx(errorQb[i], errorTypeQb[i], input[i])

        circuit.x(errorTypeQb[i])
        ccz(errorQb[i], errorTypeQb[i], input[i], circuit)

dimension = 1
qb1 = qskt.QuantumRegister(dimension, 'qb1')
qb2 = qskt.QuantumRegister(dimension, 'qb2')
errorQB = qskt.QuantumRegister(dimension * 4, 'error')
ancilla = qskt.QuantumRegister(2, 'ancilla')
cm = qskt.ClassicalRegister(2)

circuit = qskt.QuantumCircuit(qb1, qb2, errorQB, ancilla, cm)

# Basic circuit
circuit.h(qb1)

# Error code
errorCode(qb1, errorQB[:dimension], errorQB[dimension : (2*dimension)], 0.3, 0.5, circuit)
errorCode(qb2, errorQB[(2*dimension):(3*dimension)], errorQB[(3*dimension):], 0.3, 0.5, circuit)

circuit.cx(qb1, qb2)

# Fixing the sign flip on the first qbit
circuit.h(qb1)
circuit.h(qb2)
circuit.cx(qb2, ancilla[1])

circuit.cx(qb1, ancilla[1])
circuit.cx(ancilla[1], qb1)

circuit.h(qb1)
circuit.h(qb2)

# Fixing the bit flip on the second qbit
circuit.cx(qb1, ancilla[0])
circuit.cx(qb2, ancilla[0])
circuit.cx(ancilla[0], qb2)


# # Uncomputing
# circuit.cx(qb1, qb2)
# circuit.h(qb1)


circuit.measure(qb1, cm[0])
circuit.measure(qb2, cm[1])

simulator = qskt.Aer.get_backend('qasm_simulator')
nbShots = 10000
result = qskt.execute(circuit, simulator, shots = nbShots).result()
print(result.get_counts())
