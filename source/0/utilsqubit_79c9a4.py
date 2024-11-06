# https://github.com/ilverdo97/QuantumProject/blob/a6804438a8f68db52525b7a8e616eeafebaa7b96/functions/utilsQubit.py
from qiskit import *

# Prepare the register with a number of operation
def initQubits(str, qc, reg, n):
    # Flip the corresponding qubit in register if a bit in the string is a 1
    for i in range(n):
        if str[i] == "1":
            qc.x(reg[n-(i+1)])

# Calculate the correct number qubit of operation
def NQubit(operator, len1, len2, first, second):
    if operator == '+' or operator == '-' or operator == '/':
        if len2>len1:
            first,second = second, first
            len2, len1 = len1, len2
        second = ("0") * (len1 - len2) + second
        tmpnqubit = len1
    elif operator == '*' or operator == '^':
        first = ("0") * (len2) + first
        tmpnqubit = len1 + len2

    return tmpnqubit, len1, len2, first, second

# Function for temporaly result of ADVANCE operation, because the ADVANCE operations it's a more BASE operations
def collass(qc, measure, cl, nqubit):

    # Measure qubits
    for i in range(nqubit + 1):
        qc.measure(measure[i], cl[i])

    # Get results of program
    job_stats = execute(qc, backend=Aer.get_backend('qasm_simulator'), shots=100).result().get_counts()

    for key, value in job_stats.items():
        tmp = key
        prob = value

    # return temporaly result and probability
    return tmp, prob