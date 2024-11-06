# https://github.com/ilverdo97/QuantumProject/blob/a6804438a8f68db52525b7a8e616eeafebaa7b96/operations/operationsADVANCE.py
import qiskit
from functions.utils import *
from functions.utilsQubit import *
from operations.operationsBASE import *

#DIVISION---------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
def div(first, second, dividend, divisor, qc, nqubit, cl):

    # result is the number of repeat the cicle
    result = 0

    # We dont know for than time we use the operation SUBTRACTION
    while True:

        # Use operation SUBTRACTION
        sub(dividend, divisor, qc)

        # Collass the circuit for temporaly result
        tmp, prob = collass(qc, dividend, cl, nqubit)

        numdividend = int(tmp, 2)
        numdivisor = int(second, 2)

        result = result + 1

        # If dividend is >= of divisor we repeat the operation another
        if numdividend >= numdivisor:

            # We dont have a circuit because we collass that,
            # we go a build a new circuit with the temporaly result of previer circuit
            numdividend = '{0:{fill}11b}'.format(numdividend, fill='0')
            numdivisor = '{0:{fill}11b}'.format(numdivisor, fill='0')

            len1 = len(numdividend)
            len2 = len(numdivisor)
            nqubit, len1, len2, newnum, second = NQubit("/", len1, len2, numdividend, numdivisor)

            dividend = qiskit.QuantumRegister(nqubit + 1, "a")
            divisor = qiskit.QuantumRegister(nqubit + 1, "b")
            cl = qiskit.ClassicalRegister(nqubit + 1, "cl")
            qc = qiskit.QuantumCircuit(dividend, divisor, cl, name="qc")

            # Numdividend is a temporaly result, we put in a register
            initQubits(numdividend, qc, dividend, nqubit)
            initQubits(numdivisor, qc, divisor, nqubit)

        else:
            break

    # return result for print
    return int(first, 2), numdivisor, result, prob


#EXPONENTIAL------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
def exponential(a, firstDecBinary, firstDec, secondDec, result, qc, cl, nqubit):

    # We go repeat a multilpy for the number of exponential
    for x in range(secondDec - 1):

        # firstDec is a temporaly result, firstDecBinary is a original number, secondDec ia a exponential
        multiply(a, firstDec, result, qc)

        # Collass the circuit for temporaly result
        tmp, prob = collass(qc, result, cl, nqubit)

        firstDec = int(tmp, 2)

        # If x is < of exponential we repeat the operation another
        if x < (secondDec - 2):

            # We dont have a circuit because we collass that,
            # we go a build a new circuit with the temporaly result of previer circuit
            a = qiskit.QuantumRegister(nqubit + 1, "a")
            b = qiskit.QuantumRegister(nqubit + 1, "b")
            cl = qiskit.ClassicalRegister(nqubit + 1, "cl")
            qc = qiskit.QuantumCircuit(a, b, cl, name="qc")
            initQubits(firstDecBinary, qc, a, nqubit)

    # Base case of exponential
    if secondDec == 0:
        firstDec = 1
        prob = 100
    elif secondDec == 1:
        firstDec = int(firstDecBinary, 2)
        prob = 100

    # return result for print
    return int(firstDecBinary, 2), secondDec, firstDec, prob
