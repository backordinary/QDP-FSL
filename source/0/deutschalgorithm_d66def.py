# https://github.com/Wirkungsfunktional/QuantumInformationProcessing/blob/b3929f9a8e838e3dbda9feee5cf753d1204e75a1/QuantumGates/DeutschAlgorithm.py
from qiskit import QuantumProgram
import numpy as np

"""Program to perform the Deutsch Algorithm on an unknown (randomly selected)
one variable function f:{0, 1} -> {0, 1}. The Algorithm determine whether the
function is constant or balanced. """


qp = QuantumProgram()
qr = qp.create_quantum_register('qr',2)
cr = qp.create_classical_register('cr',2)
qc = qp.create_circuit('Deutsch',[qr],[cr])



def U(qc, qr, opt):
    """Defines the gates to perform the 4 options of f:{0, 1} \to {0, 1} as:
        1: constant
        2: balanced
        3: constant
        4: balanced
    """
    if opt == 1:
        return
    if opt == 2:
        qc.cx(qr[0], qr[1])
    if opt == 3:
        qc.cx(qr[0], qr[1])
        qc.x(qr[1])
    if opt == 4:
        qc.x(qr[1])


# Set the initial state |10>
qc.x(qr[1])

# Random choosen function
flag = np.random.randint(1, 5)



# Implement the Algorithm
qc.h(qr[0])
qc.h(qr[1])
U(qc, qr, flag)
qc.h(qr[0])
qc.measure(qr[0], cr[0])
result = qp.execute('Deutsch')


# Output
print("Random selected Function is: ")
if (flag == 1 or flag == 4):
    print("constant f" + str(flag))
else:
    print("balanced f" + str(flag))
print("Algorithm gives: ")
res = result.get_counts('Deutsch')
print(res)
if ('01' in res):
    print("balanced")
if ('00' in res):
    print("constant")
