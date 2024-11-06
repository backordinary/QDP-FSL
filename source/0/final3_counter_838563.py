# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/ibm-challange/final3-counter.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer

""" 
"""

q_output = QuantumRegister(3)
c_output = ClassicalRegister(3)
q_input = QuantumRegister(2)
q_sum = QuantumRegister(3)

circ = QuantumCircuit()
circ.add_register(q_output)
circ.add_register(c_output)
circ.add_register(q_input)
circ.add_register(q_sum)

S1, S2, C2 = q_output
A1, X1 = q_input

C1, A2, X2, = q_sum


def sum1():
    circ.cx(A1, S1)
    circ.cx(X1, S1)
    circ.ccx(A1, X1, C1)


def sum2():
    circ.cx(S1, S2)
    circ.cx(C1, S2)
    circ.ccx(S1, C1, C2)

    circ.cx(C2, X1)

if __name__ == '__main__':

    circ.x(A1) #+1
    sum1()
    # 001

    #circ.x(A1) #+1
    sum2()
    sum1()
    # 010

    #circ.x(A1) #+1
    sum2()
    sum1()
    # 011

    # circ.x(A1) #+1
    sum2()
    sum1()
    # 100


    # #circ.x(A1) #+1
    # sum1()
    # sum2()
    #
    # #circ.x(A1) #+1
    # sum1()
    # sum2()

    circ.measure(q_output, c_output)

    backend = Aer.get_backend('qasm_simulator')
    job = execute(circ, backend=backend, shots=1024)
    print(job.result().get_counts())
