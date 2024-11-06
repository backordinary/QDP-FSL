# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/ibm-challange/final1-cmp.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer

""" 



"""


def cmp(a, b):
    q = QuantumRegister(6)
    c = ClassicalRegister(6)
    circ = QuantumCircuit()
    circ.add_register(q)
    circ.add_register(c)

    # init
    if '1' == a[0]:
        circ.x(q[0])
    if '1' == a[1]:
        circ.x(q[1])
    if '1' == b[0]:
        circ.x(q[2])
    if '1' == b[1]:
        circ.x(q[3])

    # compare into the second vertice
    circ.cx(q[0], q[2])
    circ.cx(q[1], q[3])
    circ.cx(q[2], q[4])
    circ.cx(q[3], q[4])  #q[4] = 0 the same, 1 odd
    #OR if there was at least one adjenct
    circ.x(q[4])
    circ.cx(q[4], q[5]) #q[5] - if there were at least one the same adjenct chains q5 will by 1


    # revert scecond vertice
    circ.cx(q[1], q[3])
    circ.cx(q[0], q[2])

    circ.measure(q, c)

    # run
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circ, backend=backend, shots=1024)

    print(a+b)
    print(job.result().get_counts())


if __name__ == '__main__':
    print('The same')
    cmp('00', '00')
    cmp('01', '01')
    cmp('10', '10')
    cmp('11', '11')

    print('Odd')
    cmp('00', '01')
    cmp('11', '01')
    cmp('10', '00')
    cmp('11', '10')
