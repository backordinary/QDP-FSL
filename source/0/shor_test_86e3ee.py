# https://github.com/mine260309/qiskit_practice/blob/76c55f733874e2b0865b83104c9e8edea5c45978/shor-test.py
#!/usr/bin/env python

from qiskit import QuantumProgram
import unittest


def shorU(obc, qr, cr, controlBit):
    obc.cx(qr[controlBit], qr[4])
    obc.cx(qr[controlBit], qr[5])
    obc.cx(qr[controlBit], qr[6])
    obc.cx(qr[controlBit], qr[7])

    obc.ccx(qr[controlBit], qr[6], qr[5])
    obc.ccx(qr[controlBit], qr[5], qr[6])
    obc.ccx(qr[controlBit], qr[6], qr[5])
    obc.ccx(qr[controlBit], qr[5], qr[4])
    obc.ccx(qr[controlBit], qr[4], qr[5])
    obc.ccx(qr[controlBit], qr[5], qr[4])
    obc.ccx(qr[controlBit], qr[7], qr[4])
    obc.ccx(qr[controlBit], qr[4], qr[7])
    obc.ccx(qr[controlBit], qr[7], qr[4])


def shorTest():
    # N=15 and a=7
    # Let's find period of f(x) = a^x mod N
    Circuit = 'shorTest'

    # Create the quantum program
    qp = QuantumProgram()

    # Creating registers
    n_qubits = 8
    qr = qp.create_quantum_register("qr", n_qubits)
    cr = qp.create_classical_register("cr", n_qubits)

    # We are going to find Q=2^q, where N^2 <= Q < 2*N^2
    # So the max Q is 450, needing 9 bits?
    
    # Shor algorithm with 4 qbits, where:
    # qr[0-3] are the bits for Q
    # qr[4-7] are the bits for U-gates
    obc = qp.create_circuit(Circuit, [qr], [cr])

    # Prepare bits
    obc.h(qr[0])
    obc.h(qr[1])
    obc.h(qr[2])
    obc.h(qr[3])

    # U0
    # U-a^2^0: multi 7*1
    # Refer to https://github.com/QISKit/ibmqx-user-guides/blob/master/rst/full-user-guide/004-Quantum_Algorithms/110-Shor's_algorithm.rst
    obc.x(qr[4])
    shorU(obc, qr, cr, 0)
    
    # U1
    shorU(obc, qr, cr, 1)
    shorU(obc, qr, cr, 1)

    # U2
    shorU(obc, qr, cr, 2)
    shorU(obc, qr, cr, 2)
    shorU(obc, qr, cr, 2)
    shorU(obc, qr, cr, 2)

    # U3
    shorU(obc, qr, cr, 3)
    shorU(obc, qr, cr, 3)
    shorU(obc, qr, cr, 3)
    shorU(obc, qr, cr, 3)
    shorU(obc, qr, cr, 3)
    shorU(obc, qr, cr, 3)
    shorU(obc, qr, cr, 3)
    shorU(obc, qr, cr, 3)
    obc.measure(qr[0], cr[0])
    obc.measure(qr[1], cr[1])
    obc.measure(qr[2], cr[2])
    obc.measure(qr[3], cr[3])
    obc.measure(qr[4], cr[4])
    obc.measure(qr[5], cr[5])
    obc.measure(qr[6], cr[6])
    obc.measure(qr[7], cr[7])

    # Get qasm source
    source = qp.get_qasm(Circuit)
    print(source)

    # Compile and run
    backend = 'local_qasm_simulator'
    circuits = [Circuit]  # Group of circuits to execute

    qobj = qp.compile(circuits, backend, shots = 32)  # Compile your program

    result = qp.run(qobj, wait=2, timeout=240)
    print(result)

    results = result.get_counts(Circuit)
    print(results)
    validate(results)


def validate(results):
    from collections import defaultdict
    import math
    d = defaultdict(set)
    for r in results:
        x = int(r[4:8], 2)
        u = int(r[1:4], 2)
        print("U %d : X %d" %(u, x))
        d[u].add(x)
    r =  {i:sorted(d[i]) for i in d.keys()}
    print(r)

    # Let's assume the period is 4
    # TODO: this period shall be calculated by quantum fourier transform
    N = 15
    r = 4
    p = math.gcd(int(7 ** (r / 2) - 1), N)
    q = math.gcd(int(7 ** (r / 2) + 1), N)
    print("p: %d, q: %d" %(p, q))


if __name__ == "__main__":
    shorTest()
