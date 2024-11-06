# https://github.com/tigerjack/qiskit-tests/blob/b22361a5fbce84639a0019fa32edfba452ef8384/src/grover_2.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute, Aer
from qiskit.tools.visualization import circuit_drawer
import sys


# Grover w/ just 2 qubits
def grover(x_star):
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)
    qc = QuantumCircuit(qr, cr)
    ## uniform superpositiom
    qc.h(qr)

    ## It suffices to apply the subroutine (oracle + diffusion) just one time
    oracle(qc, qr, x_star)
    #grover_diffusion(qc, qr)
    grover_diffusion2(qc, qr)

    qc.measure(qr, cr)
    circuit_drawer(qc, filename="imgs/grover_2.png")

    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend)
    result = job.result()
    print(result.get_counts(qc))


def grover_diffusion(qc, qr):
    qc.h(qr)
    # reflection about 0, i.e. negating the amplitude of all the states aside
    # from 0 and leaving 0 unchanged

    # negate amplitude on 11
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[1])

    # negate 10
    qc.x(qr[0])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[1])
    qc.x(qr[0])

    # negate 01
    qc.x(qr[1])
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[1])
    qc.x(qr[1])

    # Final hadamards
    qc.h(qr)


def grover_diffusion2(qc, qr):
    qc.h(qr)
    # reflection about 0, i.e. negating the amplitude of all the states aside
    # from 0 and leaving 0 unchanged
    # Bcz in the end the probabilities don't change for a + or - sign in front
    # of the amplitudes, it's the same to just negate 00 and leave the other
    # states unchanged. This should be more efficient.

    # negate amplitude on 00
    qc.x(qr)
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[1])
    qc.x(qr)

    # Final hadamards
    qc.h(qr)


# mark the state 11 as our special state by default, see lec.29
def oracle(qc, qr, x_star=3):
    # mark the state 01
    if (x_star > 3):
        raise ValueError("This algorithm works only w/ x_star b/w 0 and 3")
    print("x* is equal to {0}".format(x_star))
    if (x_star == 1):
        print("special state 01")
        qc.h(qr[0])
        qc.x(qr[1])
        qc.cx(qr[1], qr[0])
        qc.h(qr[0])
        qc.x(qr[1])
        return

    if (x_star == 2):
        print("special state 10")
        qc.x(qr[0])
    if (x_star == 0):
        print("special state 00")
        qc.x(qr[0])
        qc.x(qr[1])
    print("common part")
    qc.h(qr[1])
    qc.cx(qr[0], qr[1])
    qc.h(qr[1])
    if (x_star == 2):
        print("special state 10")
        qc.x(qr[0])
    if (x_star == 0):
        print("special state 00")
        qc.x(qr[0])
        qc.x(qr[1])


def main():
    grover(int(sys.argv[1]))


if __name__ == "__main__":
    main()
