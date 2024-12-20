# https://github.com/tigerjack/isd-quantum/blob/33ee3b6e99530da358de6dcefdac9cebd3a93b83/experiments/permutations_methods_ex.py
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister


def preparing_initial_states(n, r):
    if (n == 7):
        nn = 8
    else:
        nn = n
    qr = QuantumRegister(nn)
    global cr
    cr = ClassicalRegister(nn)
    qc = QuantumCircuit(qr, cr)

    for i in range(r):
        qc.x(qr[i])
    qc.barrier()

    return qr, cr, qc


# OK
# For all cases in which log(n, 2) is an int
# In reality, for now, it's just for n = 8, but this could be easily generalized
def permutation_6(qr, qc, n):
    qrs = qr
    ancillas = QuantumRegister(20)
    qc.add(ancillas)
    qc.h(ancillas)
    qc.barrier()
    last2 = round(n / 2)  # 4
    last4 = round(n / 4)  # 2
    control = 0

    for i in range(n - last2):
        print("cswapping {0} {1}".format(i, i + last2))
        qc.cswap(ancillas[control], qrs[i], qrs[i + last2])
        control += 1
    print("*******")
    qc.barrier()

    for i in range(0, n - last4, last2):
        print("cswapping {0} {1}".format(i, i + last4))
        qc.cswap(ancillas[control], qrs[i], qrs[i + last4])
        control += 1
        print("cswapping {0} {1}".format(i + 1, i + 1 + last4))
        qc.cswap(ancillas[control], qrs[i + 1], qrs[i + 1 + last4])
        control += 1
    print("*******")
    qc.barrier()

    for i in range(0, n - 1, 2):
        print("cswapping {0} {1}".format(i, i + 1))
        qc.cswap(ancillas[control], qrs[i], qrs[i + 1])
        control += 1
    qc.barrier()
    print("*******")

    for i in range(0, n - last4, last2):
        print("cswapping {0} {1}".format(i, i + last4))
        qc.cswap(ancillas[control], qrs[i], qrs[i + last4])
        control += 1
        print("cswapping {0} {1}".format(i + 1, i + 1 + last4))
        qc.cswap(ancillas[control], qrs[i + 1], qrs[i + 1 + last4])
        control += 1
    print("*******")
    qc.barrier()

    for i in range(n - last2):
        print("cswapping {0} {1}".format(i, i + last2))
        qc.cswap(ancillas[control], qrs[i], qrs[i + last2])
        control += 1
    print("*******")
    qc.barrier()
    qc.measure(qr, cr)


#MEH
# Just for n = 7
def permutation_7(qr, qc, n):
    qrs = qr
    ancillas = QuantumRegister(20)
    qc.add(ancillas)
    qc.h(ancillas)
    qc.barrier()
    last2 = round(n / 2)  # 4
    last4 = round(n / 4)  # 2
    control = 0

    for i in range(n - last2):
        print("cswapping {0} {1}".format(i, i + last2))
        qc.cswap(ancillas[control], qrs[i], qrs[i + last2])
        control += 1
    print("*******")
    qc.barrier()

    # Just a random test
    print("Random swapping 3 5")
    qc.cswap(ancillas[control], qrs[3], qrs[5])
    control += 1

    for i in range(0, n - last4, last2):
        print("cswapping {0} {1}".format(i, i + last4))
        qc.cswap(ancillas[control], qrs[i], qrs[i + last4])
        control += 1
        if not (i == 4):
            print("cswapping {0} {1}".format(i + 1, i + 1 + last4))
            qc.cswap(ancillas[control], qrs[i + 1], qrs[i + 1 + last4])
            control += 1
    print("*******")
    qc.barrier()

    # Just a random test
    print("Random swapping 5 6")
    qc.cswap(ancillas[control], qrs[5], qrs[6])
    control += 1
    qc.barrier()

    for i in range(0, n - 1, 2):
        print("cswapping {0} {1}".format(i, i + 1))
        qc.cswap(ancillas[control], qrs[i], qrs[i + 1])
        control += 1
    qc.barrier()
    print("*******")

    # Just a random test
    print("Random swapping 3 5")
    qc.cswap(ancillas[control], qrs[3], qrs[5])
    control += 1
    qc.barrier()

    for i in range(0, n - last4, last2):
        print("cswapping {0} {1}".format(i, i + last4))
        qc.cswap(ancillas[control], qrs[i], qrs[i + last4])
        control += 1
        if not (i == 4):
            print("cswapping {0} {1}".format(i + 1, i + 1 + last4))
            qc.cswap(ancillas[control], qrs[i + 1], qrs[i + 1 + last4])
            control += 1
    print("*******")
    qc.barrier()

    for i in range(n - last2):
        print("cswapping {0} {1}".format(i, i + last2))
        qc.cswap(ancillas[control], qrs[i], qrs[i + last2])
        control += 1
    print("*******")
    qc.barrier()
    qc.measure(qr, cr)


# OK
# For all cases in which log(n, 2) is an int
def permutation_8(qr, qc, n):
    if (n == 7):
        nn = 8
    else:
        nn = n
    qrs = qr

    ancillas = QuantumRegister(nn)
    qc.add(ancillas)
    qc.h(ancillas)
    qc.barrier()
    last2 = round(nn / 2)  # 4
    last4 = round(nn / 4)  # 2
    control = 0

    for i in range(nn - last2):
        print("cswapping {0} {1}".format(i, i + last2))
        qc.cswap(ancillas[control], qrs[i], qrs[i + last2])
        control += 1
    print("*******")
    qc.barrier()

    for i in range(0, nn - last4, last2):
        print("cswapping {0} {1}".format(i, i + last4))
        qc.cswap(ancillas[control], qrs[i], qrs[i + last4])
        control += 1
        print("cswapping {0} {1}".format(i + 1, i + 1 + last4))
        qc.cswap(ancillas[control], qrs[i + 1], qrs[i + 1 + last4])
        control += 1
    print("*******")
    qc.barrier()

    for i in range(0, nn - 1, 2):
        print("cswapping {0} {1}".format(i, i + 1))
        qc.cswap(ancillas[control], qrs[i], qrs[i + 1])
        control += 1
    qc.barrier()
    print("*******")

    for i in range(0, nn - last4, last2):
        print("cswapping {0} {1}".format(i, i + last4))
        qc.cswap(ancillas[control], qrs[i], qrs[i + last4])
        control += 1
        print("cswapping {0} {1}".format(i + 1, i + 1 + last4))
        qc.cswap(ancillas[control], qrs[i + 1], qrs[i + 1 + last4])
        control += 1
    print("*******")
    qc.barrier()

    for i in range(nn - last2):
        print("cswapping {0} {1}".format(i, i + last2))
        qc.cswap(ancillas[control], qrs[i], qrs[i + last2])
        control += 1
    print("*******")
    qc.barrier()

    # Restore ancillas
    qc.h(ancillas)
    if (n == 7):
        # If last bit is 1, we can't use the value
        # Else we copy all the values in the ancillas
        qc.x(qr[7])
        qc.ccx(qr[7], qr[0], ancillas[0])
        qc.measure(ancillas[0], cr[0])
        qc.ccx(qr[7], qr[1], ancillas[1])
        qc.measure(ancillas[1], cr[1])
        qc.ccx(qr[7], qr[2], ancillas[2])
        qc.measure(ancillas[2], cr[2])
        qc.ccx(qr[7], qr[3], ancillas[3])
        qc.measure(ancillas[3], cr[3])
        qc.ccx(qr[7], qr[4], ancillas[4])
        qc.measure(ancillas[4], cr[4])
        qc.ccx(qr[7], qr[5], ancillas[5])
        qc.measure(ancillas[5], cr[5])
        qc.ccx(qr[7], qr[6], ancillas[6])
        qc.measure(ancillas[6], cr[6])


# OK
# For all cases in which log(n, 2) is an int
# In reality, for now, it's just for n = 8, but this could be easily generalized
# Testing using just one hadamard.
# TODO discard the result of measurements of ancilla qubit
def permutation_9(qr, qc, n):
    qrs = qr
    ancillas = QuantumRegister(1)
    ancillas_c = ClassicalRegister(1)
    qc.add(ancillas)
    qc.add(ancillas_c)
    qc.h(ancillas)
    qc.barrier()
    last2 = round(n / 2)  # 4
    last4 = round(n / 4)  # 2

    for i in range(n - last2):
        print("cswapping {0} {1}".format(i, i + last2))
        qc.cswap(ancillas[0], qrs[i], qrs[i + last2])
        qc.measure(ancillas, ancillas_c)
        qc.h(ancillas)
    print("*******")
    qc.barrier()

    for i in range(0, n - last4, last2):
        print("cswapping {0} {1}".format(i, i + last4))
        qc.cswap(ancillas[0], qrs[i], qrs[i + last4])
        qc.measure(ancillas, ancillas_c)
        qc.h(ancillas)
        print("cswapping {0} {1}".format(i + 1, i + 1 + last4))
        qc.cswap(ancillas[0], qrs[i + 1], qrs[i + 1 + last4])
        qc.measure(ancillas, ancillas_c)
        qc.h(ancillas)
    print("*******")
    qc.barrier()

    for i in range(0, n - 1, 2):
        print("cswapping {0} {1}".format(i, i + 1))
        qc.cswap(ancillas[0], qrs[i], qrs[i + 1])
        qc.measure(ancillas, ancillas_c)
        qc.h(ancillas)
    print("*******")
    qc.barrier()

    for i in range(0, n - last4, last2):
        print("cswapping {0} {1}".format(i, i + last4))
        qc.cswap(ancillas[0], qrs[i], qrs[i + last4])
        qc.measure(ancillas, ancillas_c)
        qc.h(ancillas)
        print("cswapping {0} {1}".format(i + 1, i + 1 + last4))
        qc.cswap(ancillas[0], qrs[i + 1], qrs[i + 1 + last4])
        qc.measure(ancillas, ancillas_c)
        qc.h(ancillas)
    print("*******")
    qc.barrier()

    for i in range(n - last2):
        print("cswapping {0} {1}".format(i, i + last2))
        qc.cswap(ancillas[0], qrs[i], qrs[i + last2])
        qc.measure(ancillas, ancillas_c)
        qc.h(ancillas)
    print("*******")
    qc.reset(ancillas)
    qc.barrier()

    qc.measure(qr, cr)


def main():
    n = 8
    r = 3
    qr, cr, qc = preparing_initial_states(n, r)
    from os import sys
    choice = int(sys.argv[1])
    if choice == 6:
        permutation_6(qr, qc, n)
    elif choice == 7:
        permutation_7(qr, qc, n)
    elif choice == 8:
        permutation_8(qr, qc, n)
    elif choice == 9:
        permutation_9(qr, qc, n)
    else:
        print("Error choice")
        return
    print("Choice", choice)

    print("Drawing")
    from qiskit.tools.visualization import circuit_drawer
    circuit_drawer(qc, filename='img/test_{0}.png'.format(choice))

    print("Preparing execution")
    # from qiskit import Aer
    # backend = Aer.get_backend('qasm_simulator')

    from qiskit import IBMQ
    IBMQ.load_accounts()
    backend = IBMQ.get_backend('ibmq_qasm_simulator')

    from qiskit import execute
    print("Execute")
    job = execute(qc, backend, shots=4098)
    print(job.job_id())
    result = job.result()
    print("Results ready")
    counts = result.get_counts(qc)
    print(counts)
    print(len(counts))

    from qiskit.tools.visualization import plot_histogram
    plot_histogram(counts, )


if __name__ == "__main__":
    main()
