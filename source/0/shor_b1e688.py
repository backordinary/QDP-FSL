# https://github.com/whnbaek/QC2020/blob/256bd6889de01f117fe8aa7730d1bfcc0414f8bd/hw7/shor.py
from qiskit import(
    QuantumCircuit,
    execute,
    Aer)
from qiskit.visualization import plot_histogram
from qiskit.extensions import UnitaryGate
import numpy as np
import math
from fractions import Fraction

N = 21
n = 5 # ceil(log 21) = 5
m = 9 # ceil(log 21^2) = 9

def c_axmodN(Ux, x):
    circuit = QuantumCircuit(n)

    # append unitary gate
    circuit.unitary(UnitaryGate(Ux), range(n))

    # encapsulate
    circuit = circuit.to_gate()
    circuit.name = "a^%i mod %i" % (x, N)

    # make control bit
    circuit = circuit.control()

    return circuit

def qft_dagger(): # n bits
    circuit = QuantumCircuit(m)

    # qft_dagger == inverse of qft

    for i in range(m // 2):
        circuit.swap(i, m - i - 1)

    for i in range(m):
        for j in range(i):
            circuit.cp(-np.pi / float(2 ** (i - j)), j, i)
        circuit.h(i)
    
    circuit.name = "QFT†"
    
    return circuit

def shor_circuit(a):
    circuit = QuantumCircuit(m + n, m)
    # 0 ~ 2 * n - 1 : x
    # 2 * n ~ 3 * n - 1 : a^x mod N

    # hadamard
    for i in range(m):
        circuit.h(i)
    circuit.barrier()

    # unitary operator
    circuit.x(m) # initialize to 1
    bits = [i + m for i in range(n)]
    
    Ux = np.zeros((2 ** n, 2 ** n), dtype = "uint32")
    for j in range(N):
        i = (j * a) % N
        Ux[i, j] = 1
    for i in range(N, 2 ** n):
        Ux[i, i] = 1

    for i in range(m):
        circuit.append(c_axmodN(Ux, 2 ** i), [i] + bits)
        print("%i^%i unitary operator appended" % (a, 2 ** i))
        Ux = Ux @ Ux
    circuit.barrier()
    
    # qft_dagger
    circuit.append(qft_dagger(), range(m))
    print("QFT† appended")
    circuit.barrier()

    # measure
    circuit.measure(range(m), range(m))

    return circuit


if __name__ == '__main__':

    np.random.seed(1)
    backend = Aer.get_backend('qasm_simulator')

    while True:
        a = np.random.randint(2, N)
        while math.gcd(a, N) != 1:
            # Actually, if gcd(a, N) is not 1, then gcd(a, N) is already a factor of N and
            # we do not have to run quantum circuits. Considering purpose of this homework,
            # we pick another a and use it to run quantum circuits.
            a = np.random.randint(2, N)

        # Assured that gcd(a, N) == 1
        print("a = %i chosen" % (a))

        # make circuit
        print("making circuit")
        circuit = shor_circuit(a)

        print("running circuit")
        counts = execute(circuit, backend, shots = 1024, memory = True).result().get_counts()

        for output in counts:
            decimal = int(output, 2)
            phase = decimal / (2 ** m)
            frac = Fraction(phase).limit_denominator(N)
            # one of candidates of r
            r = frac.denominator

            # r should be even, (a ^ (r / 2) +- 1) % N != 0
            if r % 2 == 1 or ((a ** (r // 2) - 1) % N == 0 and (a ** (r // 2) + 1) % N == 0):
                continue

            guesses = [math.gcd(a ** (r // 2) - 1, N), math.gcd(a ** (r // 2) + 1, N)]
            for guess in guesses:
                if 2 <= guess and guess < N and N % guess == 0:
                    print("factor found: %i" % (guess))

                    draw = circuit.draw(output = 'mpl')
                    draw.savefig('shor_circuit.png', bbox_inches = 'tight')
                    exit(0)
