# https://github.com/awlaplace/Quantum/blob/8b846c8dfac66d08d043565297e8e3089d501190/Shor_factorization/Shor_factorization_35.py
## 35の素因数分解を量子アルゴリズム的に行うスクリプト
## 8^r ≡ 1 mod 35 を満たすrを探す，8以外の場合は特に c_xmod35 で構成する量子ゲートが変わる
## 英語のコメントは Qiskit Textbook を残したもの

from qiskit import QuantumCircuit, Aer, transpile, assemble
# from numpy.random import randint
from math import gcd
import math
import numpy as np
from fractions import Fraction


def c_xmod35(x: int, N: int, power: int) -> list:
    """Controlled multiplication by a mod 35"""
    gate_number = int(math.log2(N))
    U = QuantumCircuit(gate_number)        
    for iteration in range(power):
        U.swap(0, 1)
        U.swap(1, 2)
        U.swap(2, 3)
        U.cx(0, 1)
        U.cx(2, 0)
        U.cx(2, 3)
        U.cx(2, 4)
        U.cx(1, 0)
        U.cx(1, 4)
    U = U.to_gate()
    U.name = "%i^%i mod 35" % (x, power)
    c_U = U.control()
    
    return c_U


def qft_dagger(n: int) -> list:
    """量子離散Fourier変換"""
    qc = QuantumCircuit(n)
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFT†"

    return qc


def qpe_xmod35(x: int, N: int) -> int:
    """量子位相推定"""
    n_count = 3
    gate_number = int(math.log2(N))
    qc = QuantumCircuit(gate_number+n_count, n_count)
    for q in range(n_count):
        qc.h(q)     
    qc.x(3+n_count) 
    for q in range(n_count): 
        qc.append(c_xmod35(x, N, 2**q), 
                 [q] + [i+n_count for i in range(gate_number)])
    qc.append(qft_dagger(n_count), range(n_count)) 
    qc.measure(range(n_count), range(n_count))
    qasm_sim = Aer.get_backend('qasm_simulator')
    t_qc = transpile(qc, qasm_sim)
    obj = assemble(t_qc, shots=1)
    result = qasm_sim.run(assemble(t_qc), memory=True).result()
    readings = result.get_memory()
    print("Register Reading: " + readings[0])
    phase = int(readings[0],2)/(2**n_count)
    print("Corresponding Phase: %f" % phase)

    return phase


def main():
    N = 35
    factor_found = False
    attempt = 0
    while not factor_found:
        x = 8 # 本当は x = randint(2, N - 1)
        if gcd(x, N) == 1:
            attempt += 1
            print("\nAttempt %i:" % attempt)
            phase = qpe_xmod35(x, N) # Phase = s/r
            frac = Fraction(phase).limit_denominator(N) 
            r = frac.denominator
            if phase != 0 :
                guesses = [gcd(x**(r//2)-1, N), gcd(x**(r//2)+1, N)]
                print("Guessed Factors: %i and %i" % (guesses[0], guesses[1]))
                for guess in guesses:
                    if guess not in [1,N] and (N % guess) == 0: 
                        print("*** Non-trivial factor found: %i ***" % guess)
                        factor_found = True


if __name__ == "__main__":
    main()
