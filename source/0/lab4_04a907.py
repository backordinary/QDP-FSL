# https://github.com/unimauro/qiskit-summer-2020/blob/21329ac62bf9922fa6ccb391a56c84a033429020/Laboratorio%204/lab4.py
def initialize_qubits(given_circuit, n, m):
    given_circuit.h(range(n))
    given_circuit.x(n+m-1)
def a_x_mod15(a, x):
    if a not in [2,7,8,11,13]:
        raise ValueError("'a' must be 2,7,8,11 or 13")
    U = QuantumCircuit(4)        
    for iteration in range(x):
        if a in [2,13]:
            U.swap(0,1)
            U.swap(1,2)
            U.swap(2,3)
        if a in [7,8]:
            U.swap(2,3)
            U.swap(1,2)
            U.swap(0,1)
        if a == 11:
            U.swap(1,3)
            U.swap(0,2)
        if a in [7,11,13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = "%i^%i mod 15" % (a, x)
    c_U = U.control()
    return c_U
def modular_exponentiation(given_circuit, n, m, a):
    for i in range(n):
        given_circuit.append(a_x_mod15(a,2**(n-i-1)),[i]+[i+n for i in range(n)])
    for qubit in range(n//2):
        given_circuit.swap(qubit,n-qubit-1)
    given_circuit.barrier()
    
def apply_iqft(given_circuit, measurement_qubits):
    n=np.size(measurement_qubits)
    given_circuit.append(QFT(n,do_swaps=False,inverse=True,name="iqft"),range(n))

def shor_program(n, m, a):
    # set up quantum circuit
    shor = QuantumCircuit(n+m, n)
    # initialize the qubits
    initialize_qubits(shor, n, m)
    shor.barrier()
    # apply modular exponentiation
    modular_exponentiation(shor, n, m, a)
    shor.draw(output="text")
    shor.barrier()
    # apply inverse QFT
    apply_iqft(shor, range(n))
    # measure the first n qubits
    shor.measure(range(n), range(n))
    return shor

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram
from qiskit import Aer, execute
import numpy as np
from math import gcd
n = 4; m = 4; a = 7
mycircuit = shor_program(n, m, a)
mycircuit.draw(output='text')
simulator = Aer.get_backend('qasm_simulator')
counts = execute(mycircuit, backend=simulator, shots=1000).result().get_counts(mycircuit)
plot_histogram(counts)
for measured_value in counts:
    print(f"Measured {int(measured_value[::-1], 2)}")
for measured_value in counts:
    measured_value_decimal = int(measured_value[::-1], 2)
    print(f"Measured {measured_value_decimal}")
    if measured_value_decimal % 2 != 0:
        print("Failed. Measured value is not an even number")
        continue
    x = int((a ** (measured_value_decimal/2)) % 15)
    if (x + 1) % 15 == 0:
        print("Failed. x + 1 = 0 (mod N) where x = a^(r/2) (mod N)")
        continue
    guesses = gcd(x + 1, 15), gcd(x - 1, 15)
    print(guesses)