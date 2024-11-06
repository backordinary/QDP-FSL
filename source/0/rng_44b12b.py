# https://github.com/LivingTheCoderDream/Quantum-Computing/blob/574a9bc6a5a0e4aba6570884ff9e654d85b04dd5/RNG.py
from qiskit import QuantumCircuit, execute, Aer
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram

sim = Aer.get_backend('statevector_simulator')


def how_many_qubits():
    print("This is a Random Number Generator")

    while True:


        try:
            n = qubits()
            if n > 10:
                print("The number of qubits should be 10 or less. Try again!")
            elif n < 0:
                print("You have entered a negative number. Try again!")
            elif n == 0:
                print("You must use at least 1 qubit. Try again!")
            else:
                return n
                break
        except:
            print("Invalid entry. Try again!")

def qubits():
    number_of_qubits = int(input("\nEnter the number of qubits you wish to use: "))
    return number_of_qubits

print(how_many_qubits())

def binary_to_decimal(b):
    b_string = str(b)
    len_b = len(b_string)
    s_empty = ""
    while len_b > 0:
        b_reverse = (b_string[len_b-1])
        s_empty = s_empty + b_reverse
        len_b = len_b - 1

    tot_add = 0
    hei = len(s_empty)
    i = 0
    while i < hei:
        p = int(b_string[i])*(2**i)
        tot_add = tot_add + p
        i = i + 1
    return tot_add
  
def get_binary_from_qubits(qc):
    counts = sim_result.get_counts(qc)
    return list(counts.keys())[0].split(' ')[0]

  
num_of_qubits = how_many_qubits()

qc = QuantumCircuit(num_of_qubits,num_of_qubits)

qc.h(range(num_of_qubits))
qc.measure_all()
qc.draw('mpl')

  
 
