# https://github.com/Yanaishaviv/AvodatQS/blob/ed993a8e896ad7ffb84a9494c4c7a456e05f6f04/introqcqh-lab-3/phase_est.py
from IPython.display import clear_output
import numpy as np
pi = np.pi
from qiskit.circuit.library import QFT
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import operator

clear_output()

def initialize_qubits(given_circuit, measurement_qubits, target_qubit):
    # move all the qubits to |-> state
    given_circuit.h(measurement_qubits)
    # except the controlled one, to |1>
    given_circuit.x(target_qubit)

def unitary_operator(given_circuit, control_qubit, target_qubit, theta):
    # create a basic unitary operator
    given_circuit.cp(2*pi*theta, control_qubit, target_qubit)

def unitary_operator_exponent(given_circuit, control_qubit, target_qubit, theta, exponent):
    # create a unitary operator which takes into account the exponent 
    given_circuit.cp(2*pi*theta*exponent, control_qubit, target_qubit)

def apply_iqft(given_circuit, measurement_qubits, n):
    # applt the quantum fourier transform on all measurement qubits
    given_circuit.append(QFT(n).inverse(), measurement_qubits)



def qpe_program(n, theta):
    
    # Create a quantum circuit on n+1 qubits (n measurement, 1 target)
    qc = QuantumCircuit(n+1, n)
    
    # Initialize the qubits
    initialize_qubits(qc, range(n), n)
    
    # Apply the controlled unitary operators in sequence
    for x in range(n):
        exponent = 2**(n-x-1)
        unitary_operator_exponent(qc, x, n, theta, exponent)
        
    # Apply the inverse quantum Fourier transform
    apply_iqft(qc, range(n), n)
    
    # Measure all qubits
    qc.measure(range(n), range(n))
  
    return qc


def est(n, theta):
    n = 24; theta = 0.5
    mycircuit = qpe_program(n, theta)
    simulator = Aer.get_backend('qasm_simulator')
    counts = execute(mycircuit, backend=simulator).result().get_counts(mycircuit)
    highest_probability_outcome = max(counts.items(), key = operator.itemgetter(1))[0][::-1]
    measured_theta = int(highest_probability_outcome, 2)/2**n
    print("Using %d qubits with theta = %.2f, measured_theta = %.2f." % (n, theta, measured_theta))


def main():
    n = 20; theta = 0.5
    for i in range(32):
        est(n, i)

if __name__ == '__main__':
    print(QFT(4).draw())
    main()
