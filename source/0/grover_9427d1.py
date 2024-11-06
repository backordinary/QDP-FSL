# https://github.com/ole-knf/Einsatz-von-Quantencomputern-zur-Abschwaechung-klassischer-Verfahren-der-Kryptographie/blob/15c0cd5613cf8096dcdb64abd027399d0339f81a/Grover.py
# The following code is based on https://qiskit.org/documentation/tutorials/algorithms/06_grover.html
from qiskit import QuantumCircuit
from qiskit import Aer
from qiskit.algorithms import AmplificationProblem
from qiskit.algorithms import Grover


# A function which uses Grover's algorithm to find the state "11" out of the four possible states "00","01","10","11"
def find_11_with_grover():
    # Mark "11" as the good state
    w = ['11']
    # Create an oracle which only returns true for the good state
    oracle = QuantumCircuit(2)
    oracle.cz(0, 1)
    # Define the amplification problem with the good state and the corresponding oracle
    problem = AmplificationProblem(oracle, is_good_state=w)
    # Create a backend which will simulate Grover's algorithm
    aer_simulator = Aer.get_backend('aer_simulator')
    # Copy the quantum circuit of Grover's algorithm
    grover = Grover(quantum_instance=aer_simulator)
    # Use the quantum circuit to solve the defined problem
    result = grover.amplify(problem)
    
    # Output the result if the simulation was successful
    print('Success!' if result.oracle_evaluation else 'Failure!')
    print('Top measurement:', result.top_measurement)

# Execute the function above
find_11_with_grover()
