# https://github.com/tommasobruggi/Number-Guesser-Quantum-Algorithm/blob/8a60fd90b9df63c6221e39b19c1fefcdbbac9409/Bernstein-Vazirani%20Quantum%20Algorithm.py
from qiskit import *

# Input any natural number and transform it into binary

number = int(input("Type your secret natural number : "))
if number - int(number) != 0 or number < 0:
    print("Please type a natural number. These are defined as being 0 or a positive integer.")
    exit
else:

    secretnumber = "{0:b}".format(number)

    # For a secret number of length n
    # "n (n-1) ... 2 1 0" corresponding qubit labels order
    # Initialize the circuit and gates 

    circuit = QuantumCircuit(len(secretnumber)+1, len(secretnumber))

    circuit.h(range(len(secretnumber)))
    circuit.x(len(secretnumber))
    circuit.h(len(secretnumber))
    
    circuit.barrier()

    # Perform the algorithm
    
    for counter, value in enumerate(reversed(secretnumber)):
        if value == "1":
            circuit.cx(counter, len(secretnumber))

    circuit.barrier()
    
    circuit.h(range(len(secretnumber)))
    circuit.barrier()
    
    circuit.measure(range(len(secretnumber)), range(len(secretnumber)))
    
    # Draw the circuit for visual stimulus
    
    circuit.draw(output = "mpl")
    print(circuit)
    
    # Execute the simulation
    
    simulator = Aer.get_backend("qasm_simulator")
    result = execute(circuit, backend = simulator, shots = 1).result()
    
    # Determine the string predicted by the quantum algorithm in 1 try
    
    counts = result.get_counts()    
    print("The number you were thinking of was " + str(int(list(counts)[0], 2)) + ". This is " + str(list(counts)[0]) + " in binary. Thanks for playing!")