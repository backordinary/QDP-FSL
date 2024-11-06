# https://github.com/samgodwood/QEC-Notebooks/blob/2726e5754b80488f680181318d7d68e18fd4b96b/bit_flip.py
# ------------------------------------------------------------------
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from qiskit_aer.noise import pauli_error
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Hard-coded constants
random.seed(42)     # For reproducibility
Nshots = 1000

q = QuantumRegister(5)              # Create 5-qubit quantum register
c = ClassicalRegister(2)            # Create 2-bit classical register
cr = ClassicalRegister(3)           # Create 3-bit classical register
simulator = Aer.get_backend('qasm_simulator')   # Choose simulator

error_probs = np.arange(0, 0.101, 0.001)
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Subroutine to loop over pauli_error probabilities
# Argument 'init' chooses between |0> and |+> initial state
def scan_probs(init):
    success_rates = []
    uncorr_rates = []
    for p_err in error_probs:
        bit_flip = QuantumCircuit(q, c)     # Create quantum circuit

        # Prepares qubit in the desired initial state
        # Encodes the qubit in a three-qubit entangled state
        if init == 'plus':
            bit_flip.h(q[0])
        bit_flip.cx(q[0], q[1])
        bit_flip.cx(q[0], q[2])

        # Construct a 1-qubit bit-flip error
        error = pauli_error([('X', 0.5 * p_err),
                             ('Z', 0.5 * p_err),
                             ('I', 1.0 - p_err)])

        # Combine the circuit and the error channel
        for i in range(0,3):
            bit_flip.append(error, [i])

        # Entangle two ancilla qubits for error-correction
        bit_flip.cx(q[0], q[3])
        bit_flip.cx(q[1], q[3])
        bit_flip.cx(q[1], q[4])
        bit_flip.cx(q[2], q[4])

        # Measure the two ancilla qubits
        bit_flip.measure(q[3], c[0])
        bit_flip.measure(q[4], c[1])

        # Do error correction
        bit_flip.x(q[0]).c_if(c,1)
        bit_flip.x(q[1]).c_if(c,3)
        bit_flip.x(q[2]).c_if(c,2)

        if init == 'plus':
            # Decode the qubit from the three-qubit entangled state
            bit_flip.cx(q[0], q[1])
            bit_flip.cx(q[0], q[2])
            bit_flip.h(q[0])

        # Add the classical register to the quantum circuit
        bit_flip.add_register(cr)
        # Check the state of the initial qubit
        for i in range(0,3):
            bit_flip.measure(q[i], cr[i])

        # Reproducibly execute the circuit on the simulator
        seed = random.randrange(0, 1e6)
        result = execute(bit_flip, simulator, shots=Nshots,
                         seed_simulator=seed).result()

        # Get the counts from the result
        counts = result.get_counts()

        num_success = 0
        raw_success = 0
        counts_dict = dict(counts)
        for key, value in counts_dict.items():
            if key[:3] == '000':
                num_success += value
                if key == '000 00':     # Correct without correction
                    raw_success += value
        success_rates.append(num_success / float(Nshots))
        uncorr_rates.append(raw_success / float(Nshots))
        print("%.3g %.3g %.3g" % (p_err, success_rates[-1], uncorr_rates[-1]))

    return success_rates, uncorr_rates
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Main code
# Loop over |0> and |+> initial states
for state in ['zero', 'plus']:
    runtime = -time.time()
    if state == 'zero':
        print("Logical0 initial state")
    else:
        # Blank line in output to separate results
        print("\nLogical+ initial state")

    success_rates, uncorr_rates = scan_probs(state)

    # Print runtime here to ignore time spent looking at plot
    runtime += time.time()
    print("Runtime: %0.1f seconds" % runtime)

    if state == 'zero':
      title = 'Logical0 performance for 3 qubit bit flip code'
    else:
      title = 'Logical+ performance for 3 qubit bit flip code'

    # Plot the success rate as a function of the error probability
    # TODO: ADD UNCORRECTED VS CORRECTED...
    plt.plot(error_probs, success_rates, label='Corrected')
    plt.plot(error_probs, uncorr_rates, label='Uncorrected')
    plt.title(title)
    plt.xlabel('Error probability')
    plt.ylabel('Success rate')
    plt.legend(loc='lower left')
    plt.show()
# ------------------------------------------------------------------
