# https://github.com/paniash/progs/blob/b7f58efeb5b5c7942af6b8af12611bdbc1a52840/python/collapse_state.py
# Demonstrates that after measurement of a state which is in a superposition of
# two pure states, the original state collapses into one of the pure states.
# Here, the initial state has an equal probability of being measures as 0 or 1.

from qiskit import Aer, QuantumCircuit, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from math import sqrt

# Initializes a circuit with one qubit, which by default is in the 0 state
qc = QuantumCircuit(1)

# Define a mixed state with equal probability amplitude of 0 and 1 (normalized,
# otherwise it doesn't work)
init_state = [1.j/sqrt(2), 1/sqrt(2)]

# initializes the 0th qubit as the mixed state
qc.initialize(init_state, 0)

# Now the qubit is in a mixed state of 0 and 1 with equal probability and is
# normalized

# define a backend simulator on which to run the circuit through Aer
backend = Aer.get_backend('statevector_simulator')

# To check its current state
state = execute(qc, backend).result().get_statevector()
print("Qubit state =" + str(state))

# Now let's measure the state which forces a collapse of the mixed state into one of
# pure states

# Tells the circuit to make the circuit
qc.measure_all()

# executes the measurement
state = execute(qc, backend).result().get_statevector()
print("State of measured qubit =" + str(state))

# After measurement, the amplitude of one of the states will always be zero, i.e.,
# a collapse of the mixed state occurs on measurement and forces the state into a
# pure state of either 0 or 1 with an equal probability in this case.
