# https://github.com/paniash/progs/blob/b7f58efeb5b5c7942af6b8af12611bdbc1a52840/python/half_adder.py
# To make a simple half adder circuit using qiskit
# The circuit aims to implement the following operations
# 0 + 0 = 00
# 0 + 1 = 01
# 1 + 0 = 01
# 1 + 1 = 10
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

qc = QuantumCircuit(4,2) # makes a circuit with 4 qubits (inputs) and 2 outputs
# Giving number ab as input where
qc.x(0)  # input b = 1
qc.x(1)  # input a = 1
qc.barrier()  # separates input stage and algorithms

# To give output cd where 'd' is calculated as
qc.cx(0,2) # controlled NOT gate with q_0 being control and q_2 being target
qc.cx(1,2) # similar implementation

# To provide output for 'c' using Toffoli gate (essentially an AND gate)
qc.ccx(0,1,3)
qc.barrier() # separates algorithm stage from output

# To measure the output and measure it from classical bits
qc.measure(2,0)  # measuring output 'd' from c0
qc.measure(3,1)  # measuring output 'c' from c1


# the circuit in all its glory
qc.draw(output='mpl')
print("Here's the circuit")
plt.show()

# to measure the output
qc.measure(0,0)
qc.measure(1,1)

simulator = Aer.get_backend('qasm_simulator') # specifying the backend (this changes when running on a quantum computer)

# execution
result = execute(qc,simulator).result()
counts = result.get_counts()

print("The sum is", counts)
