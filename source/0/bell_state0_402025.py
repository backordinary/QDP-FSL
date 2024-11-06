# https://github.com/rickapocalypse/final_paper_qiskit_sat/blob/bfd57cca11bdd3c70afb294bc74ed3e8ade27fa0/bell_state0.py
from qiskit import*
from qiskit import circuit
import matplotlib.pyplot as plt
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.visualization import plot_bloch_multivector
qr = QuantumRegister(2)   #Registrador quântico de 2bits
cr = ClassicalRegister(2) #vai ser utilizado para fazer medições 

circuit = QuantumCircuit(qr,cr)

circuit.h(qr[0])

circuit.cx(qr[0],qr[1])
circuit.measure(qr,cr)

simulator = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend = simulator).result()
plot_histogram(result.get_counts(circuit))
plt.show()

simulator = Aer.get_backend('statevector_simulator')
result = execute(circuit, backend = simulator).result()
statevector = result.get_statevector()
plot_bloch_multivector(statevector)
plt.show()