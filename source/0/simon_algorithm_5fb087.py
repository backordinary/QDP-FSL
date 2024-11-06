# https://github.com/Schwarf/qiskit_fundamentals/blob/178b80fae8cfebf2747b47ab707b93d73255aab8/quantum_circuits/simon_algorithm.py
from matplotlib import pyplot as plt
# importing Qiskit
from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit_textbook.tools import simon_oracle
from miscellaneous.misc import dot_product_bit_strings

def simon_circuit(binary_string: str) -> QuantumCircuit:
    n = len(binary_string)
    s_circuit = QuantumCircuit(2*n, n)
    s_circuit.h(range(n))
    s_circuit.barrier()
    s_circuit += simon_oracle(binary_string)
    s_circuit.barrier()
    s_circuit.h(range(n))
    # TODO: Shouldn't it be the second register to be measured?
    s_circuit.measure(range(n), range(n))
    return s_circuit

binary_string = "110"

simon_circuit = simon_circuit(binary_string)
simon_circuit.draw()

aer_sim = Aer.get_backend('aer_simulator')
shots = 1024
qobj = assemble(simon_circuit, shots=shots)
results = aer_sim.run(qobj).result()
counts = results.get_counts()
for qubits, probability in counts.items():
    result = dot_product_bit_strings(qubits, binary_string)
    print(f"Dot product (mod2): {qubits}.{binary_string} = {result}")
plot_histogram(counts)

plt.show()
