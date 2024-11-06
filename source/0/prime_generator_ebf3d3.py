# https://github.com/nenad-popovic-91/prime-qiskit-masters/blob/16357b99f2d9d012fdf078099f0b4236bae0ba8d/prime_generator.py
from qiskit import QuantumCircuit, Aer, execute
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import circuit_provider


n = 6

# kreiranje kola eksperimenta
grover = QuantumCircuit(n)
grover.x(0)
grover.h(range(1, n-1))
grover.x(n-1)
P55 = circuit_provider.oracle_prime_55().to_gate()
P55.label = 'P55'
D4 = circuit_provider.grover_diffusion_operator(n-2).to_gate()
D4.label = 'D4'
grover.append(P55, range(n))
grover.append(D4, range(1, n-1))
grover.measure_all()

# prikaz kreiranog kola
grover.draw(output='mpl', style={"name": "bw", "dpi": 300})

# izvrsavanje 100 hiljada iteracija eksperimenta
# na (lokalnom) kvantnom simulatoru
backend = Aer.get_backend('qasm_simulator')
counts = execute(grover, backend, shots=100000).result().get_counts()
plot_histogram(counts, color='gray')

plt.show()