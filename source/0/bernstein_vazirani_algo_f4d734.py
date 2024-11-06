# https://github.com/Samikmalhotra/Quantum-Computing/blob/00427b7bdf518de5e50848f35745dd75c5fea8a6/Bernstein-Vazirani-Algorithm/bernstein-vazirani-algo.py
# %%
from qiskit import *
from qiskit import circuit
from qiskit.visualization import plot_histogram
from matplotlib import *

# %%
secret_number = '1010010111001011'

# %%
circuit = QuantumCircuit(len(secret_number)+1, len(secret_number))

# %%
circuit.h([i for i in range(len(secret_number))])
circuit.x(len(secret_number))
circuit.h(len(secret_number))

circuit.barrier()
circuit.draw(output='mpl')

# %%
# circuit.cx(0, 7)
# circuit.cx(2, 7)
# circuit.cx(6, 7)

reverse_number = secret_number[:: -1]
for index, one in enumerate(reverse_number):
    print(f"index {index} is {one}")
    if one == "1":
        circuit.cx(index, len(secret_number))

circuit.barrier()
circuit.draw(output='mpl')

# %%
circuit.h([i for i in range(len(secret_number))])
circuit.barrier()
circuit.measure([i for i in range(len(secret_number))],
                [i for i in range(len(secret_number))])
circuit.draw(output='mpl')

# %%
simulator = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend=simulator, shots=1).result()
counts = result.get_counts()
print(counts)

# %%
