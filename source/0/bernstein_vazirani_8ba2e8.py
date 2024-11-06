# https://github.com/palmenros/quantum-computing/blob/d5c2a49a3b03c81a5eb588f959de47ba3f3194ea/bernstein_vazirani.py
import matplotlib.pyplot as plt
import qiskit.visualization
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.providers.basicaer import QasmSimulatorPy


# Hidden binary string
s = '10010'

# Length of the hidden string
n = len(s)

# Reverse string due to order
s = s[::-1]

q_reg = QuantumRegister(n+1, 'q')
c_reg = ClassicalRegister(n, 'c')
circ = QuantumCircuit(q_reg, c_reg)

# Initialization of last bit
circ.x(n)
circ.h(n)

for i in range(n):
    circ.h(q_reg[i])

circ.barrier()

# Oraculo para la funcion s

for i in range(n):
    if s[i] == '0':
        pass
        # circ.i(q_reg[i])
    elif s[i] == '1':
        circ.cx(q_reg[i], q_reg[n])
    else:
        raise Exception('s is not a binary string')

circ.barrier()

for i in range(n):
    circ.h(q_reg[i])

circ.barrier()

circ.measure(q_reg[:n], c_reg[:n])

circ.draw('mpl')

print(circ.qasm())

simulator = QasmSimulatorPy()
compiled_circuit = transpile(circ, simulator)

job = simulator.run(compiled_circuit, shots=1000)
result = job.result()

counts = result.get_counts(circ)

print("\nTotal count are:", counts)
qiskit.visualization.plot_histogram(counts)
plt.show()
