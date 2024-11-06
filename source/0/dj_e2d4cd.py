# https://github.com/albertye1/qiskit-basics/blob/79db5d28b2682c8e935d6345d34c22f227edd2a1/deutsch-josza/dj.py
import numpy as np
from qiskit import IBMQ, Aer 
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, assemble, transpile
import matplotlib.pyplot as plt

# creating the deutsch-josza oracle
def dj_oracle(case, n):
	oracle = QuantumCircuit(n+1)
	""" balanced """
	if case == 1 :
		b = np.random.randint(1, 2**n-1)
		b_str = format(b, '0'+str(n)+'b')
		for q in range(len(b_str)):
			if b_str[q] == '1':
				oracle.x(q)
		for q in range(n):
			oracle.cx(q, n)
		for q in range(len(b_str)):
			if b_str[q] == '1':
				oracle.x(q)
	else:
		b = np.random.randint(2)
		if b == 1:
			oracle.x(n)
	ret = oracle.to_gate()
	ret.name = "Oracle"
	return ret

# the main algorithm
def dj(oracle, n):
	dj_circuit = QuantumCircuit(n+1, n)
	# n-qubit register initialized to 0, 1-qubit register initialized to 1
	dj_circuit.x(n)
	for i in range(n+1):
		dj_circuit.h(i)
	
	dj_circuit.append(oracle, range(n+1))
	for i in range(n):
		dj_circuit.h(i)
	for i in range(n):
		dj_circuit.measure(i, i)
	return dj_circuit

n = 4
dj_circuit = dj(dj_oracle(0, n), n)
dj_circuit.draw(output = 'mpl')
plt.show()

aer_sim = Aer.get_backend('aer_simulator')
dj_final = transpile(dj_circuit, aer_sim)
qobj = assemble(dj_final)
res = aer_sim.run(qobj).result()
ans = res.get_counts()
plot_histogram(ans)
plt.show()