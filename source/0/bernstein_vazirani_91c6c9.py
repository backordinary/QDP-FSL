# https://github.com/MarcoArmenta/quantum-circuits/blob/134d7de0b45c37b54322d0a4c1fc138e4643ad53/Bernstein-Vazirani.py
from qiskit import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
	number_to_find = input("Enter binary number to guess"\
				" with Bernstein-Vazirani algorithm: ")
	
	plot_circuit = input("Do you want to plot the circuit? No=0, Yes=1: ")
	n = len(number_to_find)
	qn = list(range(n)) # auxiliary list to apply gates in order
	BV = QuantumCircuit(n+1,n)
	BV.h(qn)
	BV.x(n)
	BV.h(n)

	for i in range(n):
		if number_to_find[i]=='1':
			BV.cx(i,n)
	BV.barrier() # For visualization only
	BV.h(qn)
	BV.measure(qn,qn)

	if int(plot_circuit):
		BV.draw(output='mpl')
		plt.show()

	res = execute(BV, backend = Aer.get_backend('qasm_simulator'),
			shots = 1).result().get_counts()
	# reverse found number because of Qiskit convention for states
	number = res.popitem()[0][::-1]
	print(f'Your number was {number}')
