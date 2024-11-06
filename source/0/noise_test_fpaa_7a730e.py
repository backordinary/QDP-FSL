# https://github.com/MSwenne/BEP/blob/f2848e3121e976540fb10171fdfbc6670dd28459/Code/noise_test_FPAA.py
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import IBMQ, execute, Aer
from qiskit.providers.aer import noise
import matplotlib.pyplot as plt
from plot import plot_hist
from scipy import stats
import qiskit as qk
import numpy as np
import time
import math
import sys

from cnf_oracle import cnf_oracle
from mean_inversion import mean_inversion


worst = 423 # 43.2 microseconds
best = 577 # 57.7 microseconds

def main():
	n, clauses, oracle, index, iterations, run, answer = get_input()
	answer = answer[22:-1]
	solutions = answer.split(",")
	avg_time_noise = 0
	avg_time_base = 0
	runs = range(worst, best, math.floor((best-worst)/run))
	a = math.floor(math.log(clauses,2)+1)
	data_noise = [[0 for y in range(len(runs))] for x in range(pow(2, n))] 
	data_base = [[0 for y in range(len(runs))] for x in range(pow(2, n))] 
	nr = 0
	for i in runs:
		data, time = Grover_Search(n, clauses, oracle, index, iterations, True, i)
		for j in range(pow(2, n )):
			if bin(j)[2:].zfill(n + 1 + a) in data:
				data_noise[j][nr] = data[str(bin(j)[2:].zfill(n + 1 + a))]
		avg_time_noise = avg_time_noise + time

		data, time = Grover_Search(n, clauses, oracle, index, iterations, False, i)
		for j in range(pow(2, n)):
			if bin(j)[2:].zfill(n + 1 + a) in data:
				data_base[j][nr] = data[str(bin(j)[2:].zfill(n + 1 + a))]
		avg_time_base = avg_time_base + time
		nr = nr + 1

	avg_time_noise = avg_time_noise/len(runs)
	avg_time_base = avg_time_base/len(runs)
	print()
	print('solution(s)', solutions)
	f=open("results.txt", "a+")
	f.write("nr. of variables: %d\r\nnr. of clauses: %d\r\nsolutions: %d\r\namount of runs: %d\r\n#########\r\n" % (n,clauses,solutions,runs))
	print('Average time with noise: ', np.round(avg_time_noise,3), ' seconds')
	for i in range(pow(2, n)):
		print(bin(i)[2:].zfill(n),":",data_noise[i])
	print('Average time without noise: ', np.round(avg_time_base,3), ' seconds')
	for i in range(pow(2, n)):
		print(bin(i)[2:].zfill(n),":",data_base[i])

	x = range(len(runs))
	data_vis = ['ro','go','r-','g-']
	plt.figure(0)
	for i in range(pow(2, n)):
		slope, intercept, r_value, p_value, std_err = stats.linregress(x,data_noise[i])
		line = slope*x+intercept
		if bin(i)[2:].zfill(n) in solutions:
			plt.plot(x, line, data_vis[3])
			plt.plot(x, data_noise[i], data_vis[1], markersize=2)
		else:
			plt.plot(x, line, data_vis[2])
			plt.plot(x, data_noise[i], data_vis[0], markersize=2)
	plt.figure(1)
	for i in range(pow(2, n)):
		slope, intercept, r_value, p_value, std_err = stats.linregress(x,data_base[i])
		line = slope*x+intercept
		if bin(i)[2:].zfill(n) in solutions:
			plt.plot(x, line, data_vis[3])
			plt.plot(x, data_base[i], data_vis[1], markersize=2)
		else:
			plt.plot(x, line, data_vis[2])
			plt.plot(x, data_base[i], data_vis[0], markersize=2)
	plt.show()



def Grover_Search(n, clauses, oracle, index, iterations, has_noise, times):
	start_time = time.time()
	a = 1
	theta = math.pi/clauses
	delta = pow(2,-0.5*pow(n,2))/4
	_lambda = pow(math.sin(theta),2)/4+pow(1/2-math.cos(theta)/2,2)
	L = int(math.ceil(2*math.log(2/delta)/math.sqrt(_lambda)))

	## Too much qubits needed for operation (max qubits for simulator is 24)
	if (n+1) + a > 24:
		print("Too much qubits needed! (", (n+1) + a,")")
		print("Max qubits 24")
		sys.exit()

	## circuit generation
	q = QuantumRegister( n +1 + a )
	c = ClassicalRegister( n + 1 + a )
	qc = QuantumCircuit(q, c)

	for i in range(math.floor(n/2)):
		qc.swap(q[i],q[n-i-1])
	qc.barrier()
	for i in range(n):
		qc.h(q[i])

	for i in range(iterations):
		fpaa_oracle(qc, q, n, clauses, oracle, index)
		qc.barrier()
		mean_inversion(qc, q, n, 2)

		# for i in range(n,n+a):
		# 	qc.measure(q[i], c[0])
		# 	qc.x(q[i]).c_if(c,1)

	qc.barrier()
	for i in range(math.floor(n/2)):
		qc.swap(q[i],q[n-i-1])
	qc.barrier()

	for i in range(n):
		qc.measure(q[i], c[i])

	backend = qk.BasicAer.get_backend('qasm_simulator')

	# Execute noisy simulation and get counts
	if has_noise:
		# device = IBMQ.get_backend('ibmq_16_melbourne')
		# device = IBMQ.get_backend('ibmq_5_yorktown')
		device = IBMQ.get_backend('ibmq_5_tenerife')
		properties = device.properties()
		coupling_map = device.configuration().coupling_map

		# Note that the None parameter for u1, u2, u3 is because gate
		# times are the same for all qubits
		gate_times = [
				('u1', None, 0), ('u2', None, 100), ('u3', None, 200),
				('cx', [1, 0], 678), ('cx', [1, 2], 547), ('cx', [2, 3], 721),
				('cx', [4, 3], 733), ('cx', [4, 10], 721), ('cx', [5, 4], 800),
				('cx', [5, 6], 800), ('cx', [5, 9], 895), ('cx', [6, 8], 895),
				('cx', [7, 8], 640), ('cx', [9, 8], 895), ('cx', [9, 10], 800),
				('cx', [11, 10], 721), ('cx', [11, 3], 634), ('cx', [12, 2], 773),
				('cx', [13, 1], 2286), ('cx', [13, 12], 1504), ('cx', [], 800)
		]

		# Construct the noise model from backend properties
		# and custom gate times
		noise_model = noise.device.basic_device_noise_model(properties, gate_times=gate_times)

		# Get the basis gates for the noise model
		basis_gates = noise_model.basis_gates
		result_noise = execute(qc, backend=backend,	shots=100, noise_model=noise_model, coupling_map=coupling_map, basis_gates=basis_gates).result()
		counts_noise = result_noise.get_counts(qc)
		print("Time taken for result with noise:",np.round(time.time() - start_time,3), "seconds")
		return counts_noise, time.time() - start_time
	else:
		result_base = execute(qc, backend=backend, shots=100).result()
		counts_base = result_base.get_counts(qc)
		print("Time taken for result without noise:",np.round(time.time() - start_time,3), "seconds")
		return counts_base, time.time() - start_time


## Helper function to get variables
def get_input():
	if len(sys.argv) == 3:
		n, clauses, k, oracle, index, answer = set_cnf()
	## Incorrect input
	else:
		sys.exit()

	if not isinstance(k,int):
		k = 1
	if pow(2, n)/k < 4:
		exit()
	iterations = math.floor((math.pi*math.sqrt(pow(2, n)/k))/4)
	runs = int(sys.argv[2])
	return n, clauses, oracle, index, iterations, runs, answer


## Helper function to set cnf variables
def set_cnf():
	file = sys.argv[1]
	## Get variables from cnf file
	with open(file, 'r') as f:
		answer = f.readline()
		n, clauses, k = [int(x) for x in next(f).split()]
		oracle = [[int(x) for x in line.split()] for line in f]
	oracle_bin = np.negative([np.ones(n) for x in range(clauses)])
	for i in range(clauses):
		for j in oracle[i]:
			oracle_bin[i][abs(j)-1] = int(j < 0)
	oracle = oracle_bin
	index = []
	for j in range(clauses):
		index.extend([[]])
		for i in range(n):
			if oracle[j][i] != -1:
				index[j].extend([i])
	return n, clauses, k, oracle, index, answer


###########
## START ##
###########
main() 