# https://github.com/attianopp/CS239---Quantum-Programming/blob/ce61f0646f01b7785ae5dfa06144417e2d037f89/qiskit/Submission/grover.py
import sys
import inspect
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from qiskit.visualization import plot_histogram
from qiskit.quantum_info.operators import Operator
from itertools import combinations
import time, random
import numpy as np
import math
from operator import add
import operator
import matplotlib.pyplot as plt

class grover:

	def run_grover(self,f,n,a):
		"""
		inputs: f - the oracle function
				n - integer defining number of qubits
				a - integer defining number of inputs to f that ==1

		outputs: result - ket{x} s.t. f(x) = 1
				run_time - float number of seconds it took to run quantum simulator
				numerical_prob - probability of success based on qasm_simulator
				theo_prob - float corresponding to theoretical probability of success
				
		"""

		gate = self.create_G(f,n)
		G = Operator(gate)
		circuit, theo_prob = self.create_circuit(G,n,a)
		result, run_time, numerical_prob = self.execute_grover(circuit,n,f)
		return result, run_time, numerical_prob, theo_prob
	
	def create_circuit(self,G,n,a):
	
		""" inputs: G - Operator of matrix object from qiskit - custom gate for grovers circuit n,a
					n - int defining number of qubits
					a - int defining number of x s.t f(x)==1
			outputs: qiskit circuit object corresponding to to grovers algorithm with n,a
					 theo_prob - float corresponding to theoretical probability of success
		""" 
	
		circuit = QuantumCircuit(n, n)

		#1. Hadamard each qubit
		for i in range(n):
			circuit.h(i)

		#setup inputs to multi-bit gates
		num_arr = list(range(n))

		#2. apply G = -(H^n)zo(H^n)zf k times
		k , theo_prob = self.calc_lim(a,n)
		for iter in range(k):
			# circuit.append(G,reversed(range(n)))
			circuit.append(G,range(n))
		
		#measure each qubit
		circuit.measure(range(n), range(n))
		
		return circuit, theo_prob
		
			

	def convert_int_to_n_bit_string(self,integer, n):
		"""
		inputs: integer corresponding to x's unsigned value
				n is the number of bits

		outputs: array of ints that represents an unsigned bit string
		"""
		return [ int(char) for char in bin(integer)[2:].zfill(n) ]

	def convert_n_bit_string_to_int(self,x,n):
		"""
		inputs: x - array of ints that represents an unsigned bit string
				n is the number of bits

		outputs: integer corresponding to x's unsigned value
		"""
		return int("".join([str(num) for num in x]), 2)


	def check_correctness(self,func,result,n):
		"""
		input: func - np array of ints that represnts a function mapping with the index as input
			   result - np array of ints that represents an unsigned bit string
			   n is the number of bits
		
		output: bool if the index that corresponds to result is 1, then f(result) = 1
		"""
		if(str(type(func))!="<class 'list'>" and str(type(func))!="<class 'numpy.ndarray'>" and str(type(func))!="<class 'function'>"):
			raise TypeError('input for func not a list, numpy array or a function')
		
		
		x = self.convert_n_bit_string_to_int(result,n)
		
		
		if(str(type(func))=="<class 'function'>"):
			return (func(*result) == 1)
		else:
			return (func[x] == 1)


	def all_f(self,n,a):
		""" 
		input: int n = number of qubits, int a = number of items that return 1 for function
		
		output: list of functions (arrays that can be indexed with 'x' for function value 'f(x)'
		""" 
		if(str(type(n))!="<class 'int'>"):
			raise TypeError('input for n non-integer')	
		if(str(type(a))!="<class 'int'>"):
			raise TypeError('input for a non-integer')
		N = 2**n
		func_inputs = range(N)
		comb = combinations(func_inputs, a)
		func_list = []

		for combination in comb:
			out_list = np.zeros(N)
			for i in combination:
				out_list[i] = 1
			func_list.append(out_list)
			
		return func_list

	def calc_lim(self, a,n):
		"""
		input: int n = number of qubits, int a = number of items that return 1 for function

		output: int k = number of iterations to run grovers algorithm
		"""
		if(str(type(n))!="<class 'int'>"):
			raise TypeError('input for n non-integer')	
		if(str(type(a))!="<class 'int'>"):
			raise TypeError('input for a non-integer')

		N = 2**n
		theta = np.arcsin(np.sqrt(a/N))
		
		k_approx = ((np.pi/(4*theta)) - (1/2))
		
		k_arr = np.array([np.ceil(k_approx),np.floor(k_approx)])
		
		prob_arr = np.sin((2*k_arr + 1) * theta ) ** 2
		max = np.argmax(prob_arr)
		
		return int(k_arr[max]), prob_arr[max]
		
	def H_tensored(self,n):
		""" Returns the matrix corresponding to the tensor product of n number of Hadamard gates
		Args:
			n: Integer
		Returns:
			ndarray of size [2**n, 2**n]
		"""
		H = 1/np.sqrt(2) * np.array([[-1,1],[1,1]])
		n = n - 1
		H_n = H
		while n > 0:
			H_n = np.kron(H_n, H)
			n -= 1
		return H_n
		
	def reverse_bit_int(self,i):
		"""
		inputs: i - integer
		outputs: integer corresponding to i's unsigned bit string representation flipped e.g 4 => 0100 => 0010 => 2
		"""
		bit_string = self.convert_int_to_n_bit_string(i,n)
	
		bit_string = bit_string[::-1]
		number = self.convert_n_bit_string_to_int(bit_string,n)
		return number
		
    
		
	def create_G(self,f,n):
		"""
		Given a function f:{0,1}^n ---> {0,1}, creates and returns the corresponding Grover operator G (unitary matrix)
		Args:
			f: ndarray of length 2**n, consisting of integers 0 and 1
		Returns:
			G: ndarray of size [2**(n+1), 2**(n+1)], representing a unitary matrix.
		"""
		N = 2**n
		
		G = np.zeros([N,N], dtype = complex)
		Zf = np.eye(N)
		for i in range(N):
			j = self.reverse_bit_int(i)
			Zf[j][j]*=((-1)**f[i])
		Z0 = np.eye(N)
		Z0[N-1,N-1] = -1

		
		
		H_n = self.H_tensored(n)
		G = - H_n @ Z0 @ H_n @ Zf
		return G
		
	

	def execute_grover(self,circuit,n,f,num_shots = 1000, backend = 'qasm_simulator'):
		
		"""
		inputs: 
			circuit - 
			n - int defining number of qubits
			f - np array of ints 0/1, with 2^n indicies for f[x]=0/1
		outputs: results of 1000 shots of grovers algorithm and numerical probability of success given f, n
		"""
				
		#setup quantum computer
		simulator = Aer.get_backend(backend)
		
		#get run-time
		start = time.time()
		job = execute(circuit, simulator, shots=num_shots)
		end = time.time()
		run_time = int((end - start) * 1000)
		
		#get results
		result = job.result()
		counts = result.get_counts(circuit)
		
		#get number of results that correctly identified f(x)==1
		total = 0
		for potential_soln in counts:
			intermediate = [int(i) for i in potential_soln][::-1]
			check = self.check_correctness(f,intermediate,n)
			if(check):
				total += counts[potential_soln]
		
		y = list(max(counts, key = lambda key: counts[key]))
		numerical_prob=(total/num_shots)

		return [int(char) for char in y][::-1], run_time, numerical_prob


if __name__== '__main__':
	num_inputs = len(sys.argv)
	inputs = sys.argv
	
	g = grover()
	
	#test correctness
	min_n = 1
	max_n = 6
	max_a = 3
	n_list = list(range(min_n,max_n+1))
	a_list = list(range(1,max_a+1))
	for n in n_list:
		for a in a_list:
			if(a>n):
				break
			all = g.all_f(n,a)
			func = random.choice(all)
			result, run_time, numerical_prob, theoretical_prob = g.run_grover(func,n,a)
			print(" ")
			print("N:%d, A:%d prob_success: %.4f"%(n,a,theoretical_prob))
			print("         numerical_prob: %.4f"%numerical_prob)
	
	
	# #calculate computation time for n=6 a=1->6

	# min_n = 6
	# max_n = 6
	# max_a = 6
	# n_list = list(range(min_n,max_n+1))
	# a_list = list(range(1,max_a+1))
	# time_vs_a = []
	# for n in n_list:
		# for a in a_list:
			# if(a>n):
				# break
			# all = g.all_f(n,a)
			# func = random.choice(all)
			# result, run_time, numerical_prob, theoretical_prob = g.run_grover(func,n,a)
			# time_vs_a.append(run_time)
	
	
	# # #compute average compile and compute time for all functions for a range of n values and a given 'a'
	# a = 1
	# n_min = 1
	# n_max = 10
	# n_list = list(range(n_min,n_max+1))
	# num_times = 100
	# compute_times = np.zeros([(n_max-n_min)+1])
	# for ind, n in enumerate(n_list):
		# avg_compute = 0
		# for iter_ind in range(num_times):
			# all = g.all_f(n,a)
			# func = random.choice(all)
			# # print("---------------FUNCTION------------")
			# # print(func)
			# # print("---------------FUNCTION------------")
			# result, run_time = g.run_grover(func,n,a)
			# avg_compute += run_time
			# # print("---------------RESULT------------")
			# # print(result)
			# # print("---------------RESULT------------")
		# compute_times[ind] = avg_compute/num_times
		
	# print("compute times")
	# print(compute_times)
		
		
	# #testing for uf computation time at a fixed n
	# num_funcs = 16 #64 is (4 qubits choose 1) -> 2^4 choose 1 -> number of possible functions with a=2
	# input_list = list(range(num_funcs))
	# uf_compute_times = np.zeros([num_funcs,num_times])
	# for iter_ind in range(num_times):
		# a = 1
		# n = 4
		# all = g.all_f(n,a)
		# for ind,func in enumerate(all):
			# #print(func)
			# result, run_time = g.run_grover(func,n,a)
			# #print(result)
			# uf_compute_times[ind, iter_ind] = run_time
	
	# #%% Save the data

	# np.savez('grover_benchmarking.npz', n_list = n_list,compute_times = compute_times, time_vs_a = time_vs_a, a_list = a_list)
	# np.savez('grover_uf.npz', input_list = input_list,uf_compute_times=uf_compute_times)
	   
	# #%% Load data

	# data = np.load('grover_benchmarking.npz')
	# time_vs_a = data['time_vs_a']
	# a_list = data['a_list']
	# n_list = data['n_list']
	# computation_times = data['compute_times']
	# avg_time_compute = computation_times
	
	# uf_data = np.load('grover_uf.npz')
	# input_list = uf_data['input_list']
	# uf_compute_times = uf_data['uf_compute_times']
	
	
	# #%% Plot and save 

	#plot computation time vs n

	# fig = plt.figure(figsize=(16,10))

	# z = np.polyfit(n_list, avg_time_compute, 8)
	# p = np.poly1d(z)

	# plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
	# plt.plot(n_list, avg_time_compute, ls = '', markersize = 15, marker = '.',label = 'M') #, ls = '--'
	# plt.title('Computation time scaling for Grovers algorithm, a = 1', fontsize=25)
	# plt.xlabel('n (bit string length)',fontsize=20)
	# plt.ylabel('Average time of computation (s)',fontsize=20)
	# plt.xticks(fontsize=15)
	# plt.yticks(fontsize=15)


	# fig.savefig('Figures/grover_computation_100trials.png', bbox_inches='tight')

	
	# #plot computation time vs uf for n= 4

	# fig = plt.figure(figsize=(16,10))

	
	# plt.hist(uf_compute_times)
	# plt.title('Dependence of execution time on $U_f$ (Grover algorithm)', fontsize=25)
	# plt.xlabel('Execution time (ms)',fontsize=20)
	# plt.ylabel('Frequency of occurence',fontsize=20)
	# plt.xticks(fontsize=15)
	# plt.yticks(fontsize=15)


	# fig.savefig('Figures/grover_hist.png', bbox_inches='tight')
	
	
	# #plot computation time vs a

	# fig = plt.figure(figsize=(16,10))

	# z = np.polyfit(a_list, time_vs_a, 5)
	# p = np.poly1d(z)

	# plt.plot(np.linspace(a_list[0], a_list[-1] + 0.1, 100), p(np.linspace(a_list[0], a_list[-1] + 0.1, 100)), ls = '-', color = 'r')
	# plt.plot(a_list, time_vs_a, ls = '', markersize = 15, marker = '.',label = 'M') #, ls = '--'
	# plt.title('Computation time scaling for Grovers algorithm, n = 6, a=1->6', fontsize=25)
	# plt.xlabel('a',fontsize=20)
	# plt.ylabel('Average time of computation (s)',fontsize=20)
	# plt.xticks(fontsize=15)
	# plt.yticks(fontsize=15)


	# fig.savefig('Figures/grover_n3_va.png', bbox_inches='tight')