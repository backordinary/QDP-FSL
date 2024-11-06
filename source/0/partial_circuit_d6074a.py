# https://github.com/SaniyaM/QOSF-Assessment-Task/blob/18cb0429249ec01ef1dd97f6c11296a40f10c89b/partial_circuit.py
import matplotlib.pyplot as plt
import numpy as np
import math
# importing Qiskit
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
from qiskit.visualization import plot_histogram
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Grover, AmplificationProblem
from qiskit.circuit.library import PhaseOracle, GroverOperator
from qiskit.quantum_info import DensityMatrix, Operator, Statevector
# import basic plot tools
import os
import subprocess
from functions_qosf import clause_matrix_0, solution_states, binarize, inversion, sat, clause_matrix_1, entangle, init_vector


if not os.path.isdir("./QOSF"):
    subprocess.run("mkdir QOSF", shell = True)

#array_i = [1, 2, 3, 4] #input array; accept input, or create a random array - consult qosf doc

if __name__ == "__main__":
	enter = """\nEnter input array\n"""
	input_array = [int(i) for i in input(enter).split()]
    
array_i = input_array #input array

print(array_i)
L=len(array_i)		#length of input array
highest = max(array_i)	
#print(highest)
#l = math.ceil(math.log(highest,2))
l_b = bin(highest)[2:]
l = len(l_b)
#print(l)
bL_b = bin(len(array_i)-1)[2:]
bL = len(bL_b)	
#print(bL)
N = 2**l


address_bits = QuantumRegister(bL, name='a')
value_bits = QuantumRegister(l, name='v')
#output_bits = QuantumRegister(1, name='out')
#cbit = ClassicalRegister(1, name='c')
grover_circuit = QuantumCircuit(value_bits, address_bits)
#-------------------initialise qubits-----------------------------------

grover_circuit.h(address_bits)	#initialise address bits with all possible values of dimension bL

def init_vector(array_i, N):	#initialisation vector
	norm = 0
	init_vec = [0.]*N	
	for i in array_i:
		init_vec[i]=1.0
	print(init_vec)
	for i in init_vec:
		norm = norm + i
	for i in range(len(init_vec)):
		init_vec[i] = init_vec[i]/math.sqrt(norm)
	return init_vec
init_vec = init_vector(array_i, N)		#initialisation vector for value bits
#print(init_vec)

grover_circuit.initialize(init_vec, value_bits)#initialise value bits

#create oracle

solution_states = solution_states(l)		#solution states (2 bitstrings of alternating bits) for the length l

#create clauses for dimacs file
def dimacs(solution_states, N):
	clauses = [0]*N
	clauses = [int(i) for i in range(N)]
	for i in solution_states:
		clauses.remove(i)
	return clauses			
	
clauses = dimacs(solution_states, N)		#array of numbers from 0 to N, excluding the solution states
#print(clauses)
#clauses = binarize(clauses, l)

#generate dimacs file constrains
def sat(all_combinations, l, bL):
	L2 = ""
	temp = []
#	clause = [int(i)+1 for i in range(l)]	
#	m = max(clause)
	for i in range(0, len(all_combinations)):
		k=0
		j=all_combinations[i]
		temp = [int(p)+1 for p in range(l)]	#To create the constraints/clauses (1, 2, .. l)
		while k<=l-1:
			if j%(2)==0:
				temp[k] = (temp[k])*(-1)	#check order
			stri = str(temp[k]) + str(' ')
			L2 += (stri)
			j = j//2
			k = k+1
		L2 += '0 \n'
	return L2

#create dimacs file
path = './QOSF'  

file = 'newsat.dimacs'

with open(os.path.join(path, file), 'w') as fp:
    pass
    
file1 = open("newsat.dimacs","w")

L1 = ["c QOSF DIMACS-CNF SAT \n", "p cnf {} {} \n".format(l, 2**(l)-2)]

L2 = sat(clauses, l, bL)

file1.writelines(L1)
file1.writelines(L2)
file1.close()		


#create phase oracle 
oracle = PhaseOracle.from_dimacs_file('newsat.dimacs')	

grover_operator = GroverOperator(oracle)  

grover_circuit = grover_circuit.compose(grover_operator)
grover_circuit.measure_all()	#measure all qubits

print(grover_circuit.draw(output='text'))	#print the circuit and gates


#attributes for GroverOperator
#ref_q = clause_matrix_0(bL)
#print(ref_q)
#diffuse_operator = 2 * DensityMatrix.from_label('000') - Operator.from_label('III')
#, reflection_qubits = ref_q, zero_reflection = diffuse_operator


#simulate the circuit
sim = Aer.get_backend('aer_simulator')		#simulator
t_qc = transpile(grover_circuit, sim)	
nshots = 4096
counts = sim.run(t_qc, shots=nshots).result().get_counts()

#for one solution state
sol_state_count = max(counts.values())
counts_values = list(counts.values())
sol_state_index = counts_values.index(sol_state_count)
counts_keys = list(counts.keys())
sol_state = counts_keys[sol_state_index]
#print(sol_state)
#print(counts)
def output_vector(counts_keys, sol_state, array_i):
	out_vec = []
	ind=0
	for i in counts_keys:
		if i == sol_state:
			ind = array_i.index(int(i[bL:],2))
			out_vec.append(bin(ind)[2:].zfill(bL)) 
	#print(ind)
	return out_vec
out_vec = output_vector(counts_keys, sol_state, array_i)

print(array_i)
#print(sol_state)
#output_index = array_i.index(int(sol_state, 2))
#print(out_vec)

def superpose(out_vec):	#initialisation vector; wont work for repeated SOLUTIONS, works for repeated input
	output = ''	#equal superposition intialisation vector	
	for i in out_vec:
#		j = bin(i)[2:]
		output += '|'+ i +  '>*'+ '1/sqrt({})'.format(len(out_vec))
	#print(output)
	return output
output = superpose(out_vec)

plot_histogram(counts)
plt.show()


