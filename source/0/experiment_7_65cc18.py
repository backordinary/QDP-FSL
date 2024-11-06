# https://github.com/np84/qcmrf/blob/cf6413b6ec44f6c94c7c2c239c91e725013422b2/old/experiment_7.py
import sys
import numpy as np
from scipy.linalg import expm
np.set_printoptions(threshold=sys.maxsize,linewidth=1024)

import itertools
from colored import fg, bg, attr

from qiskit.opflow import I, X, Z, Plus, Minus, H, Zero, One, MatrixOp
from qiskit.compiler import transpile
from qiskit import QuantumRegister, QuantumCircuit
from qiskit import Aer, assemble

#######################################

# Compute fidelity between probability mass functions, given by P and Q
# P[i] and Q[i] contains probabilities for the i-th event
def fidelity(P,Q):
	F = 0
	for i in range(len(P)):
		F += np.sqrt(P[i] * Q[i])
	return F**2

# Compute Kullback-Leibler between probability mass functions, given by P and Q
# P[i] and Q[i] contains probabilities for the i-th event
def KL(P,Q):
	kl = 0
	for i in range(len(P)):
		if Q[i] > 0 and P[i] > 0:
			kl += P[i] * np.log(P[i] / Q[i])
	return kl

# Python helper for iterating over tuples of length n
def grouped(iterable, n):
	return zip(*[iter(iterable)]*n)

def merge(L):
	R = []
	for (M0,M1) in grouped(L,2):
		R.append((((I+Z)/2)^M0) + (((I-Z)/2)^M1))
	return R

def merge_all(L):
	R = L
	while len(R)>1:
		R = merge(R)
	return R[0]

turquoise = '#179c7d'
orange = '#eb6a0a'
blue = '#1f82c0'

# string formatting
def format_head(h,t):
	return fg(turquoise)+attr('bold')+h+attr('reset')+': '+fg('white')+attr('bold')+t+attr('reset')

# string formatting
def format_val(val):
	if type(val) is int:
		return fg('white')+attr('bold')+str(val)+attr('reset')
	else:
		return fg('white')+attr('bold')+np.format_float_scientific(val, precision=4)+attr('reset')

# string formatting
def format_math(math):
	brackets  = ['(',')']
	operators = ['+','-','*','/','_','^']
	mcol = turquoise
	bcol = 'white'
	ocol = orange
	for b in brackets:
		math = math.replace(b,fg(bcol)+b+fg(mcol))
	for o in operators:
		math = math.replace(o,fg(ocol)+o+fg(mcol))
	return fg(mcol)+math+attr('reset')

####################################### model funcs

# compute 2**n X 2**n binary diaginal matrix D
# D_ii is 1 if, for the i-th basis state x**(i), we have x**(i)_C = y
def genPhi(c,y):
	result = 1

	plus  = [v for i,v in enumerate(c) if not y[i]] # 0
	minus = [v for i,v in enumerate(c) if     y[i]] # 1

	for i in range(n):
		f = I
		if i in minus:
			f = (I-Z)/2
		elif i in plus:
			f = (I+Z)/2

		result = result^f

	return result

# return pair R,L
# R is the Hamiltoian sum_C sum_y theta_C,y * Phi_C,y 
# L list of lists, one list per clique C
# List for clique C contains pairs (theta,Phi) of weight theta_C,y and Phi_C,y
def genHamiltonian():
	L = []
	R = 0
	i = 0
	for l,c in enumerate(C):
		Ly = []
		for y in list(itertools.product([0, 1], repeat=len(c))):
			Phi = genPhi(c,y)
			theta = np.random.uniform(low=-5.0,high=-0.001) # we need a negative MRF
			#if i==0:
			#	theta = -3.2
			#elif i==1:
			#	theta = -1.8
			R += Phi * -theta
			Ly.append((theta,Phi)) # list of all factors that belong to same clique
			i = i + 1
		L.append(Ly) # list of factor-lists, one factor-list per clique

	return R,L

# returns unitary if A is unitary
def conjugateBlocks(A):
	return (((I+Z)/2)^A) + (((I-Z)/2)^(~A))

# computes exp(-beta H) = PROD_j exp(-beta w_j Phi_j) = PROD_j REAL( P**(beta w_j)(U_j) )
def genCircuit(beta, CLIQUES):
	num_cliques = len(CLIQUES)
	qr = QuantumRegister(n+num_cliques, 'q') # one aux per clique

	# create empty main circuit with |C| aux qubits
	circ = QuantumCircuit(n+num_cliques,n+num_cliques)
	for i in range(n):
		circ.h(qr[i])
	circ.barrier(qr)

	for ii,C in enumerate(CLIQUES):
		CLIQUE_FACTOR = I^(n+1)
		for theta,Phi in C:
			U     = (X^((I^n)-Phi)) + (Z^Phi)
			gamma = 0.5 * np.arccos(np.exp(beta*0.5*theta))
			RZ    = (-gamma * Z).exp_i() ^ (I^n)
			Ugam  = (RZ @ U)**2
			CLIQUE_FACTOR = Ugam @ CLIQUE_FACTOR

		M = CLIQUE_FACTOR.to_matrix()[:2**(n),:2**(n)] # extract upper left block -- hack to reduce aux qubits by 1
		CLIQUE_FACTOR = MatrixOp(M)

		# Write U**gamma and ~U**gamma on diagonal of matrix, creates j-th aux qubit
		# Create "instruction" which can be used in another circuit
		u = conjugateBlocks(CLIQUE_FACTOR).to_circuit().to_instruction(label='U_C('+str(ii)+')')

		# add Hadamard to j-th aux qubit
		circ.h(qr[n+ii])
		# add CLIQUE_FACTOR to main circuit
		circ.append(u, [qr[j] for j in range(n)]+[qr[n+ii]])
		# add another Hadamard to aux qubit
		circ.h(qr[n+ii])
		circ.measure([n+ii],[n+ii]) # real part extraction successful if measured 0

		circ.barrier(qr)

	# measure all qubits
	circ.measure(range(n),range(n))
	return circ

#######################################

RUNS = [[[0]],[[0,1]],[[0,1],[1,2]],[[0,1],[1,2],[2,3]]]
#,[[0,1],[1,2],[2,3],[0,3]],[[0,1,2],[0,2,3]],[[0,1,2,3]]]
#RUNS = [[[0,1],[1,2],[2,3],[0,3]],[[0,1],[1,2],[2,3],[0,3],[3,4,5]]]
#RUNS = [[[0,1,2,3],[3,4,5,6]]]

logfile = open("results_experiment_5.csv", "w")
logfile.write('n,d,num_cliques,C_max,fidelity,KL,success_rate,num_gates,depth,shots,w_min,w_max\n')

for C in RUNS:
	for II in range(10):
		n = len(np.unique(np.array(C).flatten())) # number of (qu)bits
		d = 0
		cmax = 0
		for c in C:
			m = len(c)
			if m > cmax:
				cmax = m
			d = d + (2**m)

		dim = 2**n

		HAM,LL = genHamiltonian() # L is list of factors
		beta = 1
		R0  = expm(-beta*HAM.to_matrix()) # exp(-βH) via numpy for debugging
		R2b = genCircuit(beta, LL)
		print(R2b)
		OL  = 3
		UU  = transpile(R2b, basis_gates=['cx','id','rz','sx','x'], optimization_level=OL)
		N   = 1000000
		sim = Aer.get_backend('qasm_simulator')
		j   = sim.run(assemble(UU,shots=N))
		R   = j.result().get_counts()
		#print(R)
		Y   = list(itertools.product([0, 1], repeat=n))
		P   = np.zeros(dim)

		wmin = None
		wmax = None
		for l1 in LL:
			for ww,ll in l1:
				if wmin is None:
					wmin = ww
				elif ww < wmin:
					wmin = ww

				if wmax is None:
					wmax = ww
				elif ww > wmax:
					wmax = ww

		for i,y in enumerate(Y):
			s = ''
			for b in y:
				s += str(b)
			s0 = '0'*len(C) + s

			if s0 in R:
				P[i] += R[s0]

		ZZ = np.sum(P)
		P = P/ZZ

		lnZ = np.log(np.trace(R0))
		Q = np.diag(R0/np.exp(lnZ))

		logs = str(n)+','+str(d)+','+str(len(C))+','+str(cmax)+','+str(np.real(fidelity(P,Q)))+','+str(np.real(KL(Q,P)))+','+str(ZZ/N)+','+str(len(UU))+','+str(UU.depth())+','+str(N)+','+str(wmin)+','+str(wmax)
		print(logs)
		logfile.write(logs+'\n')

logfile.close()
