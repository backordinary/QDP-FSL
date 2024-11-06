# https://github.com/GGearing314/QuanAlgLinSys/blob/601d2d18f0b1263462f2c7b36cf6cc5e3ee3ff1a/FullProjectCode.py
#Quantum Algorithm for system of linear equations 

from qiskit import QuantumCircuit, Aer, transpile
from qiskit.visualization import plot_histogram, plot_bloch_vector
from math import sqrt, pi, cos, sin, acos, log2
from cmath import phase, exp
import numpy as np
import random


from qiskit.circuit.library import U3Gate,U1Gate,RZGate,QFT,RYGate

pi = np.pi 

n = 6 #Number of qubits in the counting register

#Functions:
    
def generate_matrix(): #Generates a random 2x2 matrix and ensures that it is symmetric to satify our condition for Hermiticity
    h = np.random.rand(2,2) #
    return 10.0*(h + h.transpose()) #This forces the matrix to be symmetric

#define our implementation of the controlled Matrix Exp gate Exp(i*h)
#ah is a 2x2 matrix
#c is the number of the control Qubit in QC
#t is the number of target qubit
def ControlledMatrixExpGate(ah, c, t,inverse=False) :
    #inverse: (ah)->(-ah) and reverse order of the rotations.
    #We write our real-valued 2x2 matrix h = [[h00, h01], [h10, h11]] as v0*Id + vx*sigma_x + vy*sigma_y + vz*sigma_z
    #Id is the 2x2 identity matrix, sigma_{x,y,z} are the Pauli matrices
    #We assume the symmetric matrix with h01 = h10. This is enforced in the code, any asymmetry will be ignored
    if inverse: # Reverses sign of matrix
        ah=-ah
    v0 = 0.5*(ah[0,0] + ah[1,1])
    vx = 0.5*(ah[0,1] + ah[1,0])
    vy = 0.0 #For real-valued matrices, vy = 0
    vz = 0.5*(ah[0,0] - ah[1,1])
    #Norm of v - without v0
    nv = sqrt(vx*vx + vy*vy + vz*vz)
    #Normalizing the vector components
    nvx = vx/nv; nvz = vz/nv; nvy = vy/nv;
    theta = 2*acos(sqrt(cos(nv)*cos(nv) + nvz*nvz*sin(nv)*sin(nv)))
    print("theta  = ",theta)
    lpf = 2*phase(complex(cos(nv), -nvz*sin(nv)))
    lmf = 2*phase(complex(-nvy*sin(nv)/sin(theta/2), -nvx*sin(nv)/sin(theta/2)))
    l = 0.5*(lpf + lmf)
    f = 0.5*(lpf - lmf)
    glob_phase = l+f - 2*v0
    if inverse: #Reverse order of rotations
        QC.cu1(-glob_phase, c, t)
        QC.crz( glob_phase, c, t) #This is to get all phase factors right
        QC.cu3(theta, f, l, c, t)
    else: #Normal version
        QC.cu3(theta, f, l, c, t)
        QC.crz( glob_phase, c, t) #This is to get all phase factors right
        QC.cu1(-glob_phase, c, t)


#Generate a random Matrix:
A = generate_matrix()
success = False
while not success: #Ensures the random matrix can represent a solvable system of equations
	A=generate_matrix()
	evs = np.linalg.eigvals(A)
	success = (evs[0] > 1) and (evs[0] < 2**(n-1)) and (evs[1] > 1) and (evs[1] < 2**(n-1))
	
print("Matrix A and its eigenvalues :")
print(A," ",np.linalg.eigvals(A))

#Let's use a matrix with positive and integer eigenvalues
#A = np.array([[11.0, 4.0],[4.0, 5.0]])

#Preparing the |b> state:
a    = np.random.rand()*2*pi #Generates a random number between (0,2pi)
bsrc = np.array([np.cos(a/2),np.sin(a/2)])
print(bsrc)
print()

QC = QuantumCircuit(n+2,2) #Extra bit for ancilla(index=n)and another as the source vector(index=n+1)

QC.ry(a, n+1) # Encodes b vector in the final qubit in the circuit

#Quantum Phase Estimation:
for i in range(n) : #Apply hadamard gates to counting qubits
	QC.h(i)

for i in range(n):
	ph = pi*A*(2**(-n+i+1)) #Explicit argument for the matrix exponent
	ControlledMatrixExpGate(ph, i, n+1) #n+1 is the index of the last qubit of n+2 qubits

QFTC = QFT(n, inverse=True)
QC.compose(QFTC, range(n), inplace=True) # Adds Inverse QFT onto the first qubits of the circuit

#Get eigenvalues using NumPy
evals, evecs = np.linalg.eig(A)
print("Eigenvalues of h: ", evals[0], ", ", evals[1])

def bin_list(i, out) :
	for j in range(len(out)) :
		out[j] = (i >> j)&1
		
#gray_list returns the list of binary digits in the Gray code of i as a NumPy array
#i^(i>>1), where ^ is a bitwise XOR and >> is a bitwise shift, is exactly the Gray code of i        
def gray_list(i, out) :
	bin_list(i^(i>>1), out)
	
#Conditional rotation of qubit with index n, conditioned on qubits with indices 0 to n-1
#Depending on the state of control registers, rotates by alpha_i, where i is an integer
#constructed from the binary digits in the control qubits
alphas=list()
alphas.append(pi/2) #Sets the first angle to pi/2 to...
for i in range(1,2**n):
	alphas.append(np.arcsin(1/i))
print(alphas)

#Now we transform the array of alphas to array of thetas, 
#as described in quant-ph/0404089, paragraph below Eq. 5
#We work directly with the inverse of the matrix M^k_ij
#(M^{-1})_ij = 2^{-n}*(M_ij)^T = 2^{-n}*M_ji = 2^{-n}*(-1)^(b_j*g_i)
#b_j is the binary representation of j
#g_i is the Gray code binary representation of i, see 
#https://en.wikipedia.org/wiki/Gray_code
#Note that our indices i, j run from 0 to 2^n-1

thetas = np.full(2**n, 0.0)
b = np.full(n, 0, dtype=int) #Array for the list of binary digits
g = np.full(n, 0, dtype=int) #Array for the binary Gray code

for i in range(2**n) :
	gray_list(i, g)
	g = np.flip(g) #Reverse the gray code to agree with the conventions of quant-ph/0404089
	for j in range(2**n) :
		bin_list(j, b)
		bg = np.dot(b, g)
		thetas[i] += (alphas[j] if bg%2==0 else -alphas[j])
	
thetas *= 2**(-n)

print(thetas)

# Recursive implementation of the quantum circuit in quant-ph/0404089
# rl is the recursion depth, which determined the index of the control qubit
# this function recursively generates the circuit like Fig. 2 in quant-ph/0404089
# except for the last CNOT gate - it should be added manually (with this prescription the
# iterative structure becomes particularly simple)
def MultiControlRotation(athetas) :
	l = len(athetas)
	rl = int(log2(l))
	#print("l = ", l, ", rl = ", rl)
	if l==2 :
		QC.ry(-2.0*athetas[0], n) #We just want exp(i*theta) in the rotation operator, 
		#CRZ defined as diag(exp(-i*lambda/2), exp(i*lambda/2))
		QC.cnot(n-1, n)
		QC.ry(-2.0*athetas[1], n)
	else :
		MultiControlRotation(athetas[:l//2]) #// is an integer division, first half of array elements
		QC.cnot(n-rl, n)
		MultiControlRotation(athetas[l//2:]) # second half of array elements
	
#Conditional Rotation of Ancilla:
MultiControlRotation(thetas)
QC.cnot(0, n)

QC.measure(n, 0)

#Inverse Quantum Phase Estimation to uncompute the result:
QFTC = QFT(n, inverse=False)
QC.compose(QFTC, range(n), inplace=True) 
for i in range(n):
	ph = pi*A*(2**(-n+i+1)) #Explicit argument for the matrix exponent
	ControlledMatrixExpGate(ph, i, n+1,inverse=True) #n+1 is the index of the last qubit of n+2 qubits
    
sim = Aer.get_backend('aer_simulator')
QC = transpile(QC, sim)

##A small test that shows how to get ancilla always in state "1" and loop over all states of the counting reg and the target qubit (indices x and y)
##These qubits are the last ones in qiskit's convention, but the first bits in the standard bit order
#ib1 = np.full(n+2, 0, dtype=int)
#ib2 = np.full(n+2, 0, dtype=int)
#for ic in range(2**n):#runs through all counting register combinations
#		for x in range(2) :
#			for y in range(2) :
#				i = ((2*x + 1)<<n) + ic
#				j = ((2*y + 1)<<n) + ic
#				bin_list(i, ib1)
#				bin_list(j, ib2)
#				print(ib1," ",ib2)

QC.save_statevector()

success=False
runCount=0
while not success:
	runCount+=1
	#Running the circuit
	result = sim.run(QC).result()
	psi = result.get_statevector()
	#Checking the result - calculate the reduced density matrix, assuming that the ancilla is in the state "1"
	rho = np.full((2, 2), 0.0 + 0.0j)
	ib = np.full(n+2, 0, dtype=int)
	for ic in range(2**n):#runs through all counting register combinations
		for x in range(2) :
			for y in range(2) :
				i = ((2*x + 1)<<n) + ic
				j = ((2*y + 1)<<n) + ic
				rho[x][y] += np.conj(psi[i])*psi[j]
	print("Run number ", runCount, ", we are not successful yet ...")
	success = (np.linalg.norm(rho) > 0.00001)
	
            
print("Algorithm took ",runCount," runs")
#Print out the exact solution
#print(A," ",bsrc)
sol = np.linalg.solve(A, bsrc)
print("Exact solution: ", sol)


# count_1=0 #testing how many additions are actually made
# for i in range(len(psi)) :
#     for j in range(len(psi)) :
#         ib =np.full(n+2, 0, dtype=int)
#         jb =np.full(n+2, 0, dtype=int)
#         bin_list(i, ib)
#         bin_list(j, jb)
#         if (np.array_equal(ib[:n],jb[:n])): #compares the counting register values
#             rho[ib[n+1]][jb[n+1]]+= np.conj(psi[i])*psi[j]
#             count_1+=1
            

            
print("rho = ")
print(rho,"\n\n")

print("Exact solution ratio:",(sol[0]**2)/sol[1]**2)
print("Actual solution ratio:",abs(rho[0][0]/rho[1][1])) #abs to get rid of 0 complex component in print



