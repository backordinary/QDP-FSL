# https://github.com/julianJames196/quantumStructurePrediction/blob/f422265c368b6d9f9ea91239ad11da45128ffa5d/quantized.py
#The Quantum Computing algorithm using a VQE
import numpy as np
#from matplotlib import pyplot as plt
import sys #This allows us to exit loops

import qiskit
from qiskit import *
from qiskit.tools.visualization import plot_histogram

from qiskit.circuit.library import RealAmplitudes
#RealAmplitudes is an Ansatz for the VQE

from scipy.optimize import minimize
from itertools import combinations, product
#My library
import optimizeLib
from optimizeLib import LJ, LJMod

#Global system variables
#A global variable for the distance between adjacent bins. Sets the physical scale
binSize = 0.5*pow(2, 1/6)*optimizeLib.sigma #Ideal distance is 2 bins away
n = 6 #Number of qubits in the circuit
#n will likely have to be even
maxiter = 250
nshots = 1e4 #Is the number to be varied

#A method to take a binary string and calculate energy
def energy_int(integerIn):
    #Input: an integer
    #Out the energy of the string using encoding 2
    #Builds on energySurfaceQ in strucutrePredictQuantized
    
    #Take the input and convert to a binary array
    binArray = [bool(int(x)) for x in str( "{0:b}".format(integerIn)  )]
    #Need to make it of length n
    temp = np.full( n-len(binArray), False )
    binArray = np.concatenate((temp, binArray))
    #First split the array in half
    l = int(n /2)
    atomA = binArray[:l] #First half of the array
    atomB = binArray[l:]
    
    #Run the energySurfaceQ function
    #Note len(atomA)=len(atomB)=Number of bins pre specified
    energy = 0
    #Go through the atomA list to see repulsion
    for x in combinations(range(len(atomA)), 2):
        #x is a tuple of the form (0,3)
        if atomA[x[0]] and atomA[x[1]]:
            Dis = binSize*abs(x[0]- x[1])
            energy += LJMod(Dis) #REPULSION POTENTIAL
            
        if atomB[x[0]] and atomB[x[1]]:
            Dis = binSize*abs(x[0]- x[1])
            energy += LJMod(Dis) #REPULSION POTENTIAL
    
    #Go through both to find attraction. 
    for x in product(range(len(atomB)), repeat = 2):
        if atomA[x[0]] and atomB[x[1]]:
            Dis = binSize*abs(x[0]- x[1])
            energy += LJ(Dis) #REPULSION POTENTIAL
    
    #Output energy
    return energy
    
def energy_from_full_dist(pos):
    ensum= 0
    for j in pos:
        ensum += energy_int(j)

    return ensum

#######################################
def build_rycnot(n, depth, params):
    """RY CNOT ansaztz"""
    
    if len(params) != n*depth:
        print("ERROR!, Too few parameters")
        sys.exit()

    ansatz = RealAmplitudes(n, entanglement='linear', reps=int(depth)-1 , insert_barriers =True)

    if len(params) == ansatz.num_parameters_settable:
        ansatz.ordered_parameters = params
    else:
        print('ERROR!, Number of parameters to vary doesnt equal the number given' )
        sys.exit()

    return ansatz

#from shots counts to distribution
def find_distribution(n, counts, NSHOTS):
    
    #yes this is exponentially costly written like that..

    N = 1<<n
    distr = {}
    for n in range(N):
        try:
            distr[n] = counts[n]
        except:
            distr[n] = 0
    return distr

def qasm_distribution(n, par, NSHOTS = 1e4):
    """execute the circuit and return the dictionary with the counts"""
    """ NSHOTS is the number to be varied """
    
    N = 1<<n
   
    qc = QuantumCircuit(n)
    
    depth = int(len(par)/n)
    
    ansatz = build_rycnot(n, depth, par)
    qc.compose(ansatz, inplace = True)
    qc.measure_all()

    #backend definition and run
    backend = Aer.get_backend('qasm_simulator')
    qcc = transpile(qc, backend)
    counts = backend.run(qcc, shots = NSHOTS).result().get_counts()
    
    #coversion from counts "000010" label into integer j label (binary)
    int_counts = counts.int_outcomes()
    
    #create normalized histogram
    #pos has length 2^n
    pos = find_distribution(n, int_counts, NSHOTS)

    return pos

#Work on this
def cost_function(params):            
    '''from circuit parameters to cost function'''
              
    '''include something like'''
    
    pos = qasm_distribution(n, params)
    cost_function = energy_from_full_dist( pos)         
                  
    return cost_function

#Main for this part
param = np.random.rand(5*n) #n given above as global var
pos = qasm_distribution(n, param)

print(pos)
print("Plotting")
plot_histogram({'00': 550, '11': 450})
#plot_histogram(pos, bar_labels = False)

#Include additonal sub routine parameters in args
print("Calculting result")
result = minimize(cost_function, param,
                            method='COBYLA',tol=1e-6, options={'maxiter':maxiter} )

print(result)
print("Output of the result", result.x)
print("Done")