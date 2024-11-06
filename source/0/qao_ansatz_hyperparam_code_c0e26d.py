# https://github.com/esKemp/Cloud_QAOA/blob/75950bd27c0c8be5241cc360f83cfadccd726678/QAO_Ansatz_HyperParam_Code.py
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import qiskit.providers.aer.noise as noise

from qiskit import IBMQ, Aer
from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel

import networkx as nx

from ansatz import qaoa

matplotlib.rc('xtick', labelsize=18)     
matplotlib.rc('ytick', labelsize=18)
plt.rcParams["font.family"] = "Times New Roman"

# Get 2 qubit error probability and optimizer from command line arguments
prob2 = float(sys.argv[1])
optimizer = sys.argv[2]

# Construct square graph 
G = nx.Graph()
G.add_nodes_from([0,1,2,3])
G.add_edges_from([(0,1),(1,2),(2,3),(3,1)])

# Construct idealized probability distribution
prob_standard = [('00101', 0.99951171875),
 ('01010', 0.000244140625),
 ('00001', 0.0001220703125),
 ('00010', 0.0001220703125)]

# Construct data class for hyperparameter search
from dataclasses import dataclass

@dataclass
class prob_info:
    Lambdas: list
    Ps: list
    optimizer: str
    num_rounds: int
    isNoise: bool
    prob_1: float
    prob_2: float
    probs: dict = None
    SSO_vals: dict = None

# Define function to compute prob distribution
def calculateProbs(probs_specs):

    basis_gates = None
    noise_model = None

    if probs_specs.isNoise:
        # Depolarizing quantum errors
        error_1 = noise.depolarizing_error(probs_specs.prob_1, 1)
        error_2 = noise.depolarizing_error(probs_specs.prob_2, 2)

        # Add errors to noise model
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

        basis_gates = noise_model.basis_gates
    
    probs = {}

    for Lambda in probs_specs.Lambdas:
        for P in probs_specs.Ps:
            for (coupling_map, basis_gates, noise_model) in [(None, basis_gates, noise_model)]:
                print(str(Lambda) + ", " + str(P) + ", " + str(probs_specs.isNoise))
                out = qaoa.solve_mis(P, G, Lambda, probs_specs.isNoise, probs_specs.num_rounds, probs_specs.optimizer, coupling_map, basis_gates, noise_model)
                print("Computed Out")
                circ = qaoa.gen_qaoa(G, P, params=out['x'], barriers=False, measure=True)
                # result = execute(circ, backend=backend, shots=8192).result()
                # result = execute(circ, backend=Aer.get_backend('qasm_simulator'), shots=8192, coupling_map=coupling_map, basis_gates=basis_gates,         
                # noise_model=noise_model).result()
                result = execute(circ, backend=Aer.get_backend('qasm_simulator'), shots=8192, basis_gates=basis_gates, noise_model=noise_model).result()
                counts = result.get_counts(circ)
                total_shots = sum(counts.values())
                prob = [(bitstr, counts[bitstr] / total_shots) for bitstr in counts.keys()]
                prob = sorted(prob, key=lambda p: p[1], reverse=True)
                print("Computed Prob")
                probs[Lambda, P, probs_specs.isNoise] = prob
        
    return probs

# Define functions to compute SSO vals
def convert_Dict(probs):
    probDict = {}
    for (string, num) in probs:
        probDict[string] = num
    return probDict

def SSO(probs1, probs2):
    probs1 = convert_Dict(probs1)
    probs2 = convert_Dict(probs2)

    sso_squared = 0
    for string in list(probs1.keys()):
        if string in probs1 and string in probs2:
            sso_squared += (probs1[string]**0.5)*(probs2[string]**0.5)

    return sso_squared**0.5

def SSO_val(prob_dict, probs_std, Lambdas, Ps, Noises):
    return 4

def SSO_val(probs_specs, probs_std):

    SSO_vals = {}

    for Lambda in probs_specs.Lambdas:
        for P in probs_specs.Ps:
            SSO_vals[Lambda, P, probs_specs.isNoise] = SSO(probs_std, probs_specs.probs[Lambda, P, probs_specs.isNoise])
    
    return SSO_vals

# Compute the prob distribution and SSO vals for three different optimizers
probs_vals = {}

for prob1 in [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]:

    probInput = prob_info([0,1,2,3,4,6,8,10,20,50], [3], optimizer, 20, True, prob1, prob2)
    prob = calculateProbs(probInput)

    probInput.probs = prob

    sso_vals = SSO_val(probInput, prob_standard)

    probInput.SSO_vals = sso_vals

    probs_vals[prob1, prob2, optimizer] = probInput

# Print results in a csv file
file_name = "probs_" + str(prob2) + "_" + str(optimizer) + ".csv"

a_file = open(file_name, "w")

writer = csv.writer(a_file)
for key, value in probs_vals.items():
    writer.writerow([key, value])

a_file.close()