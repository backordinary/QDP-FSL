# https://github.com/rikcioli/ToricCode/blob/2cbaaf078e0ff964d0f81b0655ec559bf49d0d09/tools.py
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:35:03 2022

@author: rikci
"""
import matplotlib.pyplot as plt
import json, codecs
import collections
import numpy as np
from IPython.display import display
import qiskit.providers.aer.noise as noise
from qiskit.visualization import plot_histogram
import sys

def plot_counts(counts):
    #plot qiskit hist
    display(plot_histogram(counts))
    
    #plot errorbar hist
    sorted_counts = dict(sorted(counts.items()))
    keylist = list(sorted_counts.keys())
    keycounts = list(sorted_counts.values())
    N_shots = sum(keycounts)
    yerr = np.sqrt(keycounts)/N_shots
    keycounts = np.array(keycounts)/N_shots
    plt.figure(dpi=1200)
    plt.bar(keylist, keycounts, width = 0.5, yerr = yerr)
    plt.ylabel("Probabilities")
    plt.grid(axis = 'y', linestyle = '--')
    plt.xticks(rotation=70)  
    #plt.savefig('D:/Fisica/TESI/Final results/filename.png')
    plt.show()  
    
def show_results(result):
    #DEPRECATED, NOT USEFUL ANYMORE FOR HISTOGRAM ERRORS. CAN BE USED FOR GAUSSIAN
    #ERRORS INSTEAD
    
    #plot qiskit hist
    counts = result.get_counts()
    display(plot_histogram(counts))   
    # Extract memory for error estimates
    sorted_counts = dict(sorted(counts.items()))
    keylist = list(sorted_counts.keys())
    keycounts = list(sorted_counts.values())
    N_shots = sum(keycounts)
    memory = result.get_memory()
    N_bins = 10
    bin_width = N_shots//N_bins
    memory_binned = [memory[i*bin_width : (i+1)*bin_width] for i in range(N_bins)]
    hist_list = [dict(collections.Counter(memory_binned[i])) for i in range(N_bins)]
    
    dict_err = {}
    for key in keylist:
        for i in range(N_bins):
            if not key in hist_list[i]:
                hist_list[i][key] = 0
        key_results = [hist_list[i][key] for i in range(N_bins)]
        key_SDOM = np.std(key_results, ddof=1)/np.sqrt(N_bins)
        dict_err[key] = key_SDOM*N_bins
        
    yerr = list(dict_err.values())
    plt.bar(keylist, keycounts, width = 0.5, yerr = yerr)
    plt.show()
    

def build_noise(dep1 = 5.969e-4, dep2 = 0.02, T1 = 100e3, T2 = 100e3):
    # Error probabilities
    prob_1 = dep1  # 1-qubit gate
    prob_2 = dep2   # 2-qubit gate
    prob_3 = 1 - ((1-prob_2)**13)*((1-prob_1)**12) # 3-qubit gate
    
    # Depolarizing quantum errors
    dep_gate = noise.depolarizing_error(prob_1, 1)
    dep_cx = noise.depolarizing_error(prob_2, 2)
    dep_ccx = noise.depolarizing_error(prob_3, 3)
    
    # Add errors to noise model
    noise_model = noise.NoiseModel()
    #noise_model.add_all_qubit_quantum_error(dep_ccx, ['ccx'])
    if (T1 != 0) or (T2 != 0):
        therm_gate = noise.thermal_relaxation_error(T1, T2, 25)   #exact
        therm_cx = noise.thermal_relaxation_error(T1, T2, 317.8).expand(
                 noise.thermal_relaxation_error(T1, T2, 317.8))     #exact
        therm_reset = noise.thermal_relaxation_error(T1, T2, 950)    #exact
        therm_readout = noise.thermal_relaxation_error(T1, T2, 732.4)    #exact
        noise_model.add_all_qubit_quantum_error(dep_gate.compose(therm_gate), ['id', 'rz', 'sx', 'x', 'h', 'z', 's', 'sdg'])
        noise_model.add_all_qubit_quantum_error(dep_cx.compose(therm_cx), ['cx', 'cp'])
        noise_model.add_all_qubit_quantum_error(therm_reset, ['reset'])
        noise_model.add_all_qubit_quantum_error(therm_readout, ['measure'])
    else:
        noise_model.add_all_qubit_quantum_error(dep_gate, ['id', 'rz', 'sx', 'x', 'h', 'z', 's', 'sdg'])
        noise_model.add_all_qubit_quantum_error(dep_cx, ['cx', 'cp'])
    
    return noise_model


def save_state(state, path = 'C:/Users/rikci/.spyder-py3/TESI/states/', name = 'Z4_3bit'):
    #DEPRECATED, WE USE SCIPY SPARSE INSTEAD
    """
    state: np.array
    """
    lenght = len(state)
    dim = int(np.log2(lenght))
    re_sparse = [(index, re) for index, re in enumerate(state.real) if re!=0]
    im_sparse = [(index, im) for index, im in enumerate(state.imag) if im!=0]
    
    json.dump(re_sparse, codecs.open(path+name+'-real.json', 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)
    json.dump(im_sparse, codecs.open(path+name+'-imag.json', 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)
    json.dump(dim, codecs.open(path+name+'-dim.json', 'w', encoding='utf-8'),
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)
    return

def load_state(path = 'C:/Users/rikci/.spyder-py3/TESI/states/', name = 'Z4_3bit'):
    #DEPRECATED, SEE save_state ABOVE
    real_text = codecs.open(path+name+'-real.json', 'r', encoding='utf-8').read()
    imag_text = codecs.open(path+name+'-imag.json', 'r', encoding='utf-8').read()
    dim = int(codecs.open(path+name+'-dim.json', 'r', encoding='utf-8').read())
    
    re_sparse = json.loads(real_text)
    im_sparse = json.loads(imag_text)
    state = np.zeros(2**dim, complex)
    for index, re in re_sparse:
        state.real[index] = re
    for index, im in im_sparse:
        state.imag[index] = im
    return state


def _swapBits(x, p1, p2):
    """Swap bits of positions p1 and p2 in number x"""
    xor = (((x >> p1) ^ (x >> p2)) & 1)
    return x ^ ( (xor << p1) | (xor << p2))

def _findPerm(target_list):
    """target_list[i] is a value associated to index i. The index represents the
    starting configuration. The value represents the final configuration after
    the permutation. So for example, target_list[0] = 3 means the object of 
    position 0 will become the object of position 3. 
    This is opposite to qiskit initial_layout assignments, where the starting qubits
    take the place of the end qubits."""
    permutations = []
    graph = {
        start: end
        for start, end in enumerate(target_list)
        }
    for start, end in graph.items():
        check_presence = [start in perm for perm in permutations]
        if True not in check_presence:
            if end != start:
                perm = [start, end]      
                newstep = graph[end]
                while newstep != start:
                    perm.append(newstep)
                    newstep = graph[newstep]
                permutations.append(perm.copy())
    return permutations

def _permToSwap(permutations_list):
    swaps = []
    for perm in permutations_list:
        for i in range(len(perm)-1):
            swaps.append(perm[i:i+2])
    return swaps
            

def swapSparseState(sparse_state, assign_list):
    """Swap qubits in a given state (state is a scipy coo_matrix of complex). The assign
    list works this way: the index runs over the starting positions, while the values
    represent the end configuration, after the permutation. assign_list[i] = j
    means that i COPIES j. It DOES NOT take its place, which is how Qiskit instead
    operates. If you want to recover the original Qiskit statevector as it was
    before the transpilation, simply put the initial_layout of the transpiler into
    assign_list."""
    permutations = _findPerm(assign_list)
    swaps = _permToSwap(permutations)
    for index in range(sparse_state.nnz): 
        for swap in swaps:
            sparse_state.col[index] = _swapBits(sparse_state.col[index], swap[0], swap[1])
    return sparse_state


def cleanState(state, eps = sys.float_info.epsilon):
    return np.where(abs(state) > eps, state, 0)

def evaluateFidelity(ensemble, state_ideal):
    fidelity = 0
    state_ideal = state_ideal.getH()
    values = [abs(state.dot(state_ideal).toarray()[0,0])**2 for state in ensemble]
    fidelity = sum(values)/len(ensemble)
    return np.sqrt(fidelity)

def binning(object_list, N_bins):
    bin_width = len(object_list)//N_bins
    binned_items = [object_list[i*bin_width : (i+1)*bin_width] for i in range(N_bins)]
    return binned_items

def fidelityError(ensemble, state_ideal, N_bins):
    fidelity = evaluateFidelity(ensemble, state_ideal)
    subensembles = binning(ensemble, N_bins)
    subfidelities = [evaluateFidelity(ensemble, state_ideal) for ensemble in subensembles]
    SDOM = np.std(subfidelities, ddof = 1)/np.sqrt(N_bins)
    return fidelity, SDOM
    
    