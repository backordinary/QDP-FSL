# https://github.com/Jonastjep/QuantumComputing/blob/3a6a67ccedfbdc259c0a78bbc965e4bd7499029e/Final_Code/scripts/snr_finding.py
from grover_functions import *
from qiskit import QuantumCircuit, assemble, Aer, QuantumRegister, ClassicalRegister, AncillaRegister, transpile
from qiskit.visualization import plot_bloch_multivector, plot_histogram, array_to_latex
import math
from math import log, ceil, floor
import plotly.express as px
import numpy as np
import timeit
import time
import gc #garbage colector



t0_0 = time.perf_counter()

M = 2**17

snrs_ = np.load('SNRs_signal_spins.npy')
if len(snrs_)>M:
    snrs = snrs_[::len(snrs_)//M]
    if len(snrs)>M:
        snrs = snrs[:M]
    elif len(snrs)<M:
        snrs__ = snrs_[1:][::len(snrs)//M][:M-len(snrs)]
        snrs = np.concatenate((snrs,snrs__))
else:
    snrs = snrs_

###################################### SETTING UP THE SEARCH ################################################
nb_break = 2**7 #size of blocks
broken_snrs = [snrs[i:i + nb_break] for i in range(0, len(snrs), nb_break)]

data_arr_full = np.rint(broken_snrs).astype(np.int64)

thresh = 6

with open('output_dicts.py', 'w') as f:
    f.write(f"# This file stores the values from the grover iteration")

with open('output_dicts.py', 'a') as f:
    f.write(
'''
import numpy as np

M = 2**17

snrs_ = np.load('SNRs_signal_spins.npy')
if len(snrs_)>M:
    snrs = snrs_[::len(snrs_)//M]
    if len(snrs)>M:
        snrs = snrs[:M]
    elif len(snrs)<M:
        snrs__ = snrs_[1:][::len(snrs)//M][:M-len(snrs)]
        snrs = np.concatenate((snrs,snrs__))
else:
    snrs = snrs_

nb_break = 2**7 #size of blocks
broken_snrs = [snrs[i:i + nb_break] for i in range(0, len(snrs), nb_break)]
''')

#############################################################################################################
loop_nb = 0

with open('output_dicts.py', 'a') as f:
    f.write(f"dict_val = {{\n")

for data_arr in data_arr_full:
    print(f"\nLoop {loop_nb+1}; Indices {nb_break*loop_nb} to {nb_break*(loop_nb+1)}")

    #These numbers are calculated directly from the data array
    n = ceil(log(len(data_arr), 2)) #number of index qbits
    len_b =  ceil(log(max(data_arr)+1, 2))#length of the bitstrings to compare
    m = 5*len_b #number of qbits needed for the comparision

    i_reg = QuantumRegister(n, 'd') #Creates the data registry
    val_reg = QuantumRegister(m-1, 't') #Creates the template registry
    anc = QuantumRegister(1, 'ancilla') #Creates the ancillary registry for the oracle
    cr = ClassicalRegister(n, 'c') #Creates the classical bit measurment output

    qc = QuantumCircuit(i_reg, val_reg, anc, cr) #Makes the circuit with these qbits as input

    encode = encode_qc(n,len_b,data_arr).to_gate()
    encode.label = "Encoding database"

    ####### State initialization #########
    qc.h(range(n))

    #initiation of ancillary to |->
    qc.x(anc)
    qc.h(anc)

    ##### SETTING THE THRESHOLD ######

    set_thresh(qc, n, len_b, thresh)

    qc.barrier()

    #ENCODING
    #The arrays are made with list comprehentions to make something that looks like [0,1,2,n+16,n+11,n+6,n+1],
    #which connects the database (the first values) to the comparators (the values with n+...)
    qc.append(encode, [x for x in range(n)]+[n+x*5+1 for x in reversed(range(len_b))])#had to reverse the order how the qbits were entered, as the two parts (database and conparator) work in reverse

    comp_qc = n_comparator(len_b).to_gate()
    comp_qc.label = "Comparator"

    qc.append(comp_qc, range(n,n+m))

    qc.append(encode.reverse_ops(), [x for x in range(n)]+[n+x*5+1 for x in reversed(range(len_b))])

    qc.barrier()

    #DIFFUSION OP
    qc.h(range(n))
    qc.x(range(n-1))

    qc.x(n-1)
    qc.h(n-1)
    qc.mct(list(range(n-1)), n-1)
    qc.h(n-1)
    qc.x(n-1)

    qc.x(range(n-1))
    qc.h(range(n))
    qc.barrier()

    qc.measure([x for x in range(n)],cr)

    t0 = time.perf_counter()

    ############# CIRCUIT SIMULATION #################
    aer_sim = Aer.get_backend('aer_simulator')
    aer_sim.set_options(device='CPU')

    transpiled_qc = transpile(qc, aer_sim)
    shots = 2048 #we repeat the simulation 2048 times
    job = aer_sim.run(transpiled_qc)


    t1 = time.perf_counter()
    print(f'Loop time: {t1 - t0}\n')

    hist = job.result().get_counts() #simulation output
    sort_hist = sorted(hist.items()) #So that the plotting puts everything in the same increasing order of basis
    n_hist = {k:v for k,v in sort_hist}

    results = {'val':n_hist.keys(),'count':n_hist.values()}

    trig = []
    for i,j in zip(results['val'],results['count']):
        if int(j) > 20:
            trig.append({'index_dec':int(i,2),'index_bin':i,'val':data_arr[int(i,2)],'counts':j})

    for t in trig:
        print(f"Index {t['index_dec']} holds the value {data_arr[t['index_dec']]}, which is above the chosen threshold.")

    f = open('output_dicts.py', 'a')
    f.write(f"{loop_nb}:{trig}, # Indices {nb_break*loop_nb} to {nb_break*(loop_nb+1)}\n")
    f.close()

    #To limit how many iterations. Comment if want to do the whole array.
    # if loop_nb == 5:
    #     break

    loop_nb += 1
    gc.collect()

with open('output_dicts.py', 'a') as f:
    f.write("}\n\n")
    f.write(
'''
index_array = []
for i in dict_val:
    for sol in dict_val[int(i)]:
        index_array.append(i*nb_break + sol['index_dec'])

for i in index_array:
    print(f"SNR Index: {i};  \t\tSNR value: {round(snrs[i],3)}")
'''
    )

print(f"\nFull program runtime: {time.perf_counter() - t0_0} seconds.")
