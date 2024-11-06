# https://github.com/jeanpierre-feng/Projects/blob/60ccf20e23e6dcd2f3b858e460d28676902ba41e/Quantum%20computing/benchmark.py
# ===== Quantum Computing Project =====
# Author : Loris BERT
# School : CPE Lyon
# Date : 05/04/2020
# =====================================

import time
import importlib

import numpy as np
import matplotlib.pyplot as plt

import qiskit
from qiskit.ignis.characterization.coherence import t1_circuits, t2star_circuits, T1Fitter, T2StarFitter




def T1Benchmark (noise_model_file, gate_time=0.1, t=[], N=60, shots=2048, display=False):
    """
    Return:
    	- dict_coherence_times (dict)
    
    Parameters
        - noise_model_file (str)
        - gate_time=0.1 (float)
        - t=[] (list)
        - N=60 (int)
        - shots=2048 (int)
        - display=False (bool)
    """

    T0 = time.time()
    
    # === Import ===
    print("Import")
    
    if '.py' in noise_model_file:
        noise_model = importlib.import_module(noise_model_file[:-3])
    else:
        noise_model = importlib.import_module(noise_model_file)
    
    
    # === Circuit ===
    print("Circuit")
    
    num_of_gates = (np.linspace(0, 1200, N)).astype(int)
    qubits = list(range(5))
    
    t1_circs, t1_xdata = t1_circuits(num_of_gates,
                                     gate_time,
                                     qubits)
    
    # === Noise model ===
    print("Noise model")
    if t != []:
        t1_noise_model = noise_model.t1_noise_model(gate_time, t)
    else:
        t1_noise_model = noise_model.t1_noise_model(gate_time)

        
    # === Simulation ===
    print("Simulation")
    
    backend = qiskit.Aer.get_backend('qasm_simulator')
    
    t1_backend_result = qiskit.execute(t1_circs, 
                                       backend, 
                                       shots=shots,
                                       noise_model=t1_noise_model, 
                                       optimization_level=0).result()
    
    # === Fitting ===
    print("Fitting")
    
    t1_fit = T1Fitter(t1_backend_result, 
                      t1_xdata, 
                      qubits,
                      fit_p0=     [1,  50,  0],
                      fit_bounds=([0,   0, -1], 
                                  [2, 150,  1]))
    
    

    list_coherence_times = t1_fit.time()
    dict_coherence_times = { 'Q'+str(i) : list_coherence_times[i]*10**(-6) for i in range(len(list_coherence_times) ) }

    
    # === Display ===
    print("Display\n")
    
    if display:
        print("      | T1 exp (µs)")
        print("---------------------")
        for i in range(len(list_coherence_times)):
            print("  Q{}  |   {}".format(i, round(t1_fit.time()[i],2)))
            print("---------------------")
            

        plt.figure(figsize=(15, 15))
        ax = plt.subplot(3, 2, 0+1)
        t1_fit.plot(0, ax=ax)
        ax = plt.subplot(3, 2, 1+1)
        t1_fit.plot(1, ax=ax)
        ax = plt.subplot(3, 2, 2+1)
        t1_fit.plot(2, ax=ax)
        ax = plt.subplot(3, 2, 3+1)
        t1_fit.plot(3, ax=ax)
        ax = plt.subplot(3, 2, 4+1)
        t1_fit.plot(4, ax=ax)
        plt.show()
        
    print()
    
    T1 = time.time()
    print("Running time:", round(T1-T0, 2), "s")
    print()
    
    return dict_coherence_times






def T2Benchmark (noise_model_file, gate_time=0.1, t=[], N=60, shots=2048, display=False):
    """
    Return:
    	- dict_coherence_times (dict)
    
    Parameters
        - noise_model_file (str)
        - gate_time=0.1 (float)
        - t=[] (list)
        - N=60 (int)
        - shots=2048 (int)
        - display=False (bool)
    """

    T0 = time.time()

    # === Import ===
    print("Import")

    if '.py' in noise_model_file:
        noise_model = importlib.import_module(noise_model_file[:-3])
    else:
        noise_model = importlib.import_module(noise_model_file)


    # === Circuit ===
    print("Circuit")

    num_of_gates = (np.linspace(0, 1200, N)).astype(int)
    qubits = list(range(5))


    t2star_circs, t2star_xdata, osc_freq = t2star_circuits(num_of_gates, 
                                                           gate_time, 
                                                           qubits, 
                                                           nosc=4)


    # === Noise model ===
    print("Noise model")
    if t != []:
        t2_star_noise_model = noise_model.t2_star_noise_model(gate_time, t)
    else:
        t2_star_noise_model = noise_model.t2_star_noise_model(gate_time)


    # === Simulation ===
    print("Simulation")

    backend = qiskit.Aer.get_backend('qasm_simulator')

    t2star_backend_result = qiskit.execute(t2star_circs, 
                                           backend, 
                                           shots=shots,
                                           noise_model=t2_star_noise_model, 
                                           optimization_level=0).result()


    # === Fitting ===
    print("Fitting")

    t2star_fit = T2StarFitter(t2star_backend_result, 
                              t2star_xdata, 
                              qubits,
                              fit_p0=     [ 0.5,  50,   osc_freq,      0,  0.5],
                              fit_bounds=([-0.5,   0,          0, -np.pi, -0.5],
                                          [ 1.5, 100, 2*osc_freq,  np.pi,  1.5]))

    list_coherence_times = t2star_fit.time()
    dict_coherence_times = { 'Q'+str(i) : list_coherence_times[i]*10**(-6) for i in range(len(list_coherence_times)) }


    # === Display ===
    print("Display\n")

    if display:
        print("      | T2* exp (µs)")
        print("---------------------")
        for i in range(len(list_coherence_times)):
            print("  Q{}  |   {}".format(i, round(t2star_fit.time()[i],2)))
            print("---------------------")


        plt.figure(figsize=(15, 15))
        ax = plt.subplot(3, 2, 0+1)
        t2star_fit.plot(0, ax=ax)
        ax = plt.subplot(3, 2, 1+1)
        t2star_fit.plot(1, ax=ax)
        ax = plt.subplot(3, 2, 2+1)
        t2star_fit.plot(2, ax=ax)
        ax = plt.subplot(3, 2, 3+1)
        t2star_fit.plot(3, ax=ax)
        ax = plt.subplot(3, 2, 4+1)
        t2star_fit.plot(4, ax=ax)
        plt.show()

    print()

    T1 = time.time()
    print("Running time:", round(T1-T0, 2), "s")
    print()
    
    return dict_coherence_times


