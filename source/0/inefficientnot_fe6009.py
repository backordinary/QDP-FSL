# https://github.com/timholthuijsen/QuantumProject/blob/bf410651cbdd41af4552a36be480b10bc412a40b/InefficientNot.py
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 00:09:39 2021

@author: timho


When using this code, just run plotNOTS(10) and it
should do everything. May take a while depending on backend

Alternatively, plotNOTS2(10) can be used to try my alternative approach to
implementing multiple NOT gates. However, this approach has not shown to reduce
probability based on the number of NOT gates used.
"""
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt
from qiskit import *
from qiskit import IBMQ
from qiskit_gates import NAND
import json
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

provider = IBMQ.load_account()

def lb():
    return least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 2 and 
                                        not x.configuration().simulator and x.status().operational==True))

#backend = lb()
#backend = provider.get_backend('ibmq_athens')
backend = Aer.get_backend('qasm_simulator')
print(backend)

def inefficientNOT(inefficiencies, inp, layout = [0], n_times = 100):
    """
    This NOT gate includes an int number of inefficiencies, which determines 
    how many qc.x gates the NOT will apply. Note that inefficiencies
    should always be an odd number for the gate to function as a not
    
    """
    qc = QuantumCircuit(1, 1) # A quantum circuit with a single qubit and a single classical bit
    qc.reset(0)
    
    if inp=='1':
        qc.x(0)
    
    qc.barrier()
    """We apply a predefined number of inefficiencies.
    Unfortunately, and contrary to our expectations, this did not yield a decreased
    accuracy result when running on a quantum computer. For the inefficiency-reduced accuracy,
    we will use the NOT() function instead"""
    for i in range(inefficiencies):
        print(i+1, "x gates have been added")
        qc.x(0)
    #barrier between gate operation and measurement
    qc.barrier()
    qc.measure(0,0)
    trial = qc.measure(0,0)
    
    qc_trans = transpile(qc, backend, initial_layout=layout, optimization_level=3)
    job = execute(qc_trans, backend, shots= n_times)
    print(job.job_id())
    job_monitor(job)
    
    output = job.result().get_counts()
    
    return output

def NOT(inp):
    "A NOT gate."
    qc = QuantumCircuit(1, 1)
    qc.reset(0)
    if inp=='1':
        qc.x(0)   
    qc.barrier()
    qc.x(0)
    qc.barrier()
    qc.measure(0,0)
    #qc.draw('mpl')
    job = execute(qc, backend, shots=1, memory=True)
    print(job.job_id())
    job_monitor(job)
    output = job.result().get_memory()[0]
    #output = job.result().get_counts()    
    return output
    

def makeresults(inefs, number):
    "A function that runs inefs*not(0) a number times"
    results = {}
    #Define which inefficiency we're at
    for inef in range(1,inefs,2):
        print('starting with inefficiency', inef)
        #Set number of success for specific inef to 0
        results[str(inef)]=0
        #Do the following 'number' times:
        for i in range(number):
            output = '0'
            #Apply the NOT gate inef times:
            for iteration in range(inef):
                output = NOT(output)
            if output == '1':
                results[str(inef)]+=1
    return results
    
def makeplotdata(results):
    "Plot the results from makeresults()"
    plotdata = []
    for gates in results:
        amount = results[gates]
        for success in range(amount):
            plotdata.append(gates)
    return plotdata

#When using this code, just run the plotNOTS(6) function and it should do everything
#The problem with this part is that the number of qc job requests takes a huge amount of time
def plotNOTS(inefs, number = 10):
    "The function to plot the inefficiency accuracy effects"
    results = makeresults(inefs, number)
    plotdata = makeplotdata(results)
    plt.style.use('ggplot')
    plt.hist(plotdata, bins=10)
    plt.title('Decreasing NOT-gate accuracy')
    plt.xlabel('Number of Gates')
    plt.ylabel('Probability of being correct')
    plt.show
            



#another attempt at getting results with the inefficientNOT function
def makeresults2(number):
    "Runs 'number' NOT gates with increasing inefficiencies"
    outputs = []
    results = {}
    for inefs in range(1,number,2):
        output = inefficientNOT(inefs,'0')
        results[str(inefs)] = output['1']
    return results

def plotNOTS2(inefs):
    "The function to plot the inefficiency accuracy effects"
    results = makeresults2(inefs)
    plotdata = makeplotdata(results)
    plt.style.use('ggplot')
    plt.hist(plotdata, bins=10)
    plt.title('Decreasing NOT-gate accuracy')
    plt.xlabel('Number of Gates')
    plt.ylabel('Probability of being correct in %')
    plt.show









if False:    
   """ def makeresults(number):
        "Runs 'number' NOT gates with increasing inefficiencies"
    outputs = []
    results = {}
    for inefs in range(1,number,2):
        output = inefficientNOT(inefs,'0')
        results[str(inefs - 1)] = output['1']
    return results
    def run_for_NOT(inefficiencies = 1, n_tries = 100, QuantumComputer ='ibmq_qasm_simulator'):
        backend = provider.get_backend(QuantumComputer)
        output1_all = []
        qc_trans1_all = []
        prob1_all = {}
        layout1 = [0]
        
        worst = 1
        best = 0
        for input1 in ['0','1']:
                qc_trans1, output1 = inefficientNOT(inefficiencies, input1, backend, layout1, n_tries)
                
                output1_all.append(output1)
                qc_trans1_all.append(qc_trans1)
                try:
                    prob = output1[str(int( not input1=='1'))]/ n_tries
                    prob1_all[input1] = prob
                except:
                    pass
                prob1_all[input1 + "_output"] = output1
                
                print('\nProbability of correct answer for inputs',input1)
                print( '{:.2f}'.format(prob) )
                print('---------------------------------')
                
                worst = min(worst,prob)
                best = max(best, prob)
                
        print('')
        print('\nThe highest of these probabilities was {:.2f}'.format(best))
        print('The lowest of these probabilities was {:.2f}'.format(worst))
        
        return(prob1_all)
    
    def plot_not():
        opdict = newdict['NOT']
        bl = opdict['bitlist']
        l0 = opdict['list0']
        l1 = opdict['list1']
        cor = [l1[0], l0[1]]
        wr = [l0[0], l1[1]]
        tot_cor = sum(cor)
        tot_wr = sum(wr)
        ind = np.arange(2)
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        yes = ax.bar(ind, cor, width = 0.25)
        no = ax.bar(ind + 0.25, wr, width = 0.25)
        plt.title('NOT-gate')
        ax.set_xticks(ind+0.25)
        ax.set_xticklabels(bl)
        ax.legend( (yes[0],no[0]), ('correct','wrong') )
        
        return tot_cor, tot_wr
    """
