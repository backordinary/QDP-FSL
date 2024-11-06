# https://github.com/quasarUnina/QGS/blob/347e1de671369950bc19e873ece43b72d86d8a3b/periodicity_1.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, IBMQ, BasicAer
from qiskit.circuit.library import MCPhaseGate
import math, random
from qiskit.test.mock import FakeMelbourne
from qiskit.providers.aer import QasmSimulator
import numpy as np


def QuantumWheel_probs(n_qubits, n2f, n_g, beta, draw_circuit=False):
    '''Function that returns the probability to measure the states included
    in number 2 flip (n2f):
    Input:
        - n_qubits = number of qubits
        - n2f = list of individuals to flip ordered by quality.
            NB: n2f must be a list. You can pass any integer between 0 to 2^n - 1,
             the outcome of QuantumWheel_probs is independent from the integers selected
        - n_g = number of iteration for the Grover's algorithm
        - beta = angle for the progressive phase rotation.
        - draw_circuit = set as True to draw the quantum circuit.
    Output:
        Probability to measure individuals inserted in n2f
    '''
    reg = QuantumRegister(n_qubits, name='reg')
    scratch = QuantumRegister(n_qubits-3, name='scratch')
    c_reg = ClassicalRegister(n_qubits, name='output')
    qc = QuantumCircuit(reg, scratch, c_reg)
    qc.h(reg)

    for i in range(n_g):
        ## Flip the marked value
        qc.x(scratch)
        qc.barrier()
        k = 0
        for number_to_flip in n2f:
            x_bits = ~number_to_flip
            x_list = [reg[x] for x in range(len(reg)) if x_bits & (1 << x)]
            if x_list:
                qc.x(x_list)
            customize_phase_flip(qc, math.pi-k*beta, [x for x in reg], scratch[0] )
            if x_list:
                qc.x(x_list)
            k = k+1
        qc.x(scratch)
        qc.barrier()
        Grover(qc, reg, scratch)

    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(qc, backend, shots=1)
    result = job.result()

    outputstate = result.get_statevector(qc, decimals=3)
    # plot_bloch_multivector(outputstate).show()
    total_prob = 0
    marked_prob = 0
    probs = []
    for i, amp in enumerate(outputstate):
        if abs(amp) > 0.000001:
            prob = abs(amp) * abs(amp)
            probs.append(prob)
            total_prob += prob
            #print('|{}> {} probability = {}%'.format(i, amp, round(prob * 100, 5)))
            if i in n2f:
                marked_prob += prob

    if draw_circuit:
        qc.draw(output='mpl').show()

    return probs




###############################################
## Some utility functions: YOU CAN SKIP THIS PART

def Grover(qc, qreg, scratch, condition_qubits=None):
    if condition_qubits is None:
        condition_qubits = []
    qc.h(qreg)
    qc.x(qreg)
    qubits = [x for x in qreg] + condition_qubits
    multi_cz(qc=qc, qubits=qubits, scratch=scratch)
    qc.x(qreg)
    qc.h(qreg)

def multi_cx(qc, qubits, scratch, do_cz=False):
    ## This will perform a CCCCCX with as many conditions as we want,
    ## as long as we have enough scratch qubits
    ## The last qubit in the list is the target.
    target = qubits[-1]
    conds = qubits[:-1]
    #print(conds)
    #print(target, conds)
    scratch_index = 0
    ops = []
    while len(conds) > 2:
        new_conds = []
        for i in range(len(conds)//2):
            #print(i, len(conds)//2 )
            ops.append((conds[i * 2], conds[i * 2 + 1], scratch[scratch_index]))
            new_conds.append(scratch[scratch_index])
            scratch_index += 1
        if len(conds) & 1:
            new_conds.append(conds[-1])
        conds = new_conds
    for op in ops:
        qc.ccx(op[0], op[1], op[2])
    if do_cz:
        qc.h(target)
    if len(conds) == 0:
        qc.x(target)
    elif len(conds) == 1:
        qc.cx(conds[0], target)
    else:
        qc.ccx(conds[0], conds[1], target)

    if do_cz:
        qc.h(target)
    ops.reverse()
    for op in ops:
        qc.ccx(op[0], op[1], op[2])

def customize_phase_flip(qc, angle, reg, scratch):
    qc.mcp(angle, reg, scratch)




def multi_cz(qc, qubits, scratch):
    ## This will perform a CCCCCZ on as many qubits as we want,
    ## as long as we have enough scratch qubits
    multi_cx(qc, qubits, scratch, do_cz=True)

###############################################




import matplotlib.pyplot as plt


#_____________________________________________________________________________________
#SET HYPER PARAMETERS FOR QuantumWheel_probs
#n_qubits = 7
#n2f = [0, 1, 2] #Note that the outcome of QuantumWheel_probs is independent from
#the number inserted in n2f, but it depends only form the lenght of n2f
#n_g = [i for i in range(1,16)]

#beta= math.pi/8
#___________________________________________________________________________________

def plot_distribution(n_qubits, n_marked_states, n_g_max, beta):
    n_g = [i for i in range(1,n_g_max)]
    n2f = [i for i in range(n_marked_states)] #Note that the outcome of QuantumWheel_probs is independent from
    #the number inserted in n2f, but it depends only form the lenght of n2f
    probs_tot = []
    probs_I = [[] for i in range(len(n2f))]
    for iteration in n_g:
        print(str(iteration)+'/'+(str(len(n_g))))
        results_1 = QuantumWheel_probs(n_qubits, n2f, iteration, beta, draw_circuit=False)
        marked_prob=0
        for i in range(len(n2f)):
            marked_prob = marked_prob + results_1[n2f[i]]
            probs_I[i].append(results_1[n2f[i]])
        #print(marked_prob)
        probs_tot.append(marked_prob)

    plt.plot(n_g, probs_tot, label=r'$\Sigma p_i$')
    for i in range(len(n2f)):
        plt.plot(n_g, probs_I[i], label=r'$p_'+str(i)+'$')

    plt.legend()
    plt.xlabel(r'$N_G$')
    plt.ylabel('P')
    plt.title(r'$ \beta = '+ str(beta) + ' $')
    plt.show()

    
    
def plot_distribution_vs_beta(n_qubits, n_marked_states, n_g, beta_max, iteration_):
    if beta_max > math.pi/(n_marked_states+1):
        print('beta_max exceeds beta upper limit')
    betas = [beta_max/i for i in range(1,iteration_)]
    n2f = [i for i in range(n_marked_states)] #Note that the outcome of QuantumWheel_probs is independent from
    #the number inserted in n2f, but it depends only form the lenght of n2f
    probs_tot = []
    probs_I = [[] for i in range(len(n2f))]
    for beta_ in betas :
        print(str(beta_)+'/'+(str(iteration_)))
        results_1 = QuantumWheel_probs(n_qubits, n2f, n_g, beta_, draw_circuit=False)
        marked_prob=0
        for i in range(len(n2f)):
            marked_prob = marked_prob + results_1[n2f[i]]
            probs_I[i].append(results_1[n2f[i]])
        #print(marked_prob)
        probs_tot.append(marked_prob)

    plt.plot(betas, probs_tot, label=r'$\Sigma p_i$')
    for i in range(len(n2f)):
        plt.plot(betas, probs_I[i], label=r'$p_'+str(i)+'$')

    plt.legend()
    plt.xlabel(r'$\beta$')
    plt.ylabel('P')
    #plt.title(r'$  = '+ str(beta) + ' $')
    plt.show()
