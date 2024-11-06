# https://github.com/ezerc/Qiskit_Projects/blob/b41e9bf04591bc08d3c6a4f966da6e0b46b6009c/QAOA_Number_Partition_Problem/number_partition_qaoa.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 09:26:16 2021

@author: Ezequiel
"""

import sys
from qiskit import *
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
import numpy as np
pi = np.pi
from fractions import Fraction
from math import gcd
from numpy.random import *
from statistics import mode
from collections import Counter
import timeit
import time
from scipy.optimize import minimize


def fully_connected_model(S):
    ''' Given a set, this function constructs the edges of a fully connected graph, and an 
    interaction matrix based on the Ising Hamiltonian formulation of the number partition
    problem. Our convention is such that in this matrix is upper triangular.
    
    Args:
        -S: set containing all elements of the number partition problem
    '''
    
    # Number of nodes in the graph
    n = len(S)

    J = [[] for _ in range(n)]
    E = []
    
    
    for i in range(n):
        for j in range(n):
        
            # Diagonal terms
            if j == i:
                J[i].append(S[i]**2)
                
            # Upper triangular section contains all interactions
            elif j > i:
                J[i].append(2*S[i]*S[j])
                E.append([i,j])
                
            # Bottom triangular section set to 0
            elif j < i:
                J[i].append(0)
    
    return E,J

class number_partition_qaoa:
    ''' This class is used to solve the number partition problem using the QAOA algorithm.
    
    By taking the set and the edges and interaction matrix of the effective fully-connected
    Hamiltonian, the class constructs an ansatz for the QAOA state and uses it to obtain a 
    solution to the number partition problem. The ansatz is constructed by creating initial
    guesses of the angle parameters using a grid search method to obtain the parameters that 
    minimize the energy. These are then used as a seed for a Nerlder-Mead optimization 
    algorithm, which returns an optimal set of parameters that minimize the energy. Using these
    parameters, the QAOA circuit is run and a solution to the number partition problem is
    searched on the top five measurement outcomes.  
    
    Attributes:
    
        -S (list): Set to be partitioned
        -E (list): Edges in the effective Ising model
        -J (list): Interaction energy matrix. Needs to be upper triangular
        -nq (int): Number of elements in the set, i.e. number of qubits in the circuit
        -self_energy (int): Constant self-energy term in the effective Hamiltonian ()
            
    Methods:
    
        -zz_gate(circuit,q0,q1,angle):
            This function implements a two-qubit gate of the form Diag[1,exp(-i*angle),exp(-i*angle),1].
            This corresponds to exponentiating ZZ-type interaction terms between two spins, with
            H_{ij} = angle*Z_{i}*Z_{j}.
             
        -ising_circuit(circuit,qr,p,G,B):
            This function creates a Quantum Circuit which generates a QAOA ansatz, with cost
            function an Ising Hamiltonian, a mixing Hamiltonian consisting of transverse fields 
            in the x direction and a tunable amount of layers for the unitaries.
        
        -compute_energy(state):
            This function computes the energy associated of a state, given the parameters
            of an Ising Hamiltonian.
            
        -measured_energy(measurements,num_samples):
            This function computed the expectation value of the energy from a set of measurements 
            and their counts.
            
        -states_and_energies():
            This function computes the list of possible state and their associated energies.
            
        -ising_qaoa_step(parameters,p,num_samples,output = 'energy'):
            This function runs the QAOA circuit and computes the energy expectation
            value for a given set of QAOA angles.
            
        -full_grid_search(p_max,num_samples,size):
            This functions performs a grid search of minimum energy over a range of QAOA
            angle parameters.
            
        -run_qaoa(p,num_samples = 20000, grid_num_samples = 2000,grid_size = [90,10]):
            Full run of the QAOA algorithm. Inital guess for angles is obtained from a coarsed
            grained grid search performed layer by layer. 
            
        -state_results(measurements,num_samples,verbose = True):
            This function orders the resulting measured states by outcome frequency and 
            prints them if required.
            
        -num_part_result(state,verbose = True):
            This function splits a set into to subsets by comparing the position of a given 
            element to the binary value of a string state in that same position. It also computes
            the difference between the sum of the two elements.
            
        -analyze_result(top_states):
            This function searches for the minimal difference between the sum of two sets,
            which result from partitioning an original set according to a binary string.      
    '''
    
    def __init__(self,S,E,J):
        '''Initializes number_partition_qaoa class.
        
        Args:
            -S (list): Set to be partitioned
            -E (list): Edges in the effective Ising model
            -J (list): Interaction energy matrix. Needs to be upper triangular
        '''
        
        self.S = S
        self.J = J
        self.E = E
        
        # Number of elements in the set, i.e. number of qubits in the circuit
        self.nq = len(S)
        
        # Constant self-energy term in the effective Ising Hamiltonian
        self.self_energy = sum([J[j][j] for j in range(len(S))])
        
    
    ################     CIRCUIT      #####################
    
    def zz_gate(self,circuit,q0,q1,angle):
        ''' This function implements a two-qubit gate of the form Diag[1,exp(-i*angle),exp(-i*angle),1].
        This corresponds to exponentiating ZZ-type interaction terms between two spins, with
        H_{ij} = angle*Z_{i}*Z_{j}.
    
        Args:
            -circuit(QuantumCircuit): Circuit on which the gate is applied
            -q0 (Qubit): First qubit on which the gate acts
            -q1 (Qubit): Second qubit on which the gate acts
            -angle (float): Rotation angle
        
        '''
    
        # Combination of CNOTs and rotation yields ZZgate
        circuit.cx(q0,q1)
        circuit.rz(angle,q1)
        circuit.cx(q0,q1)
    
    
    
    def ising_circuit(self,circuit,qr,p,G,B):
        ''' This function creates a Quantum Circuit which generates a QAOA ansatz, with cost
        function an Ising Hamiltonian, a mixing Hamiltonian consisting of transverse fields 
        in the x direction and a tunable amount of layers for the unitaries.
    
        Args:
            -circuit (QuantumCircuit): Quantum circuit on which gates are applied
            -qr (QuantumRegister): Specific quantum register on which the circuit it built
            -p (int): Number of layers in the ansatz
            -G (list): Set of parameters associated with Ising Hamiltonian unitary
            -B (list): Set of parameters associated with Mixing Hamiltonian unitary
        '''
    
        # Act with unitaries on each layer
        for layer in range(p):
        
            # Define the angles for the unitaries in the specific layer
            gamma = G[layer]
            beta = B[layer]
        
            # Action of ZZ gates from U(C,gamma) 
            for j in range(len(self.E)):
                self.zz_gate(circuit,qr[(self.nq-1)-self.E[j][0]],qr[(self.nq-1)-self.E[j][1]],\
                             2*gamma*self.J[self.E[j][0]][self.E[j][1]])

            # Action of Rx rotations from U(B,beta) 
            for j in range(self.nq):
                circuit.rx(2*beta,qr[j])

        
        
    ##########  ENERGY COMPUTATION #############
    
    def compute_energy(self,state):
        ''' This function computes the energy associated of a state, given the parameters
        of an Ising Hamiltonian.
    
        Args:
            -state (str): State whose energy we want to compute
            
        Returns:
            - (float): Total energy associated with the state
        '''
    
        # List of physical spins to be extracted from measured state in computational basis
        spin_config = [(-1)**int(state[j]) for j in range(self.nq)]
        
        
        # Compute energy contribution from interaction
        int_energy = 0
    
        for j in range(len(self.E)):     
            int_energy += self.J[self.E[j][0]][self.E[j][1]] * \
                                 spin_config[self.E[j][0]] * spin_config[self.E[j][1]]

        # Add self energy contribution
        return int_energy + self.self_energy




    def measured_energy(self,measurements,num_samples):
        '''This function computes the expectation value of the energy from a set of measurements 
        and their counts.
    
        Args:
            -measurements (dict): Dictionary containing measured states and measurement counts
            -num_samples (int): Total number of samples
            
        Returns:
            - (float): Energy expectation value
        '''
        
        E_avg = 0
    
        # Compute energy contribution to the average from each state and its statistical weight
        for state in measurements.keys():
        
            E_avg += measurements[state] * self.compute_energy(state)
        
    
        return E_avg/num_samples

    def states_and_energies(self):
        ''' This function computes the list of possible state and their associated energies.
    
        Returns:
            -states_energies (dict): Dictionary containing states as keys and their respective 
                                     energies as values
        '''
    
        energies = []
        states = []
    
        for i in range(2**self.nq):
        
            # Configurations in computation basis
            state = format(i,'0'+str(self.nq)+'b')
        
            states.append(state)
        
            # List of physical spins to be extracted from measured state in computational basis
            spin_config = [(-1)**int(state[j]) for j in range(self.nq)]

        
            # Interaction energy contribution
            int_energy = 0
        
            for j in range(len(self.E)):

                int_energy += self.J[self.E[j][0]][self.E[j][1]] * \
                                spin_config[self.E[j][0]] * spin_config[self.E[j][1]] 
            
            # Add self energy contribution    
            energies.append(self.self_energy + int_energy)  
        
        states_energies = dict(zip(states,energies))
    
        return states_energies
        
    
        
    ##########      QAOA       #############
        
        
    def ising_qaoa_step(self,parameters,p,num_samples,output = 'energy',use_qpu = False):
        ''' This function runs the QAOA circuit and computes the energy expectation
        value for a given set of QAOA angles.
        
        Args:
            -parameters (list): Contains all angles to be used in QAOA. The first half
                                of the list contains the ising Hamiltonian angles and
                                the second half the mixing Hamiltonian angles
            -p (int): Number of layers in the ansatz
            -num_samples (int): Number of times the circuit is run
            -output (str): If 'energy', the function returns the energy expectation 
                           value. If 'full', it also returns the measurement reuslts
            -use_qpu (bool): If True, circuit is run in an IBM qpu
                           
        Returns:
            -E_exp_val (float): Energy expectation value. It is always returned
            -measurements (dict): Dictionary containing measurment outcomes and their
                                  frequency. Only returned if output = 'full'
        
        '''

        # Initial QAOA parameters

        G = [parameters[j] for j in range(p)]
        B = [parameters[j] for j in range(p,2*p)]

        # Circuit definition
        qr = QuantumRegister(self.nq)
        cr = ClassicalRegister(self.nq)
        ising_qaoa = QuantumCircuit(qr,cr)


        # Initialization on symmetric superposition
        ising_qaoa.h(range(self.nq))

        # Generate Ising circuit implementing one layer of U(C,gamma) and U(B,beta)
        self.ising_circuit(ising_qaoa,qr,p,G,B)

        # Perform measurements on all qubits
        for q in range(self.nq):
            ising_qaoa.measure(q,q)

        # Run simulation with QPU
        if use_qpu:
            
            # Select QPU and shot number
            if self.nq > 5:
                print('Given the number of qubits, the default QPU is \'ibmq_16_melbourne \' ')
                QPU = 'ibmq_16_melbourne'
            else:
                print('Insert the name of qpu to be used, e.g. \'ibmq_athens\'')
                QPU = str(input())
            print('')
            print('Number of shot samples wanted (needs to be smaller than 8192):')
            shots = int(input())
            
            # Circuit execution given the specific QPU and the number of shots received
            provider = IBMQ.get_provider(hub = 'ibm-q')
            simQ = provider.get_backend(QPU)
            print('\nExecuting QPU')
            qaoa_job = execute(ising_qaoa, backend = simQ, shots = shots)
            print('Job id: {}'.format(qaoa_job.job_id()))
            job_monitor(qaoa_job)
            qaoa_result = qaoa_job.result()
            measurements = qaoa_result.get_counts()
            
            # Compute expectation value of the energy
            E_exp_val = self.measured_energy(measurements,shots)
        
        # Run simulation with Qasm simulator
        else:
            # Circuit execution
            sim = Aer.get_backend('qasm_simulator')
            qaoa_result = execute(ising_qaoa, backend = sim, shots = num_samples).result()
            measurements = qaoa_result.get_counts()
        
            # Compute expectation value of the energy
            E_exp_val = self.measured_energy(measurements,num_samples)
    
        if output == 'full':
            return E_exp_val,measurements
    
        elif output == 'energy':
            return E_exp_val
    
    
    
    
    
    def full_grid_search(self,p_max,num_samples,size):
        ''' This functions performs a grid search of minimum energy over a range of QAOA
        angle parameters. 
        
        This is done layer by layer, and the angles of previous layers are fixed to the 
        values that minimize the energy when searching that parameter space. By symmetry 
        of the Hamiltonian, the Ising Hamiltonian angles scanned from 0 to pi, and the 
        mixing Hamiltonian angles from 0 to pi/2.
        
        Args:
            -p_max (int): Maximium number of layers to be scanned
            -num_samples (int): Number of times the circuit is run
            -size (list): Containes the number of grid points to be scanned. The first 
                           element coresponds to the gamma direction and the second to 
                           the beta direction
                           
        Returns:
            -gamma_opt (list): Best angles for the Ising Hamiltonian unitaries
            -beta_opt (list): Best angles for the mixing Hamiltonian unitaries
        '''
    
        # Initialize lists containing optimal values
        gamma_opt = [0 for _ in range(p_max)]
        beta_opt  = [0 for _ in range(p_max)]
        
        # Extract grid sizes for each angle
        size_g = size[0]
        size_b = size[1]
        
        # Define grid ranges
        gamma_grid = [pi*j/size_g for j in range(size_g)]
        beta_grid  = [pi/2*j/size_b for j in range(size_b)]
    
        # Start timer
        start = time.time()
    
        # Layer by layer scan
        for p in range(1,p_max+1):
            
            # Initialize energy at a high value and a counter to keep track of progress
            E_min = 10000
            counter = 0
            
            # Initiate grid search
            for g in gamma_grid:
                
                # Report progress
                counter +=1
                print('Grid search process (p={}): {}%  '.format(p,int(np.floor(counter*100/size_g))),end = '\r')
    
                # Set angle for QAOA run
                gamma_opt[p-1] = g
    
            
                for b in beta_grid:
                
                    # Set angle for QAOA run
                    beta_opt[p-1] = b
            
                    parameters = gamma_opt + beta_opt
                    
                    # Run circuit and obtain energy expectation value
                    avg_energy = self.ising_qaoa_step(parameters,p,num_samples)
           
                    
                    # If computed energy is a new minimum store its value and parameter values
                    if avg_energy < E_min:
            
                        g_min = round(g,3)
                        b_min = round(b,3)
                        E_min = avg_energy
            
            # Store optimal values of layer p
            gamma_opt[p-1] = g_min
            beta_opt[p-1]  = b_min
    
        # Finalize timer
        end = time.time()
        t = round(end-start)
        
        # Report results
        print('Grid search has taken {}min {}s     \n'.format(t//60,t%60))
        print('Optimization seed parameters: \u03B3 = {}, \u03B2 = {} - E = {}\n'\
              .format(gamma_opt,beta_opt,E_min))
        
        return gamma_opt,beta_opt




    def run_qaoa(self,p,num_samples = 20000, grid_num_samples = 2000,grid_size = [80,20],\
                 show_states = False, use_qpu = False):
        '''Full run of the QAOA algorithm. Inital guess for angles is obtained from a coarsed
        grained grid search performed layer by layer. 
        
        This guess is used as a seed for a Nelder-Mead optimization procedure. Optimal values 
        are used to run the circuit and compute answer. This is carried out by analyzing the 
        top six measured states, partitioning the original set accordingly, and checking the 
        difference between the sums of the two sets. The ones yielding the minimal sum are 
        returned as the answer.
        
        
        Args:
            -p (int): Number of layers in the ansatz
            -num_samples (int): Number of times we run the circuit when optimizing parameters
            -grid_num_samples (int): Number of times we run the circuit in the grid search
            -show_states (bool): If True print measured states and their percentage frequency
            -use_qpu (bool): If True the final solution will be computed using a QPU
            
        Returns:
            -S1 (list): First set resulting from the partition
            -S2 (list): Second set resulting from the partition
            -states (dict): Measurement outcomes and their frequency
        '''

        # Initial QAOA parameters
        G, B = self.full_grid_search(p,grid_num_samples,grid_size)
        parameters = G + B
    
        # Initialize timer for QAOA
        start = time.time()
    
        # Arguments to perform optimization of QAOA parameters
        argum = tuple([p,num_samples])


        # Initialize parameter optimization
        print('\nInitiating QAOA parameter optimization\n')
        qaoa_result = minimize(self.ising_qaoa_step,parameters, method = 'nelder-mead',\
                               args = argum, options = {'fatol' : 10**(-8)})
    
    
        # Inform the time the optimization took
        end = time.time()
        t = round(end - start)
        print('Optimization took {}min {}s \n'.format(t//60,t%60))
    

        # Run QAOA using results for optimization
        gamma_opt = [round(qaoa_result['x'][j],3) for j in range(p)]
        beta_opt  = [round(qaoa_result['x'][j],3) for j in range(p,2*p)]

        print('Running QAOA with optimal parameters...\n')
        E_opt,measurements = self.ising_qaoa_step(qaoa_result['x'],p,num_samples,'full',use_qpu)
        
        # If QPU used retrieve the number of samples taken
        if use_qpu:
            num_samples = sum((measurements.values()))

        # Report information of the final run with optimal parameters
        print('Solution:    Energy = {} \n'.format(E_opt))

        for j in range(p):
    
            if j == 0:
                print('Parameters:   \u03B3 = {}     \u03B2 = {}   (p={})'.format(gamma_opt[j],beta_opt[j],j+1))
        
            else:
                print('              \u03B3 = {}     \u03B2 = {}   (p={})'.format(gamma_opt[j],beta_opt[j],j+1))
    
        print('')

        # Report measured states ordered by measurement frequency
        states = self.state_results(measurements,num_samples,show_states)
    
        # Select top five measured states for post-processing (or all if set is of size 2)
        if self.nq <= 2:
            top_states = list(states)
        else:
            top_states = list(states)[0:5]
    
        # Check which partition yields a minimal sum
        S1,S2 = self.analyze_result(top_states)
    
        return S1,S2,states
    
    
    
    
    ###################     POST PROCESSING     ###########################
    
    
    def state_results(self,measurements,num_samples,verbose = True):
        ''' This function orders the resulting measured states by outcome frequency and 
        prints them if required.
    
        Args:
            -measurements (dict): Dictionary containing state and number of counts 
                                as a result of a circuit execution     
            -num_samples (int): Total number of circuit runs in the execution
            -verbose (bool): If True, print the ordered list of results 
        '''
    
        # Order results from the measurements dictionary
        ordered_results = dict(sorted(measurements.items(), key=lambda item: item[1], reverse = True))
    
        if verbose:
        
            # Print ordered list of states
            for state in ordered_results:    
                print('State: |{}>   Probability: {}% \n'.format(state,\
                                                             round(measurements[state]/num_samples*100,2)))
        
        return ordered_results
    
    
    
    def num_part_result(self,state,verbose = True):
        ''' This function splits a set into to subsets by comparing the position of a given 
        element to the binary value of a string state in that same position. It also computes
        the difference between the sum of the two elements.
    
        Args:
            -S (list): Set to be decomposed
            -state (str): Binary string, resulting from QAOA computation
            -verbose (bool): If True, print the partitioned sets and their difference
            
            
        Returns:
            -S1 (list): First set resulting from the partition
            -S2 (list): Second set resulting from the partition
            -diff (int): Difference between the sums of each set
        '''
    
        S1 = []
        S2 = []
    
        for k in range(len(state)):
        
            # Store element in first set if position associated with 0
            if state[k] == '0':
                S1.append(self.S[k])
        
            # Store element in second set if position associated with 1
            else:
                S2.append(self.S[k])
    
        # Sum difference
        diff = abs(sum(S1)-sum(S2))
    
        # Report partitioned sets and their difference
        if verbose:
    
            print('\nThe set can be split in two sets: \n')
            print('S1 = {}'.format(S1))
            print('S2 = {} \n'.format(S2))
            print('with minimal difference | \u2211 S1 - \u2211 S2 | = {}'.format(diff))
    
    
        return S1,S2,diff




    def analyze_result(self,top_states):
        ''' This function searches for the minimal difference between the sum of two sets,
        which result from partitioning an original set according to a binary string.
        
        Args:
            -top_states (list): List of binary strings to be used for set partitioning
            
        Returns:
            -S1 (list): First set resulting from partition with minimal difference
            -S2 (list): Second set resulting from partition with minimal difference
        
        '''
        
        # Initialize difference at a high value
        diff_min = 10000
    
        # Search for minimal difference by analyzing each state 
        for j in range(len(top_states)):
        
            # Obtain difference from the partition without printing info
            _,_,diff = self.num_part_result(top_states[j],False)
            
            # If difference is new minimal value, store its value and the corresponding state
            if diff < diff_min:
            
                state = top_states[j]
                diff_min = diff
                
        # Use best state to report the final answer
        print('The final answer is state |{}> \n'.format(state))
        S1,S2,_ = self.num_part_result(state)
    
        return S1,S2