# https://github.com/attianopp/CS239---Quantum-Programming/blob/ce61f0646f01b7785ae5dfa06144417e2d037f89/ibmq/submission/Simon.py



#%% Load required modules
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit.compiler import transpile, assemble
import qiskit
from qiskit.visualization import plot_histogram
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)
from qiskit.providers.ibmq.managed import IBMQJobManager

import numpy as np
import os
from qiskit import(QuantumCircuit, execute, Aer)
import random as rd
import time
import matplotlib.pyplot as plt
from datetime import datetime


#%% Define functions


def load_account():
    IBMQ.load_account()
    
    # Get a list of all backends
    provider = IBMQ.get_provider(group='open')
    provider.backends()  # print a list of all available backends
    

def get_all_backends(n_qubits = 1):
    """ Print a list of all backends with at least as many qubits as n_qubits
    Args:
        n_qubits: integer > 0
    Returns:
        backend_list: list of backends
        least_busy_backend: least busy backend
    """
    provider = IBMQ.get_provider(group='open')
    device_list = provider.backends(filters = lambda x: x.configuration().n_qubits >= n_qubits 
                                              and not x.configuration().simulator 
                                              and x.status().operational == True)
    least_busy_backend = least_busy(device_list)
    
    return device_list, least_busy_backend


def get_backends(device_preference = 0, n_qubits = 1):
    """
    Arg:
        n_qubits: int, number of qubits required for the circuit
        device_preference: If zero, return least busy backend.
                            Else, if string, return corresponding backend
        If value is 0, the lest busy backend is chosen for the quantum hardware
    Returns:
        simulator_backend: string denoting simulator backend
        device_backend: string denoting quantum hardware backend
    """
    provider = IBMQ.get_provider(group='open')
    simulator_backend = provider.get_backend('ibmq_qasm_simulator')
    
    if device_preference == 0:
        _, least_busy_backend = get_all_backends(n_qubits)
        device_backend = least_busy_backend
    else:
        device_backend = provider.get_backend(device_preference)
        
    return device_backend, simulator_backend

def binary_add(s1,s2):
        """
        Binary addition (XOR) of two bit strings s1 and s2.
        Args: 
            Two binary strings s1 and s2 of the same length n. They should only consist of zeros and ones
        Returns:
            s: String s given by s = s1 + s2 (of length n) containing only zeros and ones
        """
        if type(s1) != str or type(s2) != str:  #If inputs are not strings, raise an error
            raise ValueError('The inputs are not strings')
        if sum([1 for x in s1 if (x != '0' and x != '1')]) > 0 or sum([1 for x in s2 if (x != '0' and x != '1')]) > 0:
            raise ValueError('Input strings contain characters other than 0s and 1s') 
        if len(s1) != len(s2):
            raise ValueError('Input strings are not of the same length') 
            
        x1 = np.array([int(i) for i in s1])
        x2 = np.array([int(i) for i in s2])
        x = (x1 + x2 ) % 2
        x = ''.join(str(e) for e in x.tolist())
        return x

def dot_product(s1,s2):
        """
        Dot product of strings s1 and s2.
        Args: 
            Two binary strings s1 and s2 of the same length n. They should only consist of zeros and ones
        Returns:
            prod: Integer (0 or 1)
        """
        if type(s1) != str or type(s2) != str:  #If inputs are not strings, raise an error
            raise ValueError('The inputs are not strings')
        if sum([1 for x in s1 if (x != '0' and x != '1')]) > 0 or sum([1 for x in s2 if (x != '0' and x != '1')]) > 0:
            raise ValueError('Input strings contain characters other than 0s and 1s') 
        if len(s1) != len(s2):
            raise ValueError('Input strings are not of the same length') 
            
        x1 = np.array([int(i) for i in s1])
        x2 = np.array([int(i) for i in s2])
        x = np.sum(x1 * x2) %2
        return x


def give_binary_string(z,n):
        """
        Function which takes in an integer input, and outputs the corresponding binary representation
        Args: 
            z: (Integer) z must lie in the set {0,1,...,2**n- 1}
            n: (Integer) Number of bits
        Returns: 
            s: String of length n containing the binary representation of integer z.
        """
        if type(z) != int or type(n) != int:
            raise ValueError('Both arguments must be integers.')
        
        s = bin(z)
        m = n + 2 - len(s)
        if m < 0:
            raise ValueError('z is greater than the 2 raised to n. Need to input a larger value of n') 
        s = s.replace("0b", '0' * m)
        return s 



def generate_random_s(n, seed_val = 0):
        """
        Generates a random bit string of length s
        Args:
            n: integer (length of bitstring)
        KwArgs:
            seed_val: integer. Seed. If = 0, s will be created randomly, else, the seed will be used
        Returns:
            s: string of length n, consisting of 0s and 1s only.
        """
        
        if seed_val != 0:
             rd.seed(seed_val) 
        
        if type(n)!= int or n < 1 :
            return("The input must be a positive integer")
        # select random number 
        z = rd.randint(0, 2**n - 1)
        # convert the number to binary representation
        s = give_binary_string(z,n)
        return s

def s_function(s, seed_val = 0):
        """ Given a string s, create a Simon function f
        Args: 
            s: string of length n. Must contain only zeroes and ones.
        KwArgs:
            seed_val: integer. Seed. If = 0, f will be created randomly, else, the seed will be used
        Output:
            f: list of length 2^n. f[i] contains the value of f corresponding to i. Each element of the list is a string
            f is such that, f(x) = f(y) iff x + y \in {0,s}
        """
        
        if type(s) != str:  #If inputs are not strings, raise an error
            raise ValueError('s should be a string')
        if sum([1 for x in s if (x != '0' and x != '1')]) > 0 or sum([1 for x in s if (x != '0' and x != '1')]) > 0:  
            #check if s contains only zeros and ones
            raise ValueError('s should only contain 0s and 1s') 
        
        if seed_val != 0:
             np.random.seed(seed_val) 
             
        n = len(s)
        N = 2**n
        f_temp = [-1 for i in range(N)]
        
        perm_random = np.random.permutation(N)
        if s == '0'*n:
            for i in range(N):
                f_temp = np.copy(perm_random)
        else:
            chosen_perm = perm_random[0:int(N/2)]
            ind_1 = 0
            for i in range(N):
                if f_temp[i] == -1:
                    f_temp[i] = chosen_perm[ind_1]
                    f_temp[int(binary_add(give_binary_string(i,n) , s), 2)] = chosen_perm[ind_1]
                    ind_1 += 1
        f = [give_binary_string(z.item(),n) for z in f_temp]
        return f

def create_Uf_matrix(f):
        """
        Given a function f:{0,1}^n ---> {0,1}^n, creates and returns the corresponding oracle (unitary matrix) Uf
        Args:
            f: list of length n. f[i] is a string of length n.
        Returns:
            Uf: ndarray of size [2**(2n), 2**(2n)], representing a unitary matrix.
        """
        if type(f) != list:
            raise ValueError('Input the function in the form of a list')
        if any((x != '0' and x != '1') for x in "".join([y for y in f])):
            raise ValueError('The input function should only contain zeros and ones.')
        len_fs = [len(x) for x in f]
        if np.sum(np.diff(len_fs)) != 0:
            raise ValueError('Each element of f must be a string of the same length as the others.')
        n = len_fs[0]
        N = 2 ** (2*n)
        Uf = np.zeros([N,N], dtype = complex )
        
        for z in range(2**n):
            f_z = f[z]
            z_bin = give_binary_string(z,n)
            
            all_binary_strings = [give_binary_string(c,n) for c in range(2**n)]
            for j in all_binary_strings:
                inp = z_bin + j
                out = z_bin + binary_add(f_z,j)
                Uf[int(out[::-1],2),int(inp[::-1],2)] = 1
        return Uf

def get_Simon_circuit_from_matrix(Uf):
        """
        Simon's algorithm: Determines the value of s, for a function f = a . x + b
        Args:
            Uf: numpy array. Must be of shape [N,N], with N being an even power of 2.
        Returns: 
            circ: Qiskit QuantumCircuit object (not transpiled).
        """
        if not isinstance(Uf, np.ndarray):
            raise ValueError("Uf must be a numpy array.")
        if Uf.ndim != 2:
            raise ValueError("Uf should be a two dimensional array.")
        if Uf.shape[0] != Uf.shape[1]:
            raise ValueError("Uf must be a square array.")
        if np.log2(Uf.shape[0])%2 != 0:
            raise ValueError("number of rows and columns of Uf must be an even power of 2")
        n = int(np.log2(Uf.shape[0]) / 2)
        ## Define the circuit
        circ = QuantumCircuit(2*n, 2*n) 
        
        for i in range(n):
            circ.h(i)
        
        circ.iso(Uf, list(range(2*n)), [])
            
        for i in range(n):
            circ.h(i)
            
        circ.measure(list(range(0,2*n)), list(range(2*n)))  # have not reversed the qubits! Dont forget to reverse later
        
        
        return circ
def get_transpiled_circ(circ, backend):
    transpiled_circ = transpile(circ, backend = backend, optimization_level = 3)
    return transpiled_circ
    
         
def run_created_circuit(circ, backend, num_shots = 1024):
    """
    Run the circuit on the backend
    Returns:
        job
        result
    """
    job = execute(circ, backend = backend, shots = num_shots)
        
    return job, job.result()

    

def retrieve_job_from_id(job_id, backend):
    retrieved_job = backend.retrieve_job(job_id)
    return retrieved_job

def give_y_vals(counts):
    
    """
    Takes in the raw measurement outcomes from the circuit, and returns y values
    Args:
        counts: Dictionary
    Returns:
        new_counts: dictionary with strings reversed
        dict_y: dictionary with y values and counts
    """
    n = int(len(list(counts)[0])/2)
    
    n_list = [give_binary_string(z,n) for z in range(2**n)]
    new_counts ={}
    cropped_counts = {}
    for x in n_list:
        cropped_counts[x] = 0
    for i in counts:
        new_counts[i[::-1]] = counts[i]
        cropped_i = i[::-1][0:n]
        cropped_counts[cropped_i] += counts[i]
    
    dict_y = {}
    for key, value in cropped_counts.items():
        if value != 0:
            dict_y[key] = value
        
    return new_counts, dict_y

def random_Uf_multiple_circuits_matrix_method(n, num_circs):
    """
    For a given value of n, create num_circs number of circuits with randomly created Ufs
    """
    circ_list = []
    s_list = []
    f_list = []
    for i in range(num_circs):
        s = generate_random_s(n, seed_val = 0)
        f = s_function(s, seed_val = 0)
        s_list.append(s)
        f_list.append(f)
        Uf = create_Uf_matrix(f)
        circ_list.append(get_Simon_circuit_from_matrix(Uf))
    return s_list, f_list, circ_list


def get_meas_filter(num_qubits, backend, num_shots = 1024):
    meas_calibs, state_labels = complete_meas_cal(qr = num_qubits, 
                                              circlabel='measureErrorMitigation')
    job_calib = qiskit.execute(meas_calibs, backend = backend, shots=1024, optimization_level = 0)
    job_monitor(job_calib)
    calib_job_id = job_calib.job_id()
    print(calib_job_id)
    
    cal_results = job_calib.result()  
    meas_fitter = CompleteMeasFitter(cal_results, state_labels)
    meas_filter = meas_fitter.filter
    return meas_filter, calib_job_id

def s_solution_unmodified(list_y):
        """ The classical post-processing
        Args:
            list_y: List of strings of length n
        Returns:
            list_s: List of solutions to y.s=0
        """
        
        n = len(list_y[0])
        list_s = []
        
        for z in range(2**n):
            potential_s = give_binary_string(z,n)
            if np.sum([dot_product(potential_s, y) for y in list_y]) == 0:
                list_s.append(potential_s)
                
        return list_s

def qc_vs_sim_comparison_execute_matrix_method(n = 2, num_circs = 10, num_shots = 20, 
                                               device_preference = 'ibmq_burlington',
                                               save_fig_setting = 0):
    """ For a given n, obtain num_circs number of random Ufs (by creating Uf matrices), and obtain 
    num_shots number of y values, from the simulator, from the hardware backend, as well as error mitigated counts
    Write the three job ids in a notepad file.
    Create and save figures of histograms of y
    Because of significant errors in the quantum hardware, s is not calculated at all.
    Applicability of this function is limited to n<=2, since the circuit depth is very high, making it impossible to run
    for larger n
    Args:
        n: Int- value of n for f
        num_circs: Int - Number of random Ufs to be created
        num_shots: Int - Number of runs per Uf
        device_preference: If integer 0, obtain the least busy backend
                            If string, obtain corresponding backend
        save_fig_setting: Int- If 0, do not save figures
                                If 1, save figures
    Returns:
        list_y_dicts_sim: List of length num_circs, with each element containing y dictionary for simulation results
        list_y_dicts_qc: List of length num_circs, with each element containing y dictionary for qc results
        list_y_dicts_mit: List of length num_circs, with each element containing y dictionary for error mitigated results
        s_list: list of s strings
        f_list: list of functions f (list of lists)
        qc_id, sim_id, calib_job_id: Job ids for future reference
    """
    n = 1
    num_circs = 10
    
    n_qubits = 2 * n
    s_list, f_list, circ_list = random_Uf_multiple_circuits_matrix_method(n, num_circs)
    
    # load accounts and get backends
    load_account()
    device_backend, simulator_backend = get_backends(device_preference = 'ibmq_burlington', n_qubits = n_qubits)
    print("Chosen device backedn is " + device_backend.name())
    
    # run on device hardware
    job_qc = execute(circ_list, backend = device_backend, shots = num_shots, optimization_level = 3)
    job_monitor(job_qc)
    qc_id = job_qc.job_id()
    print('Job id for device job' + qc_id )
    result_qc = job_qc.result()
    
    # run on IBM simulator
    job_sim = execute(circ_list, backend = simulator_backend, shots = num_shots, optimization_level = 3)
    job_monitor(job_sim)
    sim_id = job_sim.job_id()
    print('Job id for simulation job' + sim_id)
    result_sim = job_sim.result()
    
    # error mitigation
    meas_filter, calib_job_id = get_meas_filter(num_qubits = n_qubits, backend = device_backend, num_shots = 1024)
    print('Job id for mitigation circuits' + calib_job_id)
    mitigated_results = meas_filter.apply(result_qc)
    mitigated_counts = [mitigated_results.get_counts(i) for i in range(num_circs)]
    
    
    # print details to notepad file
    file2write=open("job_tracking.txt",'a')
    file2write.write("Current time is " + str(datetime.now()) + " \n")
    file2write.write("Multiple circuits run for n=%i, backend = %s, job ID = %s" 
                     %(n,device_backend, qc_id))
    file2write.write("\n Multiple circuits run for n=%i, backend = %s, job ID = %s" 
                     %(n,simulator_backend, sim_id))
    file2write.write("\n Error mitigation circuits run for n=%i, backend = %s, job ID = %s \n \n"
                     %(n,device_backend, calib_job_id))
    file2write.close()
    
    # Obtain y_lists in the form of dictionaries for the simulation job, the qc backend job, and mitigated counts
    raw_counts_qc = result_qc.get_counts()
    raw_counts_sim = result_sim.get_counts()
    raw_counts_mitigated = mitigated_counts
    list_y_dicts_sim = []
    list_y_dicts_qc = []
    list_y_dicts_mit = []
    
    for i in range(num_circs):
        _, cur_y_list_sim = give_y_vals(raw_counts_sim[i])
        _, cur_y_list_qc = give_y_vals(raw_counts_qc[i])
        _, cur_y_list_mit = give_y_vals(raw_counts_mitigated[i])
        list_y_dicts_sim.append(cur_y_list_sim)
        list_y_dicts_qc.append(cur_y_list_qc)
        list_y_dicts_mit.append(cur_y_list_mit)
    
    if save_fig_setting:
        # Plot and save all values
        legend = [device_backend.name(), simulator_backend.name(), 'After mitigation'] 
        figures_folder = 'Figures'
        if not os.path.isdir(figures_folder):
            os.mkdir(figures_folder)    
        
        
        for expt_num in range(num_circs):
            fig = plt.figure(figsize=(16,10))
            ax = fig.gca()
            plot_histogram([list_y_dicts_qc[expt_num],list_y_dicts_sim[expt_num],list_y_dicts_mit[expt_num]],
                           legend=legend, ax = ax)
        #    plt.tight_layout()
            plt.title('Comparison of outputs: n=%i, circuit_number = %i. Exact s = %s' %(n,expt_num,s_list[expt_num]),
                      fontsize=20)
            
            fig.savefig(figures_folder + '/n=%i_circ_num=%i_backend=%s.png' 
                        %(n, expt_num, device_backend.name()), bbox_inches='tight')
            plt.close(fig)
        
    return list_y_dicts_sim, list_y_dicts_qc, list_y_dicts_mit, s_list, f_list, qc_id, sim_id, calib_job_id
    
def create_compact_circuit(s, device_backend):
    """ Given a string s of length n, obtain a random function, and the corresponding Simon's circuit
    Importantly, Uf is constructed directly as a combination of X,NOT, XNOT and SWAP gates, and then transpiled
    separately. Thus, we do not compute the UF matrix, which reduces the circuit depth significantly,
    imporving the results of executing the correponding quanutum circuit drastically.
    Args: 
        s: string of length n. Only contains 0's and 1's
        device_backend: Device backend
    Returns:
        circ: A random circuit corresponding to provided s
        transpiled_circ: Transpiled circuit, corresponding to the given device_backend
    
    """
    ## Define the circuit
    circ = QuantumCircuit(2 * n, 2 * n) 
    barriers = True
    ###################### Operate H gates on the first n qubits
    for i in range(n):
        circ.h(i)
    if barriers:
        circ.barrier()
    ##################### Create Uf 
    #First, transform |x>|0> to |x>|x>
    for i in range(n):
        circ.cx(i, n + i)
    # Obtain the first index of s which is non-zero
    j = s.find('1')
    if j != -1:
        # xj now controls whether s is to be XORed with the second register, or not
        # if xj = 0, do XOR
        # else, do nothing
        circ. x(j)
        for i in range(n):
            if s[i] == '1':
                circ.cx(j,n+i)
        circ. x(j)
    
    # Now, randomly swap qubits in the second register
    # Swap a random number of times
    num_swaps = np.random.randint(0,min(3,n))
    dummy = 0
    while dummy < num_swaps:
        pair = np.random.choice(list(range(n,2*n)), size = 2, replace = False)
        circ.swap(*pair)
        dummy += 1
    # randomly invert qubits
    num_flips = np.random.randint(0,min(3,n))
    flip_indices = np.random.choice(list(range(n,2*n)), size = num_flips, replace = False)
    for ind in flip_indices:
        circ.x(ind)
    
    
    if barriers:
        circ.barrier()
        
    ##################### Operate H gates on the first n qubits 
    for i in range(n):
        circ.h(i)
    if barriers:
        circ.barrier()

#    circ.measure(list(range(0,n))[::-1], list(range(n)))      
    circ.measure(list(range(0, 2 * n)), list(range(2 * n)))    
    transpiled_circ = get_transpiled_circ(circ, backend = device_backend)
    
    
    return circ, transpiled_circ



def frac_prod_zero(y_dict, s):
    """
    For any given bitstring s, and a dictionary of y values, calculate the fraction of times y.s = 0
    Args:
        s: strings of only zeros and ones
        y_dict: Dictionary of y values
                The keys are bitstrings corresponding to y values
                The values are the number of times the corresponding y occured
    Returns:
        frac_perp: float- fraction of the times y.s=0 was satisfied
    """
    tot_num = sum(y_dict.values())
    times_perp = 0
    for cur_y, freq in y_dict.items():
        is_perp = dot_product(cur_y,s)
        if is_perp:
            times_perp += freq 
    frac_perp = times_perp / tot_num
    
    return frac_perp
            

def s_solution(list_y):
    """
    Args:
        list_y: List of strings of length n
    Returns:
        list_s: List of solutions to y.s=0
    """
    
    n = len(list_y[0])
    list_s = []
    
    for z in range(2**n):
        potential_s = give_binary_string(z,n)
        if np.sum([dot_product(potential_s, y) for y in list_y]) == 0:
            list_s.append(potential_s)
            
    return list_s
    
    
#%%  Using the Uf matrix method 
    


n = 1
num_circs = 10
num_shots = 20
device_preference = 'ibmq_burlington'
save_fig_setting = 0
list_y_dicts_sim, list_y_dicts_qc, list_y_dicts_mit, s_list, f_list, qc_id, sim_id, calib_job_id = qc_vs_sim_comparison_execute_matrix_method(n = n,
                                                                                                                                              num_circs = num_circs, num_shots = num_shots,
                                                                                                                                              device_preference = device_preference,
                                                                                                                                              save_fig_setting = save_fig_setting)





#%% Method of creating Uf gate directly, without creating the Uf matrix

load_account()
device_preference = 'ibmq_16_melbourne'#'ibmq_burlington'
device_backend, simulator_backend = get_backends(device_preference = device_preference, n_qubits = 2 * n)


#%%  Create and save circuits, made by explicit construction of Uf, without making the Uf matrix
n = 7
s = generate_random_s(n)
circ, transpiled_circ = create_compact_circuit(s, device_backend)


figures_folder = 'Figures/Circuits/n=%i'%n
if not os.path.isdir(figures_folder):
    os.mkdir(figures_folder) 
cur_time = str(datetime.now())
cur_time = cur_time.replace(":","-")

fig = plt.figure(figsize=(16,10))
ax = fig.gca()
circ.draw(output  = 'mpl', ax = ax, vertical_compression  = 'high', idle_wires = False)
plt.title('Randomly created Simon\'s circuit for s = %s. Depth = %i' %(s,circ.depth()),fontsize=20)
fig.savefig(figures_folder + '/backend=%s_time=%s_untranspiled.png' 
            %(device_backend.name(), cur_time), bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(16,10))
ax = fig.gca()
transpiled_circ.draw(output  = 'mpl', ax = ax, vertical_compression  = 'high', idle_wires = False)
plt.title('Randomly created (transpiled) Simon\'s circuit for s = %s. Depth = %i' %(s,circ.depth()),fontsize=20)
fig.savefig(figures_folder + '/backend=%s_time=%s_transpiled.png' 
            %(device_backend.name(), cur_time ), bbox_inches='tight')
plt.close(fig)

print("Initial circuit depth = %i, transpiled circuit depth = %i"%(circ.depth(),transpiled_circ.depth()))



#%% Obtain average circuit depths for transpiled circuits, using the usual Uf matrix method

#Obtain transpiled depths of circuits
list_of_depths = []
num_circs = 10
n = 3
device_preference = 'ibmq_16_melbourne'#'ibmq_burlington'
load_account()
device_backend, _ = get_backends(device_preference = device_preference, n_qubits = 2 * n)
for i in range(num_circs):
    s = generate_random_s(n, seed_val = 0)
    f = s_function(s, seed_val = 0)
    Uf = create_Uf_matrix(f)
    circ = get_Simon_circuit_from_matrix(Uf)
    trans_circ = get_transpiled_circ(circ, backend = device_backend)
    list_of_depths.append(trans_circ.depth())
    
# Obtained circuit depths: 4 and 109, 3092.5 for n=1 and n=2 n=3 respectively.
    
    
    
    
#%% Prompt 1: Run a circuit multiple times using the compact circuit method

n = 1
num_shots = 20
n_qubits = 2 * n
num_circs = 50

load_account()
device_preference = ['ibmq_16_melbourne', 'ibmq_burlington']
device_backend, simulator_backend = get_backends(device_preference = device_preference[1], n_qubits = n_qubits)


s = '1'
circ, transpiled_circ = create_compact_circuit(s, device_backend)
circ_list = [circ for i in range(num_circs)]


figures_folder = 'Figures/Prompt_1/'
if not os.path.isdir(figures_folder):
    os.makedirs(figures_folder) 
######## Plot and save circuits    ################################
fig = plt.figure(figsize=(16,10))
ax = fig.gca()
circ.draw(output  = 'mpl', ax = ax, vertical_compression  = 'high', idle_wires = False)
plt.title('Circuit for s = %s' %(s),fontsize=20)
fig.savefig(figures_folder + '/s=%s_backend=%s.png' 
            %(s,device_backend.name() ), bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(16,10))
ax = fig.gca()
transpiled_circ.draw(output  = 'mpl', ax = ax, vertical_compression  = 'high', idle_wires = False)
plt.title('Transpiled circuit for s = %s' %(s),fontsize=20)
fig.savefig(figures_folder + '/s=%s_backend=%s_transpiled.png' 
            %(s,device_backend.name() ), bbox_inches='tight')
plt.close(fig)
################################################################



job_qc = execute(circ_list, backend = device_backend, shots = num_shots)
job_monitor(job_qc) 
qc_id = job_qc.job_id() 
job_sim = execute(circ_list, backend = simulator_backend, shots = num_shots, optimization_level = 3)
job_monitor(job_sim) 
sim_id = job_sim.job_id() 

meas_filter, calib_job_id = get_meas_filter(num_qubits = n_qubits, backend = device_backend, num_shots = 1024)
print('Job id for mitigation circuits' + calib_job_id)



file2write=open("job_tracking.txt",'a')
file2write.write("Current time is " + str(datetime.now()) + " \n")
file2write.write("Statistics for same circuit run multiple times: \n Multiple circuits run for n=%i, backend = %s, job ID = %s" 
                 %(n, device_backend.name(), qc_id))
file2write.write("\n Multiple circuits run for n=%i, backend = %s, job ID = %s" 
                 %(n, simulator_backend.name(), sim_id))
file2write.write("\n Error mitigation circuits run for n=%i, backend = %s, job ID = %s \n \n"
                 %(n,device_backend, calib_job_id))
file2write.close()


result_sim = job_sim.result()
result_qc = job_qc.result()
mitigated_results = meas_filter.apply(result_qc)
# Obtain y_lists in the form of dictionaries for the simulation job, the qc backend job, and mitigated counts
raw_counts_qc = [result_qc.get_counts(i) for i in range(num_circs)]
raw_counts_sim = [result_sim.get_counts(i) for i in range(num_circs)]
raw_counts_mitigated = [mitigated_results.get_counts(i) for i in range(num_circs)]

list_y_dicts_qc = [give_y_vals(raw_counts_qc[i])[1] for i in range(num_circs)]
list_y_dicts_sim = [give_y_vals(raw_counts_sim[i])[1] for i in range(num_circs)]
list_y_dicts_mit = [give_y_vals(raw_counts_mitigated[i])[1] for i in range(num_circs)]


######## Plot and save histogram for experiment 1
fig = plt.figure(figsize=(16,10))
ax = fig.gca()
legend = ['QC noisy', 'QC mitigated', 'simulator']
plot_histogram([list_y_dicts_qc[0], list_y_dicts_mit[0], list_y_dicts_sim[0]],
               legend=legend, ax = ax)
plt.title('Comparison of outputs: n=%i Exact s = %s' %(n, s),
          fontsize=20)

fig.savefig(figures_folder + '/Representative_hisrogram_s=%s_backend=%s.png' 
            %(s, device_backend.name()), bbox_inches='tight')
plt.close(fig)
########

list_counts_0_qc = np.array([list_y_dicts_qc[i]['0'] for i in range(num_circs)])
list_counts_0_mit = np.array([list_y_dicts_mit[i]['0'] for i in range(num_circs)])
list_counts_0_sim = np.array([list_y_dicts_sim[i]['0'] for i in range(num_circs)])
list_counts_1_qc = 20 - list_counts_0_qc
list_counts_1_mit = 20 - list_counts_0_mit
list_counts_1_sim = 20 - list_counts_0_sim
avg_0_qc = np.average(list_counts_0_qc)
avg_0_mit = np.average(list_counts_0_mit)

######## Plot and save a figure with counts across different experiments
fig = plt.figure(figsize=(16,10))
ax = fig.gca()
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
plt.title('Comparison of outputs for s = %s backend = %s'%(s, device_backend.name()),
          fontsize=20)
plt.xlabel('Experiment number',fontsize=20)
plt.ylabel('Counts',fontsize=20)

plt.plot(list_counts_0_qc, '--.', label='Noisy QC: counts for 0', markersize = 15)
plt.plot(list_counts_0_sim, '--.', label='Simulator: counts for 0', markersize = 15)
plt.plot(list_counts_0_mit, '--.', label='Mitigated QC: counts for 0', markersize = 15)
plt.plot([0,num_circs], [avg_0_mit, avg_0_mit], '-', label='Mitigated QC: average', markersize = 15)
plt.plot([0,num_circs], [avg_0_qc, avg_0_qc], '-', label='Noisy QC: average', markersize = 15)

plt.legend(fontsize = 20)
plt.ylim([0-1,num_shots+1])
plt.yticks(ticks = np.arange(0,num_shots,2))
plt.xticks(ticks = np.arange(0,num_circs,4))



fig.savefig(figures_folder +
            '/comparison_counts_n=%i_backend=%s.png'%(n, device_backend.name()),
            bbox_inches='tight')
plt.close(fig)

### Save data
results_folder = 'Results/Prompt_1/'
if not os.path.isdir(results_folder):
    os.makedirs(results_folder) 

np.savez(results_folder+'n=%i_backend=%s_s=%s_num_circs=%i'%(n,device_backend.name(),s,num_circs),
         n = n, num_shots = num_shots, num_circs = num_circs, device_backend = device_backend,
         simulator_backend = simulator_backend, circ = circ, transpiled_circ = transpiled_circ,
         qc_id = qc_id , sim_id = sim_id, calib_job_id = calib_job_id,
         list_y_dicts_qc = list_y_dicts_qc,
         list_y_dicts_sim = list_y_dicts_sim,
         list_y_dicts_mit = list_y_dicts_mit,
         avg_0_qc = avg_0_qc,
         avg_0_mit = avg_0_mit) 

#%% Prompt 1: Run a circuit multiple times using the compact circuit method
##### For n = 2

n = 2
num_shots = 20
n_qubits = 2 * n
num_circs = 50

load_account()
device_preference = ['ibmq_16_melbourne', 'ibmq_burlington']
device_backend, simulator_backend = get_backends(device_preference = device_preference[1], n_qubits = n_qubits)


s = '10'# generate_random_s(n)
circ, transpiled_circ = create_compact_circuit(s, device_backend)
circ_list = [circ for i in range(num_circs)]


figures_folder = 'Figures/Prompt_1/'
if not os.path.isdir(figures_folder):
    os.makedirs(figures_folder) 
######## Plot and save circuits    ################################
fig = plt.figure(figsize=(16,10))
ax = fig.gca()
circ.draw(output  = 'mpl', ax = ax, vertical_compression  = 'high', idle_wires = False)
plt.title('Circuit for s = %s' %(s),fontsize=20)
fig.savefig(figures_folder + '/s=%s_backend=%s.png' 
            %(s,device_backend.name() ), bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(16,10))
ax = fig.gca()
transpiled_circ.draw(output  = 'mpl', ax = ax, vertical_compression  = 'high', idle_wires = False)
plt.title('Transpiled circuit for s = %s' %(s),fontsize=20)
fig.savefig(figures_folder + '/s=%s_backend=%s_transpiled.png' 
            %(s,device_backend.name() ), bbox_inches='tight')
plt.close(fig)
################################################################



job_qc = execute(circ_list, backend = device_backend, shots = num_shots)
job_monitor(job_qc) 
qc_id = job_qc.job_id() 
print(qc_id)
job_sim = execute(circ_list, backend = simulator_backend, shots = num_shots, optimization_level = 3)
job_monitor(job_sim) 
sim_id = job_sim.job_id() 
print(sim_id)

meas_filter, calib_job_id = get_meas_filter(num_qubits = n_qubits, backend = device_backend, num_shots = 1024)
print('Job id for mitigation circuits' + calib_job_id)



file2write=open("job_tracking.txt",'a')
file2write.write("Current time is " + str(datetime.now()) + " \n")
file2write.write("Statistics for same circuit run multiple times: \n Multiple circuits run for n=%i, backend = %s, job ID = %s" 
                 %(n, device_backend.name(), qc_id))
file2write.write("\n Multiple circuits run for n=%i, backend = %s, job ID = %s" 
                 %(n, simulator_backend.name(), sim_id))
file2write.write("\n Error mitigation circuits run for n=%i, backend = %s, job ID = %s \n \n"
                 %(n,device_backend, calib_job_id))
file2write.close()


result_sim = job_sim.result()
result_qc = job_qc.result()
mitigated_results = meas_filter.apply(result_qc)
# Obtain y_lists in the form of dictionaries for the simulation job, the qc backend job, and mitigated counts
raw_counts_qc = [result_qc.get_counts(i) for i in range(num_circs)]
raw_counts_sim = [result_sim.get_counts(i) for i in range(num_circs)]
raw_counts_mitigated = [mitigated_results.get_counts(i) for i in range(num_circs)]

list_y_dicts_qc = [give_y_vals(raw_counts_qc[i])[1] for i in range(num_circs)]
list_y_dicts_sim = [give_y_vals(raw_counts_sim[i])[1] for i in range(num_circs)]
list_y_dicts_mit = [give_y_vals(raw_counts_mitigated[i])[1] for i in range(num_circs)]


######## Plot and save histogram for experiment 1
fig = plt.figure(figsize=(16,10))
ax = fig.gca()
legend = ['QC noisy', 'QC mitigated', 'simulator']
plot_histogram([list_y_dicts_qc[2], list_y_dicts_mit[2], list_y_dicts_sim[2]],
               legend=legend, ax = ax)
plt.title('Comparison of outputs: n=%i Exact s = %s' %(n, s),
          fontsize=20)

fig.savefig(figures_folder + '/Representative_hisrogram_s=%s_backend=%s.png' 
            %(s, device_backend.name()), bbox_inches='tight')
plt.close(fig)
########

list_all_n_bin_strings = [give_binary_string(z,n) for z in range(2**n)]
arr_counts = np.zeros([2**n, num_circs, 3])  # y index, circuit index, which type of result (backend)
for y_ind, possible_y in enumerate(list_all_n_bin_strings):
    for cur_circ_num in range(num_circs):
        if possible_y in list_y_dicts_qc[cur_circ_num]:
            arr_counts[y_ind, cur_circ_num, 0] = list_y_dicts_qc[cur_circ_num][possible_y]
        if possible_y in list_y_dicts_mit[cur_circ_num]:
            arr_counts[y_ind, cur_circ_num, 1] = list_y_dicts_mit[cur_circ_num][possible_y]
        if possible_y in list_y_dicts_sim[cur_circ_num]:
            arr_counts[y_ind, cur_circ_num, 2] = list_y_dicts_sim[cur_circ_num][possible_y]

avg_counts_arr = np.average(arr_counts, axis = 1)
######## Plot and save a figure with counts across different experiments

for y_ind, possible_y in enumerate(list_all_n_bin_strings):
    
    fig = plt.figure(figsize=(16,10))
    ax = fig.gca()
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    plt.title('Comparison of outputs for s = %s backend = %s'%(s, device_backend.name()),
              fontsize=20)
    plt.xlabel('Experiment number',fontsize=20)
    plt.ylabel('Counts',fontsize=20)
    plt.plot(arr_counts[y_ind,:,0], '--.', label='Noisy QC: counts for %s'%possible_y, markersize = 15)
    plt.plot(arr_counts[y_ind,:,1], '--.', label='Simulator: counts for %s'%possible_y, markersize = 15)
    plt.plot(arr_counts[y_ind,:,2], '--.', label='Mitigated QC: counts for %s'%possible_y, markersize = 15)
    plt.plot([0,num_circs], [avg_counts_arr[y_ind,1], avg_counts_arr[y_ind,1]], '-', label='Mitigated QC: average', markersize = 15)
    plt.plot([0,num_circs], [avg_counts_arr[y_ind,0], avg_counts_arr[y_ind,0]], '-', label='Noisy QC: average', markersize = 15)
    plt.plot([0,num_circs], [avg_counts_arr[y_ind,2], avg_counts_arr[y_ind,2]], '-', label='Simulator: average', markersize = 15)
    
    plt.legend(fontsize = 20)
    plt.ylim([0-1,num_shots+1])
    plt.yticks(ticks = np.arange(0,num_shots,2))
    plt.xticks(ticks = np.arange(0,num_circs,4))
    
    
    
    fig.savefig(figures_folder +
                '/comparison_counts_n=%i_backend=%s_y=%s.png'%(n, device_backend.name(),possible_y),
                bbox_inches='tight')
    plt.close(fig)

### Save data
results_folder = 'Results/Prompt_1/'
if not os.path.isdir(results_folder):
    os.makedirs(results_folder) 

np.savez(results_folder+'n=%i_backend=%s_s=%s_num_circs=%i'%(n,device_backend.name(),s,num_circs),
         n = n, num_shots = num_shots, num_circs = num_circs, device_backend = device_backend,
         simulator_backend = simulator_backend, circ = circ, transpiled_circ = transpiled_circ,
         qc_id = qc_id , sim_id = sim_id, calib_job_id = calib_job_id,
         list_y_dicts_qc = list_y_dicts_qc,
         list_y_dicts_sim = list_y_dicts_sim,
         list_y_dicts_mit = list_y_dicts_mit,
         arr_counts = arr_counts, avg_counts_arr = avg_counts_arr) 

#%%  Prompt 2
## Create a number of circuits corresponding to different values of s, chosen randomly. 
## Obtain a histogram of execution times

n = 7
num_shots = 20
n_qubits = 2 * n
num_circs = 20

load_account()
device_preference = ['ibmq_16_melbourne', 'ibmq_burlington']
device_backend, simulator_backend = get_backends(device_preference = device_preference[0], n_qubits = n_qubits)



s_list = []
circ_list = []
transpiled_circ_list = []
transpilation_time_list = []
for ind in range(num_circs):
    s = generate_random_s(n)
    s_list.append(s)
    start = time.time()
    circ, transpiled_circ = create_compact_circuit(s, device_backend)
    end = time.time()
    transpilation_time_list.append(end-start)
    
    circ_list.append(circ)
    transpiled_circ_list.append(transpiled_circ)
    

jobs_id_list = []
for i in range(num_circs):
    cur_trans_circ = transpiled_circ_list[i]
    job_qc = execute(cur_trans_circ, backend = device_backend, shots = num_shots)
    job_monitor(job_qc) 
    qc_id = job_qc.job_id() 
    print("Job number %i, corresponding to ID=%s is done!"%(i,qc_id))
    file2write=open("job_tracking.txt",'a')
    file2write.write("Current time is " + str(datetime.now()) + " \n")
    file2write.write("\nTrying Uf exec time for n=%i, backend = %s, job ID = %s, s = %s" 
                     %(n, device_backend.name(), qc_id, s_list[i]))
    file2write.close()
    jobs_id_list.append(qc_id)
#    time.sleep(60*10)
    

circ_run_time_arr = []
circ_depth_arr = []
y_counts_list_of_dicts = []
for i in range(num_circs):
    job_id = jobs_id_list[i]
    job = retrieve_job_from_id(job_id, backend = device_backend)
    result = job.result()
    circ_run_time_arr.append(result.time_taken)
    circ_depth_arr.append(transpiled_circ_list[i].depth())
    #### Also obtain the fraction of time y.s was 0
    raw_counts = result.get_counts()
    y_counts_list_of_dicts.append(give_y_vals(raw_counts)[1])

    
    
frac_perp_arr = [frac_prod_zero(y_counts_list_of_dicts[i], s_list[i]) for i in range(num_circs)]


# save results to to file
results_folder = 'Results/Prompt_2/'
if not os.path.isdir(results_folder):
    os.makedirs(results_folder) 

np.savez(results_folder+'n=%i_backend=%s_s=%s_num_circs=%i'%(n,device_backend.name(),s,num_circs),
         n = n, num_circs = num_circs, device_backend = device_backend, 
         simulator_backend = simulator_backend, s_list = s_list,
         circ_list = circ_list, transpiled_circ = transpiled_circ, jobs_id_list = jobs_id_list,
         transpilation_time_list = transpilation_time_list, circ_run_time_arr = circ_run_time_arr,
         circ_depth_arr = circ_depth_arr, frac_perp_arr = frac_perp_arr,
         y_counts_list_of_dicts = y_counts_list_of_dicts)


###### Save the two historgram- one for transpilation, and the other for run time, as well as circuit depth
figures_folder = 'Figures/Prompt_2/'
if not os.path.isdir(figures_folder):
    os.makedirs(figures_folder) 

plt.rcParams["font.family"] = "serif"


### Plot run time histogram
fig = plt.figure(figsize=(16,10))
plt.hist(circ_run_time_arr)
plt.title('Dependence of execution time on $U_f$ (Simon\'s algorithm)', fontsize=25)
plt.xlabel('Execution time (s)',fontsize=20)
plt.ylabel('Frequency of occurence',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

fig.savefig(figures_folder+'/n=%i_backend=%s_runtime.png', bbox_inches='tight')
plt.close(fig)      
            

### Plot transpilation time histogram
fig = plt.figure(figsize=(16,10))
plt.hist(transpilation_time_list)
plt.title('Dependence of transpilation time on $U_f$ (Simon\'s algorithm)', fontsize=25)
plt.xlabel('Transpilation time (s)',fontsize=20)
plt.ylabel('Frequency of occurence',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.savefig(figures_folder+'/n=%i_backend=%s_transpilation.png', bbox_inches='tight')
plt.close(fig)  


### Plot circuit depth, transpilation time, and execution time 

fig = plt.figure(figsize=(16,10))
plt.title(r'$U_f$ dependence', fontsize=25)
plt.xlabel('Circuit number',fontsize=20)

ax1 = fig.add_subplot(111)
ax1.plot(list(range(num_circs)), transpilation_time_list, ls = '--', marker = 's', color = 'r', label = 'Transpilation time')
ax1.plot(list(range(num_circs)), circ_run_time_arr, ls = '--', marker = 's', color = 'g', label = 'Circuit execution time')
ax1.set_ylabel('Time taken (s)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

ax2 = ax1.twinx()
ax2.plot(list(range(num_circs)), circ_depth_arr, ls = '-.', marker = 'o', color = 'b',  label = 'Circuit depth')
ax2.set_ylabel('Transpiled circuit depth',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(ticks = np.arange(0,num_circs,4))
fig.legend(bbox_to_anchor=(0.95, 0.95),fontsize = 15)

fig.savefig(figures_folder+'/n=%i_backend=%s_time_and_depth.png') #, bbox_inches='tight'
plt.close(fig)
    

### Plot circuit depth and frac prod zero 

fig = plt.figure(figsize=(16,10))
plt.title(r'Accuracy and circuit depth', fontsize=25)
plt.xlabel('Circuit number',fontsize=20)

ax1 = fig.add_subplot(111)
ax1.plot(list(range(num_circs)), frac_perp_arr, ls = '--', marker = 's', color = 'r', label = r'accuracy')
ax1.set_ylabel(r'Fraction of times output satisfied $y.s=0$',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

ax2 = ax1.twinx()
ax2.plot(list(range(num_circs)), circ_depth_arr, ls = '-.', marker = 'o', color = 'b',  label = 'Circuit depth')
ax2.set_ylabel('Transpiled circuit depth',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(ticks = np.arange(0,num_circs,4))
fig.legend(bbox_to_anchor=(0.95, 0.95),fontsize = 15)

fig.savefig(figures_folder+'/n=%i_backend=%s_accuracy_and_depth.png') #, bbox_inches='tight'
plt.close(fig)


#%% Prompt 3
### Scalability as n grows

load_account()
device_preference = ['ibmq_16_melbourne', 'ibmq_burlington']
n_qubits = 14
device_backend, simulator_backend = get_backends(device_preference = device_preference[0], n_qubits = n_qubits)

n_list = np.arange(1,8)
num_s = 10 # number of different s values, i.e. number of distinct circuits
num_shots =20
# create random s and circuits
list_of_lists_s = []
list_of_lists_circ = []
list_of_lists_trans_circ = []
for n in n_list:
    n = int(n)
    s_list = []
    circ_list = []
    transpiled_circ_list = []
    for s_ind in range(num_s):
        s = generate_random_s(n)
        s_list.append(s)
        circ, transpiled_circ = create_compact_circuit(s, device_backend)
        circ_list.append(circ)
        transpiled_circ_list.append(transpiled_circ)
    list_of_lists_s.append(s_list)
    list_of_lists_circ.append(circ_list)
    list_of_lists_trans_circ.append(transpiled_circ_list)


jobs_id_list = []

for n_ind, n in enumerate(n_list):

    cur_trans_circ_list = transpiled_circ_list[n_ind]
    
    job_qc = execute(cur_trans_circ_list, backend = device_backend, shots = num_shots)
    job_monitor(job_qc) 
    qc_id = job_qc.job_id() 

    print("For n =%i, job ID=%s is done!"%(n,qc_id))
    file2write=open("job_tracking.txt",'a')
    file2write.write("Current time is " + str(datetime.now()) + " \n")
    file2write.write("\nTrying scaling with n, exec time for n=%i, backend = %s, job ID = %s" 
                     %(n, device_backend.name(), qc_id))
    file2write.close()
    jobs_id_list.append(qc_id)


### Obtain average circuit length and average time vs n
avg_circ_depth = []
avg_trans_circ_depth = []
for n_ind, n in enumerate(n_list):
    transpiled_circ_list = list_of_lists_trans_circ[n_ind]
    circ_list = list_of_lists_circ[n_ind]
    circ_depth_list = [circ_list[i].depth() for i in range(num_s)]
    
#    transpiled_circ_list = get_transpiled_circ(circ_list, backend = device_backend)
    trans_circ_depth_list = [transpiled_circ_list[i].depth() for i in range(num_s)]
    
    avg_circ_depth.append(np.average(circ_depth_list))
    avg_trans_circ_depth.append(np.average(trans_circ_depth_list))

### Obtain average execution time vs n
avg_exec_time_list = []
for n_ind, n in enumerate(n_list):
    job_id = jobs_id_list[n_ind]
    job_n = retrieve_job_from_id(job_id, backend = device_backend)
    if str(job_n.status()) != 'JobStatus.ERROR':
        result_n = job_n.result()
        avg_exec_time_list.append(result_n.time_taken / num_s)



###### Save the plots
figures_folder = 'Figures/Prompt_3/'
if not os.path.isdir(figures_folder):
    os.makedirs(figures_folder) 

plt.rcParams["font.family"] = "serif"



### Scaling analysis plot figure save
fig = plt.figure(figsize=(16,10))
plt.title(r'Execution time scaling', fontsize=25)


plt.plot(n_list[1:], avg_exec_time_list, ls = '--', marker = 's', color = 'r', label = 'Execution time')
plt.xlabel(r'$n$',fontsize=20)
plt.ylabel('Execution time (s)',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim([0,1])
fig.savefig(figures_folder+'/Exec_time_scaling.png', bbox_inches='tight')
plt.close(fig)  

### Circuit depth vs n
fig = plt.figure(figsize=(16,10))
plt.title(r'Average circuit depth vs n', fontsize=25)


plt.plot(n_list, avg_trans_circ_depth, ls = '--', marker = 's', color = 'r', label = 'Transpiled circuit depth')
plt.plot(n_list, avg_circ_depth, ls = '--', marker = 's', color = 'b', label = 'Circuit depth')
plt.xlabel(r'$n$',fontsize=20)
plt.ylabel('Average circuit depth',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.savefig(figures_folder+'/circuit_depth_scaling.png', bbox_inches='tight')
plt.legend(fontsize = 25)
plt.close(fig)  


###### Save data
results_folder = 'Results/Prompt_3/'
if not os.path.isdir(results_folder):
    os.makedirs(results_folder) 

np.savez(results_folder+'Scaling_analysis',
         device_backend = device_backend,
         n_list = n_list, num_s = num_s, num_shots = num_shots,
         list_of_lists_s = list_of_lists_s, list_of_lists_circ = list_of_lists_circ,
         list_of_lists_trans_circ = list_of_lists_trans_circ,
         jobs_id_list = jobs_id_list, avg_circ_depth = avg_circ_depth,
         avg_trans_circ_depth = avg_trans_circ_depth,
         avg_exec_time_list = avg_exec_time_list)
