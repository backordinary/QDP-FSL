# https://github.com/askery/Ibm_openprize_2022_qtime/blob/f1185a9e0d5223ebbfee72c41ebfc8801e7dcea7/final_versions/troter_utils.py
import numpy as np
import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt
from IPython.display import Image, display

plt.rcParams.update({'font.size': 16})  # enlarge matplotlib fonts

import seaborn as sns

# Import qubit states Zero (|0>) and One (|1>), and Pauli operators (X, Y, Z)
from qiskit.opflow import Zero, One, I, X, Y, Z

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Importing standard Qiskit modules
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, execute, transpile, Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.monitor import job_monitor
from qiskit.circuit import Parameter

# Import state tomography modules
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import state_fidelity

import time

from qiskit.algorithms.optimizers import GradientDescent, SPSA, SciPyOptimizer

# reproducibility
import qiskit
qiskit.utils.algorithm_globals.random_seed = 42

import itertools

from os import listdir
from os.path import isfile, join

import pickle

#################################################################
# ============================================================= #
#################################################################

def show_figure(fig):
    '''
    auxiliar function to display plot 
    even if it's not the last command of the cell
    from: https://github.com/Qiskit/qiskit-terra/issues/1682
    '''
    
    new_fig = plt.figure()
    new_mngr = new_fig.canvas.manager
    new_mngr.canvas.figure = fig
    fig.set_canvas(new_mngr.canvas)
    plt.show(fig)
    
#################################################################
# ============================================================= #
#################################################################

def show_decompose(qc, n, show_original=False):
    '''
    auxiliar function to show different levels of circuit decomposition
    '''
    
    if n <= 0 or show_original:
        
        show_figure(qc.draw("mpl"))
        print("#"*80)
           
    for _ in range(n):

        qc = qc.decompose()

        show_figure(qc.draw("mpl"))
        print("#"*80)

#################################################################
# ============================================================= #
#################################################################

from qiskit.compiler import transpile

def transpile_jakarta(qc, sim_noisy_jakarta, show_fig=False):
    '''
    function to transpile given quantum circuit using the jakarta backend
    '''

    transp_circ = transpile(qc, sim_noisy_jakarta, optimization_level=0)

    if show_fig:
        
        show_figure(transp_circ.draw("mpl"))

    print(f'Optimization Level {0}')
    print(f'Depth: {transp_circ.depth()}')
    print(f'Gate counts: {transp_circ.count_ops()}')
    print(f'Total number of gates: {sum(transp_circ.count_ops().values())}')

    print()
    print("#"*80)
    print()

    transp_circ = transpile(qc, sim_noisy_jakarta, optimization_level=1)

    if show_fig:
        
        show_figure(transp_circ.draw("mpl"))

    print(f'Optimization Level {1}')
    print(f'Depth: {transp_circ.depth()}')
    print(f'Gate counts: {transp_circ.count_ops()}')
    print(f'Total number of gates: {sum(transp_circ.count_ops().values())}')

    print()
    print("#"*80)
    print()

    transp_circ = transpile(qc, sim_noisy_jakarta, optimization_level=2)

    if show_fig:
        
        show_figure(transp_circ.draw("mpl"))

    print(f'Optimization Level {2}')
    print(f'Depth: {transp_circ.depth()}')
    print(f'Gate counts: {transp_circ.count_ops()}')
    print(f'Total number of gates: {sum(transp_circ.count_ops().values())}')

    print()
    print("#"*80)
    print()

    transp_circ = transpile(qc, sim_noisy_jakarta, optimization_level=3)

    if show_fig:
        
        show_figure(transp_circ.draw("mpl"))

    print(f'Optimization Level {3}')
    print(f'Depth: {transp_circ.depth()}')
    print(f'Gate counts: {transp_circ.count_ops()}')
    print(f'Total number of gates: {sum(transp_circ.count_ops().values())}')

    print()
    print("#"*80)
    print()
#################################################################
# ============================================================= #
#################################################################

def XX(parameter):
    '''
    decomposition of XX gate
    - parameter: Parameter object
    '''

    # Build a subcircuit for XX(t) two-qubit gate
    XX_qr = QuantumRegister(2)
    XX_qc = QuantumCircuit(XX_qr, name='XX')

    XX_qc.ry(np.pi/2,[0,1])
    XX_qc.cnot(0,1)
    XX_qc.rz(2 * parameter, 1)
    XX_qc.cnot(0,1)
    XX_qc.ry(-np.pi/2,[0,1])

    # Convert custom quantum circuit into a gate
    XX = XX_qc.to_instruction()
    
    return XX

##############################################################
##############################################################

def YY(parameter):
    '''
    decomposition of YY gate
    - parameter: Parameter object
    '''
    
    # Build a subcircuit for YY(t) two-qubit gate
    YY_qr = QuantumRegister(2)
    YY_qc = QuantumCircuit(YY_qr, name='YY')

    YY_qc.rx(np.pi/2,[0,1])
    YY_qc.cnot(0,1)
    YY_qc.rz(2 * parameter, 1)
    YY_qc.cnot(0,1)
    YY_qc.rx(-np.pi/2,[0,1])

    # Convert custom quantum circuit into a gate
    YY = YY_qc.to_instruction()
    
    return YY

##############################################################
##############################################################

def ZZ(parameter):
    '''
    decomposition of ZZ gate
    - parameter: Parameter object
    '''

    # Build a subcircuit for ZZ(t) two-qubit gate
    ZZ_qr = QuantumRegister(2)
    ZZ_qc = QuantumCircuit(ZZ_qr, name='ZZ')

    ZZ_qc.cnot(0,1)
    ZZ_qc.rz(2 * parameter, 1)
    ZZ_qc.cnot(0,1)

    # Convert custom quantum circuit into a gate
    ZZ = ZZ_qc.to_instruction()
    
    return ZZ

#################################################################
# ============================================================= #
#################################################################

def trotter_first_order(parameter):
    '''
    single trotter step, first order
    - parameter: Parameter object
    '''
    
    num_qubits = 3

    Trot_qr = QuantumRegister(num_qubits)
    Trot_qc = QuantumCircuit(Trot_qr, name='Trot')

    for i in range(0, num_qubits - 1):
        
        Trot_qc.append(ZZ(parameter), [Trot_qr[i], Trot_qr[i+1]])
        Trot_qc.append(YY(parameter), [Trot_qr[i], Trot_qr[i+1]])
        Trot_qc.append(XX(parameter), [Trot_qr[i], Trot_qr[i+1]])

    Trot_gate = Trot_qc.to_instruction()
    
    return Trot_gate

#################################################################
# ============================================================= #
#################################################################

def trotter_second_order(parameter):
    '''
    single trotter step, second order
    - parameter: Parameter object
    '''
    
    num_qubits = 3

    Trot_qr = QuantumRegister(num_qubits)
    Trot_qc = QuantumCircuit(Trot_qr, name='Trot')
    
    # in the circuit picture, it only appears the label "t" in the gates (instead of "t/2"),
    # because I used a single parameter vector, which makes the code more concise.
    # But notice that here it's correctly done, t/2!!
    Trot_qc.append(ZZ(parameter/2), [Trot_qr[0], Trot_qr[1]])
    Trot_qc.append(YY(parameter/2), [Trot_qr[0], Trot_qr[1]])
    Trot_qc.append(XX(parameter/2), [Trot_qr[0], Trot_qr[1]])
    
    Trot_qc.append(ZZ(parameter), [Trot_qr[1], Trot_qr[2]])
    Trot_qc.append(YY(parameter), [Trot_qr[1], Trot_qr[2]])
    Trot_qc.append(XX(parameter), [Trot_qr[1], Trot_qr[2]])
    
    Trot_qc.append(ZZ(parameter/2), [Trot_qr[0], Trot_qr[1]])
    Trot_qc.append(YY(parameter/2), [Trot_qr[0], Trot_qr[1]])
    Trot_qc.append(XX(parameter/2), [Trot_qr[0], Trot_qr[1]])

    Trot_gate = Trot_qc.to_instruction()
    
    return Trot_gate

#################################################################
# ============================================================= #
#################################################################

def trotter_step(order, parameter):
    '''
    wrapper function for a single trotter step
    - order: int, desired order, must be 1 or 2
    - parameter: Parameter object
    '''
    
    if order == 1:
        
        Trot_gate = trotter_first_order(parameter)
        
    elif order == 2:
        
        Trot_gate = trotter_second_order(parameter)
        
    else:
        
        raise ValueError("Only 1st or 2nd orders allowed!")
        
    return Trot_gate

#################################################################
# ============================================================= #
#################################################################

def view_single_trotter_step(order, parameter):
    '''
    draws circuit for a single trotter step of specified order (1 or 2)
    - order: int, desired order, must be 1 or 2
    - parameter: Parameter object
    '''
    
    num_qubits = 3
    qc = QuantumCircuit(num_qubits)

    qc.append(trotter_step(order, parameter), range(num_qubits))

    print("Single trotterization step:")
    show_decompose(qc, 1)
     

#################################################################
# ============================================================= #
#################################################################

def full_trotter_circ(order, trotter_steps=4, target_time=np.pi,
                      uniform_times=True, steps_times=None):
    '''
    construct the full trotterization circuit
    
    args:
    - order: 1 or 2 for first or second order;
    - trotter_steps: number of steps, must be >=4;
    - target_time: final evolution must be t=pi, but added asa parameter, so we can simulate other times;
    - uniform: boolean indicating wheter or not uniform times will be used;
    - steps_times: list with times for each step, in order. length must be trotter_steps!
    
    returns quantum register and quantum circuit
    '''

    # Initialize quantum circuit for 7 qubits
    qr = QuantumRegister(7)
    qc = QuantumCircuit(qr)

    # Prepare initial state (remember we are only evolving 3 of the 7 qubits on
    # jakarta qubits (q_5, q_3, q_1) corresponding to the state |110>)
    # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)
    qc.x([3,5])  
    
    # in this case, all times parameter are the same, because all times will be
    # the same, equal to target_time/trotter_steps, as indicated below in bind_parameters
    if uniform_times:

        Trot_gate = trotter_step(order, Parameter('t'))

        for _ in range(trotter_steps):

            qc.append(Trot_gate, [qr[1], qr[3], qr[5]])
            
        # evaluate simulation at target_time (t=pi) meaning each trotter step evolves pi/trotter_steps in time
        qc = qc.bind_parameters({qc.parameters[0]: target_time/trotter_steps})
    
    # now, in this case, we'll have different times for each step
    # but such that they sum to target_time, of course.
    # these times are in the parameter "steps_times".
    # and, because they're different, we'll have different parameters as well!
    else:
        
        # check
        if len(steps_times) != trotter_steps:
            raise ValueError(f"Incorrect quantity of times {len(steps_times)}! Must be equal to number of steps {trotter_steps}")
                             
        for i in range(trotter_steps):
            
            Trot_gate = trotter_step(order, Parameter(f't{i}'))
                                     
            qc.append(Trot_gate, [qr[1], qr[3], qr[5]])
                                     
        params_dict = {param: time for param, time in zip(qc.parameters, steps_times)}
                                     
        qc = qc.bind_parameters(params_dict)
         

    return qr, qc

#################################################################
# ============================================================= #
#################################################################

def state_tomagraphy_circs(order, trotter_steps=4, target_time=np.pi,
                           uniform_times=True, steps_times=None):
    '''
    build and returns circuits for state tomography
    trotter_steps: number of steps, must be >=4
    order: 1 or 2 for first or second order
    '''
    
    qr, qc = full_trotter_circ(order, trotter_steps, target_time,
                               uniform_times, steps_times)

    # Generate state tomography circuits to evaluate fidelity of simulation
    st_qcs = state_tomography_circuits(qc, [qr[1], qr[3], qr[5]])

    return st_qcs

#################################################################
# ============================================================= #
#################################################################

def execute_st_simulator(st_qcs, backend, id_str):
    '''
    execute state tomography jobs
    backend: preferably sim_noisy_jakarta or sim_no_noise
    '''
    
    shots = 8192
    reps = 8

    jobs = []
    print()
    for i in range(reps):
        # execute
        job = execute(st_qcs, backend, shots=shots, qobj_id=f"{id_str}_run_{i+1}")
        print(f'{i+1}/{reps} - Job ID', job.job_id())
        jobs.append(job)
        
    return jobs


#################################################################
# ============================================================= #
#################################################################


def state_tomo(result, st_qcs):
    '''
    Computes the state tomography based on the st_qcs quantum circuits and the results from those ciricuits
    '''
    
    # The expected final state; necessary to determine state tomography fidelity
    target_state = (One^One^Zero).to_matrix()  # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)
    
    # Fit state tomography results
    tomo_fitter = StateTomographyFitter(result, st_qcs)
    
    rho_fit = tomo_fitter.fit(method='lstsq')
    
    # Compute fidelity
    fid = state_fidelity(rho_fit, target_state)
    
    return fid

#################################################################
# ============================================================= #
#################################################################

def final_fidelities(jobs, st_qcs, order, trotter_steps):
    '''
    return list of fidelities, for each job
    '''
    
    # Compute tomography fidelities for each repetition
    fids = []
    
    for job in jobs:
        
        fid = state_tomo(job.result(), st_qcs)
        
        fids.append(fid)

    print()
    print("#"*80)
    print()
    print(f"Final results - order: {order} - strotter steps: {trotter_steps}\n")
    print('State tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids)))
    
    return fids

#################################################################
# ============================================================= #
#################################################################

def simulate_full_circ(qc, backend):
    '''
    returns p(psi = 110) at the end of the evolution
    '''
    
    counts = execute(qc, backend, shots=1e5, seed_simulator=42).result().get_counts()

    return counts["110"]/sum(counts.values())

#################################################################
# ============================================================= #
#################################################################

def simulate_H_all_t(order, trotter_steps, backend,
                     uniform_times=True, steps_times=None):
    '''
    this function simulates the full trotter evolution (t from 0 to pi)
    it calculates the fidelity at every point in the interval
    (see meaning of args in the definition of the full_trotter_circ() function)
    '''
    
    print()
    print("#"*80)
    print()
    
    print("Starting simulation for times from 0 to pi!")
    start = time.time()

    ts = np.linspace(0, np.pi, 100)
    probs = []
        
    for target_time in ts:
        
        if uniform_times:
            # keep as is, in this case, None
            steps_times_current = None
        else:          
            # re-normalize times, so that they sum to target_time
            steps_times_current = np.array(steps_times)*(target_time/sum(steps_times))

        st_qcs = state_tomagraphy_circs(order, trotter_steps, target_time,
                                        uniform_times, steps_times_current)

        # last one in state tomography is always the one in which
        # only the assigned qubits are measured
        prob_110 = simulate_full_circ(st_qcs[-1], backend)

        probs.append(prob_110)

    print("Simulation ended!")
    stop = time.time()
    duration = time.strftime("%H:%M:%S", time.gmtime(stop-start))

    print(f"Total time of simulation: {duration}\n")
    
    # fidelity (prob) at t=pi
    fidelity_pi = np.array(probs)[np.where(ts == np.pi)].squeeze()
    
    return ts, probs, fidelity_pi

#################################################################
# ============================================================= #
#################################################################

def plot_simulation_H_all_t(ts, probs, fidelity_pi, order, trotter_steps, params_bounds_min, plot_theoretical=True):
    '''
    this plots the theoretical and simulated hamiltonian evolution,
    via the fidelity of the target state as a function of time, from 0 to pi
    '''
      
    fig = plt.figure(figsize=(10, 7))
    
    plt.plot(ts, probs, label="simulated")
    
    plt.xlabel('time')
    plt.ylabel(r'Prob. of state $|110\rangle$')
    plt.title(r'Evolution of $|110\rangle$ under $H_{Heis3}(t)$' + f' - order {order}, {trotter_steps} steps')
    plt.grid()

    plt.axhline(y=fidelity_pi, color="red", ls=":", label=f"F($\pi)={fidelity_pi}$")
    
    if plot_theoretical:
        
        # computed with opflow
        probs_theoretical = [1.0, 0.9959801051027279, 0.9840172692892207, 0.9643990582381565, 0.937594911988758, 0.9042417387878469,
                             0.865124429911968, 0.821151950110675, 0.7733298053832053, 0.7227298088581207, 0.670458152474241, 
                             0.6176228439841472, 0.5653015837895042, 0.5145111338492695, 0.46617917227787303, 0.42111953445810696, 
                             0.3800116179277513, 0.3433845784384221, 0.31160677383035945, 0.28488072684522886, 0.26324368434414297,
                             0.24657365551117913, 0.23460062242465882, 0.22692243956194447, 0.22302478059314604, 0.22230435674939336, 
                             0.22409452576938554, 0.22769233752892054, 0.23238602435861347, 0.23748194190916663, 0.2423300000849707, 
                             0.24634669159746717, 0.2490349254154318, 0.2500000000000001, 0.24896120190274262, 0.24575868345200233, 
                             0.24035545262059993, 0.2328344921370996, 0.22339120671060558, 0.21232157022640713, 0.20000650262308176, 
                             0.1868931431419818, 0.1734737977909714, 0.16026342019134074, 0.1477765335911936, 0.13650451605088834, 
                             0.126894150225658, 0.11932828465859018, 0.11410936717336749, 0.11144649610946522, 0.11144649610946515, 
                             0.1141093671733676, 0.11932828465859018, 0.12689415022565823, 0.13650451605088837, 0.14777653359119364, 
                             0.16026342019134082, 0.1734737977909711, 0.18689314314198185, 0.20000650262308153, 0.2123215702264073, 
                             0.22339120671060558, 0.23283449213709922, 0.24035545262059999, 0.24575868345200222, 0.2489612019027428,
                             0.24999999999999994, 0.24903492541543146, 0.24634669159746728, 0.2423300000849708, 0.23748194190916688, 
                             0.23238602435861336, 0.22769233752892049, 0.22409452576938554, 0.22230435674939347, 0.2230247805931461, 
                             0.2269224395619447, 0.2346006224246586, 0.24657365551117874, 0.26324368434414297, 0.2848807268452291, 
                             0.3116067738303597, 0.3433845784384221, 0.380011617927751, 0.42111953445810757, 0.46617917227787303, 
                             0.5145111338492697, 0.5653015837895042, 0.6176228439841478, 0.6704581524742411, 0.7227298088581207, 
                             0.7733298053832053, 0.8211519501106751, 0.8651244299119683, 0.9042417387878462, 0.9375949119887566, 
                             0.9643990582381559, 0.9840172692892216, 0.9959801051027278, 1.0000000000000004]
        
        plt.plot(ts, probs_theoretical, label="theoretical")

    plt.legend(prop={'size': 12}, loc='center left', bbox_to_anchor=(1, 0.5))
      
    t_min_str = params_bounds_min if params_bounds_min > 0 else "neg"
    fig.savefig(f"figs/full_evolution_order_{order}_{trotter_steps}_steps_tmin_{t_min_str}.png", bbox_inches = "tight")
    
    plt.show()
    
#################################################################
# ============================================================= #
#################################################################

def full_trotter_circ_no_bind(order, trotter_steps=4, uniform_times=True):
    '''
    this is basically the same as `full_trotter_circ`, but without
    the bind_parameters. That is, the circuit will be returned without the parameters set
    plus, measurement of the 3 determined qubits is added.
    '''

    qr = QuantumRegister(7)
    cr = ClassicalRegister(3)
    
    qc = QuantumCircuit(qr, cr)

    qc.x([3,5])  
    
    if uniform_times:

        Trot_gate = trotter_step(order, Parameter('t'))

        for _ in range(trotter_steps):

            qc.append(Trot_gate, [qr[1], qr[3], qr[5]])
            
    else:
                             
        for i in range(trotter_steps):
            
            Trot_gate = trotter_step(order, Parameter(f't{i}'))
                                     
            qc.append(Trot_gate, [qr[1], qr[3], qr[5]])   

    # measurement -- very important for the optmization! 
    qc.measure([1, 3, 5], [0, 1, 2])     

    return qc

#################################################################
# ============================================================= #
#################################################################

def plot_loss(losses_dict, quadratic_loss):
    '''
    this plots the evolution of the loss function per iterations
    '''
    
    plt.figure(figsize=(12, 6))
    
    plt.title("Optimization process - with SLSQP")
    
    for eps, losses in losses_dict.items():
        
        plt.plot(losses, label=f'eps={eps}')

    if quadratic_loss:
        plt.axhline(y=0, color='red', ls='--', label='Global minimum')
    else:
        plt.axhline(y=-1, color='red', ls='--', label='Global minimum')
    
    plt.ylabel('loss')
    plt.xlabel('iterations')
    
    plt.legend(prop={'size': 12}, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.show()
    
#################################################################
# ============================================================= #
#################################################################

def plot_param_evolution(params, opt_name):
    '''
    plot evolution of parameters values over the optimization steps
    params: list of params at each step, only for the best opt
    opt_name: name of the best opt
    '''

    params = np.array(params)

    plt.figure(figsize=(12, 6))
    
    plt.title(f"Evolution of parameters - {opt_name} optimizer")

    for i in range(params.shape[1]):

        plt.plot(params[:, i], label=f't{i+1}')

    plt.ylabel('parameter value')
    plt.xlabel('iterations')

    plt.legend(prop={'size': 12}, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
    
#################################################################
# ============================================================= #
#################################################################

def optimize_params_constrained(qc, backend, target_time=np.pi,
                                maxiter=200,
                                eps=0.1, tol=1e-10, ftol=1e-10,
                                params_bounds_min=0,
                                quadratic_loss=False):
    '''
    this function performs the classical optimization of the variational circuit.
    it employs the SLSQP constrained optimization method 
    (given the constraints on the total evolution time and allowed valus for parameters which are trotter steps times)
    please see comments throughout the function to understand its inner workings
    - eps (float): single epsilon, or list of epsilons (step size used for numerical approximation of the Jacobian)
    - params_bounds_min: inferior bound of parameters
    - quadratic_loss: binary flag, wheter or not to use a quadratic loss
    '''
    
    # ==================================================
    
    # loss function, it's just the fidelity
    def loss_trotter(parameters, qc=qc, backend=backend, quadratic_loss=quadratic_loss):
    
        params_dict = {param: time for param, time in zip(qc.parameters, parameters)}

        qc = qc.bind_parameters(params_dict)

        counts = execute(qc,
                         backend, 
                         shots=1e5, 
                         seed_simulator=42).result().get_counts()

        fid = counts["110"]/sum(counts.values())

        if quadratic_loss:
            return (1-fid)**2
        else:
            # because the optmizer will minimize, but we wanto to
            # maximize the fidelity.
            return -fid
    
    # ==================================================
    
    # params_bounds
    params_bounds=(params_bounds_min, target_time)
    
    # random params to start with
    # new: within the bounds!!
    np.random.seed(42)
    trotter_init_params = np.random.uniform(params_bounds[0], params_bounds[1], qc.num_parameters)
    
#     # but we'll normalize the params to sum to target_time
#     trotter_init_params = trotter_init_params*(target_time/trotter_init_params.sum())

    # ==================================================
        
    # it it's not a list, put single value within
    # a list, so that the loop below works!
    if isinstance(eps, list):
        eps_list = eps
    else:
        eps_list = [eps]
        
    # ==================================================
    
    # to save optimization results
    results = {"optimizer" : [],
               "eps" : [],
               "final_params" : [],
               "final_loss": []}
    
    # ==================================================
    
    print("\nStarting optimization!\n")
        
    # these are important for the plots
    losses_dict = {}
    params_dict = {}
    
    for eps_ in eps_list:

        start = time.time()

        print()
        print("="*50)
        print(f"Optimizer: SLSQP\neps = {eps_}")
        print("="*50)
        print()

        slsqp_loss = []
        slsqp_params = []

        def slsqp_callback(xk):

            # callback doesn't give the values of loss directly, only the current parameters
            # so I must calculate the loss again.
            loss = loss_trotter(parameters=xk, qc=qc, backend=backend, quadratic_loss=quadratic_loss)

            slsqp_loss.append(loss)
            slsqp_params.append(xk)

            n_iters = len(slsqp_params)

            print(f'Iter {n_iters} done!')
            print(f'Loss value: {loss}')
            with np.printoptions(precision=3, suppress=False):
                print(f'Current parameters: {xk} (sum to {xk.sum():.2f})\n')

        # all parameters must be positive
        bounds = [params_bounds]*qc.num_parameters

        # parameters must sum to target_time!
        constraints = {'type': 'eq', 'fun': lambda x: target_time - sum(x)}

        # options for the opt solver
        options = {"maxiter" : maxiter,
                   "verbose" : 3,
                   "disp" : True,
                   "eps" : eps_,
                   "tol" : tol,
                   "ftol" : ftol}

        results_slsqp = SciPyOptimizer(method="SLSQP",
                                       options=options,
                                       constraints=constraints,
                                       callback=slsqp_callback).optimize(num_vars=qc.num_parameters, 
                                                                         objective_function=loss_trotter, 
                                                                         variable_bounds=bounds,
                                                                         initial_point=trotter_init_params)

        results["optimizer"].append("slsqp")
        results["eps"].append(eps_)
        results["final_params"].append(results_slsqp[0])
        results["final_loss"].append(results_slsqp[1])
        
        losses_dict[eps_] = slsqp_loss
        params_dict[eps_] = slsqp_params

        stop = time.time()
        duration = time.strftime("%H:%M:%S", time.gmtime(stop-start))
        print(f"\nTotal time of optimization: {duration}")    

    # ==================================================
    # plot loss function evolution over optimization

    plot_loss(losses_dict, quadratic_loss)
    
    # ==================================================
    
    df_results = pd.DataFrame(results).sort_values("final_loss")
    
    print("\nOptimization results:\n")
    display(df_results)
    
    best_params = df_results.iloc[0]["final_params"]
    best_eps = df_results.iloc[0]["eps"]
    
    # ==================================================
    # plot parameters evolution, only for the best params
        
    plot_param_evolution(params=params_dict[best_eps], 
                         opt_name=f'SLSQP with eps={best_eps}')
    
    # ==================================================
    
    # sum to 1 (proportions)
    best_params_props = best_params/best_params.sum()
    
    with np.printoptions(precision=3, suppress=False):
        print(f"Best parameters (sum to {best_params.sum():.2f}):\t{best_params}")
        print(f"Best parameters (sum to 1):\t{best_params_props}")
    
    # ==================================================
    
    params_dict = {param: time for param, time in zip(qc.parameters, best_params)}
                                     
    qc = qc.bind_parameters(params_dict)
    
    # ==================================================
    
    return qc, best_params


#################################################################
# ============================================================= #
#################################################################

def optimize_params_and_run(order, trotter_steps, uniform_times, params_bounds_min, 
                            backend_opt, backend_state_tomo, quadratic_loss=False):
    '''
    this function builds the full variational circuit optimization pipeline, by calling
    previously defined functions. Please see their definitions for details.
    '''
    
    qc = full_trotter_circ_no_bind(order, trotter_steps, uniform_times)

    show_decompose(qc, n=0)

    # ==================================================

    qc, best_params = optimize_params_constrained(qc, backend=backend_opt, target_time=np.pi,
                                                  maxiter=200,
                                                  eps=[0.1, 0.01, 0.001], tol=1e-10, ftol=1e-10,
                                                  params_bounds_min=params_bounds_min,
                                                  quadratic_loss=quadratic_loss)
    
    # ==================================================

    show_decompose(qc, n=0)

    # ==================================================
    ###################################################################################
    # ==================================================

    view_single_trotter_step(order, Parameter('t'))

    # ==================================================
    # Trotterized Time Evolution

    st_qcs = state_tomagraphy_circs(order, trotter_steps,
                                    uniform_times=uniform_times, steps_times=best_params)

    print("\nAll steps + measurements of state tomography:")
    show_decompose(st_qcs[-1], 1)

    # ==================================================
    # Execution

    jobs = execute_st_simulator(st_qcs, backend=backend_state_tomo, id_str="")

    # ==================================================
    # Results Analysis

    fids = final_fidelities(jobs, st_qcs, order, trotter_steps)

    # ==================================================
    # Full evolution for t \in (0, pi)

    ts, probs, fidelity_pi = simulate_H_all_t(order, trotter_steps, backend_opt,
                                              uniform_times=uniform_times, steps_times=best_params)
    
    plot_simulation_H_all_t(ts, probs, fidelity_pi, order, trotter_steps, params_bounds_min, plot_theoretical=True)
    
    return fids, fidelity_pi, best_params


#################################################################
# ============================================================= #
#################################################################

def send_jobs(results_df, backend_state_tomo, uniform_times=False):
    '''
    this function sends jobs for execution. It's very important for hardware execution!
    '''
    
    jobs_dict = {}

    for i in range(results_df.shape[0]):

        order, n_steps, best_params, t_min = results_df.loc[i, "order n_steps best_params t_min".split()]
        
        key = f"order_{order}_{n_steps}_steps_tmin_{t_min}"

        # ==================================================
        # Trotterized Time Evolution with opt params

        st_qcs = state_tomagraphy_circs(order=order, trotter_steps=n_steps,
                                        uniform_times=uniform_times, steps_times=best_params)

        # ==================================================
        # Execution

        jobs = execute_st_simulator(st_qcs, backend=backend_state_tomo, id_str=key)

        # ==================================================
        # getting jobs

        jobs_dict[key] = jobs
        
    return jobs_dictf

#################################################################
# ============================================================= #
#################################################################

def hardware_exec_final_analysis(jobs_dict):
    '''
    returns a dict with the fidelities. 
    '''

    fids_dict = {}
    
    for key, jobs in jobs_dict.items():
        
        # ==================================================
        # Results Analysis

        fids = final_fidelities(jobs, st_qcs, order, trotter_steps)
        
        fids_dict[key] = fids
    
    return fids_dict

#################################################################
# ============================================================= #
#################################################################

# below, functions used to retrieve results

#################################################################
# ============================================================= #
#################################################################

def final_fidelities_retrieved(jobs, print_all_details=True):
    '''
    return the jobs' fidelities (8 for each), their mean and std.
    also prints their distribution (optionally via the flag print_all_details)
    '''
    
    # Compute tomography fidelities for each repetition
    fids = []
    
    for job in jobs:
        
        st_qcs = job.circuits()
        
        fid = state_tomo(job.result(), st_qcs)
        
        fids.append(fid)
        
    if print_all_details:
        print(f"\nFinal results for jobs above")
        
    print('\nState tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids)))

    if print_all_details:
        print("\nDistribution of state tomo fidelities:")
        display(pd.Series(fids, name="fids").describe())
    
    return fids, np.mean(fids), np.std(fids)

#################################################################
# ============================================================= #
#################################################################


def retrieve_job_ids_from_file(file, jakarta, print_all_details=True):
    '''
    reads the pickle file with the dict of IDs, and retrieve jobs.
    returns a dataframe with results of job execution.
    - file: string with the name of pickle file to be read
    - jakarta: IBMQBackend object (fixed as jakarta for this project)
    - print_all_details: binary flag which controls whether or not information about the jobs is printed
    '''
    
    # read the dict from pickle file
    with open(f'results/{file}', 'rb') as f:
        jobs_dict_ids = pickle.load(f)
    
    # ====================================
    
    results_hardware = {'order' : [], 
                        'n_steps' : [], 
                        't_min' : [], 
                        'state_tomo_fids_hardware' : [],
                        'mean_fid_hardware' : [],
                        'std_fid_hardware' : []}


    for kind, jobs_ids in jobs_dict_ids.items():

        print(f"\nType of circuit: {kind}")

        jobs = [jakarta.retrieve_job(job_id) for job_id in jobs_ids]

        status = [job._api_status for job in jobs]

        if print_all_details:
            print(f"\nStatus of group of jobs (should be only 'COMPLETED'): {set(status)}\n")
        if set(status) != {'COMPLETED'}:
            raise ValueError(f"\n\nJobs {kind} are not completed!\n\n")

        if print_all_details:
            print(*[f"ID job {i+1}: {x._job_id}" for i, x in enumerate(jobs)], sep="\n")

        # calss function which calculate the fidelities
        fids, fids_mean, fids_std = final_fidelities_retrieved(jobs, print_all_details)
  
        print()
        print("="*80)

        # ============================================

        order = int(kind.split("_")[1])
        steps = int(kind.split("_")[2])
        t_min = float(kind.split("_")[-1])

        results_hardware["order"].append(order)
        results_hardware["n_steps"].append(steps)
        results_hardware["t_min"].append(t_min)
        results_hardware["state_tomo_fids_hardware"].append(fids)
        results_hardware["mean_fid_hardware"].append(fids_mean)
        results_hardware["std_fid_hardware"].append(fids_std)

    df_results_hardware = pd.DataFrame(results_hardware)
    
    return df_results_hardware

#################################################################
# ============================================================= #
#################################################################

def merge_simulator_and_hardware_results(file, df_results_hardware, save_here=False):
    '''
    merges hardware results (second argument) and simulation results (read from file)
    - file: string with the name of pickle file to be read (same input of function retrieve_job_ids_from_file())
    - save_here: binary flag. Must be True if final results are to be saved in the same directory
        - in the main notebook, save_gere=True is to be used!
    '''
    
    # getting the kind of experiment and its date
    # this works because the structure of the pickle files' names was fixed as: 
    # dict_jobs_ids_EXPERIMENT_TYPE_YYYY-MM-DD.pkl
    # so, this variable extracts "EXPERIMENT_TYPE_YYYY-MM-DD"
    experiment_date_label = "_".join(file.split(".")[0].split("_")[-3:])
    
    # read results of simulation
    df_results_simulator = pd.read_parquet(f'./results/results_opt_{experiment_date_label}.parquet')

    df_results_simulator = df_results_simulator.rename(columns={"state_tomo_fids" : "state_tomo_fids_simulator"})

    # calculate mean and std fidelity for simulator
    df_results_simulator["mean_fid_simulator"] = df_results_simulator["state_tomo_fids_simulator"].apply(lambda x: np.mean(x))
    df_results_simulator["std_fid_simulator"] = df_results_simulator["state_tomo_fids_simulator"].apply(lambda x: np.std(x))

    cols_order =  ['order', 'n_steps', 't_min',  
                  'fid_pi', 'best_params', 
                  'state_tomo_fids_simulator','mean_fid_simulator', 'std_fid_simulator']

    df_results_simulator = df_results_simulator[cols_order].copy()

    # =========================================

    # merge simulator and hardware results
    final_results = (df_results_simulator.merge(df_results_hardware,
                                                                   on="order n_steps t_min".split(),
                                                                   how="outer")
                                                             .sort_values("mean_fid_hardware",
                                                                          ascending=False))
    
    # save in same directory (used in the main notebook)
    if save_here:
        final_results.to_parquet(f'./final_results_{experiment_date_label}.parquet')
    else:
        final_results.to_parquet(f'./results/final_results_{experiment_date_label}.parquet')
    
    return final_results

#################################################################
# ============================================================= #
#################################################################

def final_results_analysis(file, jakarta, print_all_details, save_here=False):
    '''
    this calls the functions retrieve_job_ids_from_file() and merge_simulator_and_hardware_results()
    - file: string with the name of pickle file to be read
    - jakarta: IBMQBackend object (fixed as jakarta for this project)
    - print_all_details: binary flag which controls whether or not information about the jobs is printed
    - save_here: binary flag. Must be True if final results are to be saved in the same directory
        - in the main notebook, save_gere=True is to be used!
    '''
    
    df_results_hardware = retrieve_job_ids_from_file(file, jakarta, print_all_details)
    
    final_results = merge_simulator_and_hardware_results(file, df_results_hardware, save_here)
    
    print(f"\nFinal results (comparing simulation and hardware execution):\n")
    display(final_results[['order', 'n_steps', 't_min', 
                           'mean_fid_simulator', 'std_fid_simulator',
                           'mean_fid_hardware', 'std_fid_hardware']])

#################################################################
# ============================================================= #
#################################################################

def final_df_results(final_results):
    
    df_final = pd.concat([pd.read_parquet(f"./results/{file}") for file in final_results]).sort_values("mean_fid_hardware",
                                                                                                        ascending=False)
    df_final['max_fid_hardware'] = df_final['state_tomo_fids_hardware'].apply(lambda x: x.max())
    df_final['max_fid_simulator'] = df_final['state_tomo_fids_simulator'].apply(lambda x: x.max())

    cols_order = ['order', 'n_steps', 't_min', 'fid_pi', 'best_params',
                  'state_tomo_fids_simulator', 'mean_fid_simulator', 'std_fid_simulator', 'max_fid_simulator',
                  'state_tomo_fids_hardware', 'mean_fid_hardware', 'std_fid_hardware', 'max_fid_hardware']

    df_final = df_final[cols_order].copy()

    df_final.to_parquet(f'./results/final_results_all_experiments.parquet')

    return df_final

#################################################################
# ============================================================= #
#################################################################

def generate_fidelity_graphs(df_final, backend_opt):
    '''
    this functions shows the full dynamics fidelity plots for all the experiments, from best to worse
    graphs are generated and saved if not existent.
    '''

    for i in range(df_final.shape[0]):

        order, trotter_steps, best_params, params_bounds_min = df_final.iloc[i][['order', 'n_steps', 'best_params', "t_min"]]

        t_min_str = params_bounds_min if params_bounds_min > 0 else "neg"
        pic_path = f"figs/full_evolution_order_{order}_{trotter_steps}_steps_tmin_{t_min_str}.png"

        print("="*80)
        print(f"Order {order}; {trotter_steps} steps; t_min = {t_min_str}".center(80))
        print("="*80)

        try:

            display(Image(filename=pic_path))

        except:

            ts, probs, fidelity_pi = simulate_H_all_t(order, trotter_steps, sim_noisy_jakarta,
                                                      uniform_times=False, steps_times=best_params)

            plot_simulation_H_all_t(ts, probs, fidelity_pi, order, trotter_steps, params_bounds_min, plot_theoretical=True)
            
#################################################################
# ============================================================= #
#################################################################

def full_trotter_gates(order, trotter_steps=4, target_time=np.pi,
                       uniform_times=True, steps_times=None, draw_circ=False):
    '''
    this funtion returns a 3-qubit circuit with the gates implementing
    the specificed trotterization. This is used to inspect wheter or not
    the corresponding unitary differs from the identity, in the check_circuit_unitary() function
    '''
    
    qr = QuantumRegister(3)
    qc = QuantumCircuit(qr)
    
    if uniform_times:

        Trot_gate = trotter_step(order, Parameter('t'))

        for _ in range(trotter_steps):

            qc.append(Trot_gate, [qr[0], qr[1], qr[2]])
            
        qc = qc.bind_parameters({qc.parameters[0]: target_time/trotter_steps})
    
    else:
        
        # check
        if len(steps_times) != trotter_steps:
            raise ValueError(f"Incorrect quantity of times {len(steps_times)}! Must be equal to number of steps {trotter_steps}")
                             
        for i in range(trotter_steps):
            
            Trot_gate = trotter_step(order, Parameter(f't{i}'))
                                     
            qc.append(Trot_gate, [qr[0], qr[1], qr[2]])
                                     
        params_dict = {param: time for param, time in zip(qc.parameters, steps_times)}
                                     
        qc = qc.bind_parameters(params_dict)
        
    if draw_circ:
        show_figure(qc.draw("mpl"))
         
    return qc

#################################################################
# ============================================================= #
#################################################################

def check_circuit_unitary(df_final, plot_matrix=True, draw_circuit=True):
    '''
    checks wheter or not the trotterization unitary is significantly different from the identity
    updates the final results table with this info.
    '''
    
    id_matrix = np.eye(8)

    dif_id_norm = []
    is_id = []

    for i in range(df_final.shape[0]):

        order, trotter_steps, best_params, params_bounds_min = df_final.iloc[i][['order', 'n_steps', 'best_params', "t_min"]]

        t_min_str = params_bounds_min if params_bounds_min > 0 else "neg"

        print("="*80)
        print(f"Order {order}; {trotter_steps} steps; t_min = {t_min_str}".center(80))
        print("="*80)

        # construct circuit - this has only the gates in the proposed trotterization
        qc = full_trotter_gates(order=order, trotter_steps=trotter_steps, target_time=np.pi,
                                uniform_times=False, steps_times=best_params, draw_circ=draw_circuit)

        # ===================================
        # simulate and plot the circuit unitary

        backend = Aer.get_backend("unitary_simulator")
        circuit_unitary_matrix = execute(qc, backend).result().get_unitary()

        if plot_matrix:
            print("\nCircuit unitary:\n")

            fig, ax = plt.subplots(1, 2, figsize=(15, 6))

            sns.heatmap(np.real(circuit_unitary_matrix), annot=True, ax=ax[0], annot_kws={"size": 8})
            ax[0].set_title("Real part")

            sns.heatmap(np.imag(circuit_unitary_matrix), annot=True, ax=ax[1], annot_kws={"size": 8})
            ax[1].set_title("Imaginary part")

            plt.tight_layout()
            plt.show()

        # ===================================
        # compare unitary with identity

        # frobenius norm of the difference, ||U - Id||_F
        diff_norm = np.linalg.norm(circuit_unitary_matrix-id_matrix, ord="fro") 
        cutoff = 1e-4

        dif_id_norm.append(diff_norm)

        print(f"\n||U - Id||_F = {diff_norm:.2e}")

        if diff_norm > cutoff:
            print(f"\nThis is above the cutoff {cutoff:.2e}, so the unitary significantly differs from the identity!")
            is_id.append(False)
        else:
            print(f"\nThis is below the cutoff {cutoff:.2e}, so the unitary does not significantly differ from the identity!")
            print(f"\nThis means that, effectively, the trotterization quantum circuit is not quite doing much.")
            is_id.append(True)

        print()

    df_final["dif_id_norm"] = dif_id_norm
    df_final["is_id"] = is_id
    
    # savin again, now with two added columns above
    df_final.to_parquet(f'./results/final_results_all_experiments.parquet')
    
    return df_final

#################################################################
# ============================================================= #
#################################################################


#################################################################
# ============================================================= #
#################################################################