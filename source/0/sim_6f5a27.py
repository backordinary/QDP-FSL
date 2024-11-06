# https://github.com/NYUAD-Team12/backend/blob/c082eb63a0b6c4095dd6cc4f754b2ff0c6ae9db0/Main/api/utils/sim.py

import time
import os
import numpy as np
import itertools

# from qiskit import  # for local classical simulator
# from qiskit import QuantumCircuit # to creat quantum circuits

# from qiskit.algorithms import NumPyMinimumEigensolver # classical solver to compare
# from qiskit.utils import QuantumInstance # to modify transpiler options for simulator

# from qiskit.algorithms import QAOA # VQE algorithm
# from qiskit_optimization.algorithms import MinimumEigenOptimizer # find Minimum
# from qiskit.algorithms.optimizers import COBYLA, POWELL # the optimizer
import qiskit_optimization as qo # for making quadratic programs
import qiskit_optimization.converters as qubo_convert

from scipy.optimize import minimize
from braket.circuits import Circuit
from braket.devices import LocalSimulator





def getskilldata(pep,skill):
    ret = np.zeros(len(pep))
    for ii,idx in enumerate(pep):
        for idx2 in pep[idx]:
            if idx2 == skill:
                ret[ii] = pep[idx][idx2]
    return ret





def qubo_form(data):
    qp = qo.QuadraticProgram()
    volunters, aid_providers = data
    peo_len, pro_len = len(volunters), len(aid_providers)
    qp.binary_var_dict([f'{j}{i}' for j in range(0,pro_len) for i in range(0,peo_len)])

    for idx in range(peo_len):
        constraint_vars = np.insert(np.zeros((pro_len,peo_len-1)), idx, np.ones(pro_len), axis=1)
        qp.linear_constraint(constraint_vars.flatten(),'=',rhs=1,name=f'nd{idx}')

    tot_sum = np.array([])
    for work in aid_providers:
        linear = -2*len(aid_providers[work][1])*np.ones(len(volunters))
        for skill in aid_providers[work][1]:
            linear += (aid_providers[work][1][skill] - getskilldata(volunters,skill))
        tot_sum = np.append(tot_sum, linear*aid_providers[work][0])
    qp.maximize(constant = 0, linear = tot_sum)

    QUBO = qubo_convert.QuadraticProgramToQubo()
    qp2 = QUBO.convert(qp)
    return qp2





def qaoa_solve(qubo_program,shots=1000):
    num_qubits = qubo_program.get_num_binary_vars()
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    # running on local simulator
    backend = Aer.get_backend('qasm_simulator')
    seed = 123
    cobyla = COBYLA(maxiter=500)
    quantum_instance = QuantumInstance(backend=backend, shots=shots, seed_simulator=seed, seed_transpiler=seed)
    qaoa_mes = QAOA(optimizer=cobyla, reps=3, quantum_instance=quantum_instance, initial_state = qc)
    qaoa = MinimumEigenOptimizer(qaoa_mes)
    result = qaoa.solve(qubo_program)
    return result





def bin2dict(data,result):
    volunters, aid_providers = data
    result = result.reshape(len(aid_providers),len(volunters))

    assignments = {}
    for idx, projects in enumerate(aid_providers):
        idxes = np.nonzero(result[idx])[0]
        assignments[projects] = [list(volunters.keys())[idx] for idx in idxes]
    return assignments





DEVICE =  LocalSimulator()

def qaoa_circuit(gammas, betas, n_qubits, qubo):
    """
    Given a QUBO instance and the number of layers p, constructs the corresponding parameterized QAOA circuit with p layers.
    Args:
        qubo: The quadratic program instance
        p: The number of layers in the QAOA circuit
    Returns:
        The parameterized QAOA circuit
    """
    size = len(qubo.variables)
    qubo_matrix = qubo.objective.quadratic.to_array(symmetric=True)
    qubo_linearity = qubo.objective.linear.to_array()
    
    circ = Circuit()
    X_on_all = Circuit().x(range(0, n_qubits))
    circ.add(X_on_all)
    H_on_all = Circuit().h(range(0, n_qubits))
    circ.add(H_on_all)
    
    p = len(gammas)
    
    #Outer loop to create each layer
    for i in range(p):
        for j in range(size):
            qubo_matrix_sum = np.sum(qubo_matrix[j,:])
            gate = Circuit().rz(j, angle=2 * gammas[i] * (qubo_linearity[j] + qubo_matrix_sum))
            circ.add(gate)
                  
        for j in range(size):
            for k in range(size):
                if j>k:
                    gate = Circuit().zz(k,j, angle=0.5 * qubo_matrix[j,k] * gammas[i])
                    circ.add(gate)
        
        for k in range(size):
            gate = Circuit().rx(k, 2 * betas[i])
            circ.add(gate)
    return circ

def cost_func(params, n_qubits, ising, n_shots, tracker):
    circuit_length = len(params)//2
    gamma, beta = params[:circuit_length], params[circuit_length:]
    circuit = qaoa_circuit(gamma, beta, n_qubits, ising)
    
    task = DEVICE.run(circuit, shots=n_shots)
    result = task.result()
    metadata = result.task_metadata

    meas_ising = result.measurements
    all_energies = [ising.objective.evaluate(responses) for responses in meas_ising] # np.diag(meas_ising @ ising @ meas_ising.T)
    
    energy_min = np.min(all_energies)
    optimal_string = meas_ising[np.argmin(all_energies)]
    if energy_min < tracker['opt_energy']:
        tracker.update({'opt_energy':energy_min})
        tracker.update({'opt_string':optimal_string})
    
    energy_expect = np.sum(all_energies) / n_shots
    return energy_expect

# OPTIONS = {'disp': False, 'maxiter': 500}
# OPT_METHOD = 'Powell'  # SLSQP, COBYLA, Nelder-Mead, BFGS, Powell, ...

def aws_solve(ising, depth = 3, n_shots = 100):
    n_qubits = ising.get_num_binary_vars()
    TRACK = {'opt_energy':float('inf'),'opt_string':None}

    # randomly initialize variational parameters within appropriate bounds
    gamma_initial = np.random.uniform(0, 2 * np.pi, depth).tolist()
    beta_initial = np.random.uniform(0, np.pi, depth).tolist()
    params0 = np.array(gamma_initial + beta_initial)

    # set bounds for search space
    bnds_gamma = [(0, 2 * np.pi) for _ in range(int(len(params0) / 2))]
    bnds_beta = [(0, np.pi) for _ in range(int(len(params0) / 2))]
    bnds = bnds_gamma + bnds_beta

    # run classical optimization (example: method='Nelder-Mead')
    result = minimize(
        cost_func,
        params0,
        args=(n_qubits, ising, n_shots, TRACK),
        options={'maxiter': 500},
        method='Powell',
        bounds=bnds,
    )

    # store result of classical optimization
    result_energy = result.fun
    result_angle = result.x

    return TRACK





def main(volunters, aid_providers, aws = True, run_quantum = False):
    qp = qubo_form((volunters, aid_providers))
    if aws:
        result_ob = aws_solve(qp)
        result = result_ob['opt_string']
        print('AWS result', 'GS', result_ob['opt_string'], 'With Energy', result_ob['opt_energy'])
    else:
        if run_quantum:
            result_ob = qaoa_solve(qp)
            result = result_ob.x
            print('Qiskit result', 'GS', result, 'With Energy', result_ob.fval)
        else:
            exact_mes = NumPyMinimumEigensolver()
            exact_eigensolver = MinimumEigenOptimizer(exact_mes)
            result_ob = exact_eigensolver.solve(qp)
            result = result_ob.x
            print('NumPy result', 'GS', result, 'With Energy', result_ob.fval)
    
    return bin2dict((volunters, aid_providers), result)
    





volunters = {'Anna': {'CLEANING': 2}, 'Bob': {'REPAIRING': 5, 'NURSING': 5}, 'Maria': {'COOKING': 3}, 'Ahmed':{'CLEANING':3}}

aid_providers = {'NGO1': (10, {'CLEANING': 3}),
  'NGO2': (15, {'REPAIRING': 3, 'CLEANING': 2}),
  'NGO3': (20, {'COOKING': 3, 'REPAIRING': 3})}





# main(volunters, aid_providers, False, False)








