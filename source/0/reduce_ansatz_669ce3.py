# https://github.com/amarjahin/kitaev_models_vqe/blob/839a1231c8870ab3f4c2015ec3ba6210fc4d0895/reduce_ansatz.py
from numpy import zeros, zeros_like, array, round, array, nonzero
from qiskit import transpile

from qiskit.circuit.library.standard_gates.rz import RZGate


def reduce_params(params, num_old_params, threshold=7, return_indices=False):
    params = array(params)
    imp_indx = [*range(num_old_params)] + list(nonzero(round(params[num_old_params::], threshold))[0]+num_old_params) 
    # imp_params = [params[i] for i in range(num_old_params, len(params)) if round(params[i], threshold) != 0]
    # imp_params = params[0:num_old_params] + imp_params
    imp_params = list(params[imp_indx])
    if return_indices:
        return imp_params, imp_indx
    return imp_params



# def reduce_params(params):
#     params = array(params)
#     # params = round(params, 5)
#     imp_indx = list(nonzero(round(params, 5))[0]) 
#     imp_params = [params[i] for i in imp_indx]
#     return imp_params, imp_indx

def enlarge_params(red_params, indices, length):
    big_params = zeros(length)
    big_params[indices] = red_params
    return list(big_params)

def rearrange_params(params, n=2):
    """The need of this function is basically for the difference in how qiskit arrange the parameters alphabitacally
       and how I arrange them in the actual circuit

    Args:
        params (list): list of params as arranged by qiskit
        n (int, optional): The number of terms grouped together (using same parameter) in the ansatz. Defaults to 2.

    Returns:
        list: rearranged list of params
    """
    arranged_params = zeros_like(params)
    l = len(params)//n
    for i in range(l): 
        for j in range(n):
            arranged_params[n*i + j] = params[i + j*l]
        # arranged_params[n*i + 1] = params[i + l]
    return list(arranged_params)

def reduce_ansatz(ansatz, params, num_terms, num_old_params, last_element, threshold=7): 
    """This reduce the ansatz by removing terms in the ansatz that have very small 
       value of paramters coming out of the optimizer

    Args:
        ansatz (QuantumCircuit): The ansatz
        params (list): list of the optimized paramters 
        num_terms (int): how many terms are grouped together using the same parameter
        num_old_params (int): number of already reduced parameters from previous calling of this function
        last_element (tuple): A marker tuple of the last element when this function was called before

    Returns:
        QuantumCircuit: The reduced ansatz
    """
    # arranged_params = list(params[0:num_old_params]) + rearrange_params(params[num_old_params::])
    arranged_params = rearrange_params(params[num_old_params::])
    start_inx = ansatz.data.index(last_element)
    # l = len(arranged_params)
    arranged_params = round(arranged_params, threshold)
    counter = 0
    indx = 0
    for i in ansatz.data[start_inx::]: 
        if isinstance(i[0], RZGate): 
            if arranged_params[indx] == 0: 
                ansatz.data.remove(i)
            counter = 1 + counter
            if counter % num_terms == 0:
                indx = indx +1 
                counter = 0

    ansatz.data.remove(last_element)
    ansatz = transpile(ansatz)

    return ansatz

