# https://github.com/BOBO1997/qip2021_poster549/blob/bf363c03dd14f45a8dc130aa8eb438f40c3ec7d8/experiments/libs/star_bell_ineq.py
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pprint
# import multiprocessing as multi
# from multiprocessing import Pool
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit import IBMQ
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from qiskit.compiler import transpile
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
import qiskit.ignis.mitigation as mit
from qiskit.ignis.mitigation.measurement import complete_meas_cal, tensored_meas_cal, CompleteMeasFitter, TensoredMeasFitter


# ## Stabilizers

# In[ ]:


def compute_stabilizer_group(circuit):
    """
    Compute the stabilizer group for stabilizer circuit.
    
    Input
        circuit : qiskit.QuantumCircuit
    Output
        labels : list of string (each element is coeffient + pauli label)
    """
    state = qi.Statevector.from_instruction(circuit)
    labels = []
    for i in qi.pauli_basis(state.num_qubits):
        val = round(qi.state_fidelity(i.to_matrix()[0], state, validate=False))
        if val != 0:
            label = i.to_labels()[0]
            if val == 1:
                label = '+' + label
            else:
                label = '-' + label
            labels.append(label)
    return labels


# In[ ]:


def stabilizer_coeff_pauli(stabilizer):
    """
    Return the 1 or -1 coeff and Pauli label.
    Input
        stabilizer : string
    Output
        coeff : 1 or -1
        pauli : string (label of stabilizer)
    """
    coeff = 1
    pauli = coeff
    if stabilizer[0] == '-':
        coeff = -1
    if stabilizer[0] in ['+', '-']:
        pauli = stabilizer[1:]
    else:
        pauli = stabilizer
    return coeff, pauli


# In[ ]:


def stabilizer_measure_circuit(stabilizer, initial_circuit=None):
    """Return a stabilizer measurement circuits.
    
    Args:
        stabilizer (str): a stabilizer string
        initial_circuit (QuantumCircuit): Optional, the initial circuit.
    
    Returns:
        QuantumCircuit: the circuit with stabilizer measurements.
    """
    _, pauli = stabilizer_coeff_pauli(stabilizer)
    if initial_circuit is None:
        circ = QuantumCircuit(len(pauli))
    else:
        circ = initial_circuit.copy()
    for i, s in enumerate(reversed(pauli)):
        if s == 'X':
            circ.h(i)
        if s == 'Y':
            circ.sdg(i)
            circ.h(i)
    # circ.measure_all() # !?
    circ.barrier()
    num_cregs = 0
    for i, s in enumerate(reversed(pauli)):
        if s != 'I':
            circ.measure(i, i)
            num_cregs += 1
    return circ, num_cregs


# ## Prepare Circuits

# ### calibration circuits
# - Noisy measurements impact our ability to accurately measure the state fidelity. 
# - For our default example we calibrate our measurements for the CTMP method using states with two-qubit excitations.

# In[ ]:


def prepare_observables(qc, vertex, adj_list, optimization_level, measure_last=False, initial_layout=None):
    """
    Input
        qc, 
        vertex, 
        adj_list, 
        optimization_level, 
        measure_last = False, 
        initial_layout = None
    Output
        new_qc : QuantumCircuit
        : stabilizer label
    """
    new_qc = qc.copy("vertex" + str(vertex))
    new_qc += QuantumCircuit(new_qc.num_qubits, 1 + len(adj_list[vertex]))
    new_qc.h(vertex) # focus on vertex measured by stabilizer "X"
    if measure_last:
        new_qc.barrier()
    new_qc.measure([vertex] + adj_list[vertex], range(1 + len(adj_list[vertex]))) # measure all
    new_qc = transpile(new_qc, basis_gates=['u1','u2','u3','cx'], optimization_level=optimization_level, initial_layout=initial_layout)
    new_qc.reverse_bits() # reverse the endian
    return new_qc, "+" + "X" + "Z" * len(adj_list[vertex])


# In[ ]:


def prepare_two_qubit_exitation_labels(n):
    if n == 2:
        return ["00","01","10","11"]
    labels = ["0" * n]
    for i in range(n):
        for j in range(i+1, n):
            if not i == j:
                labels.append("0"* i + "1" + "0" * (j - i - 1) + "1" + "0" * (n - 1 - j ))
    labels.append("1" * n)
    return labels


# ### Prepare quantum circuits for star graph state using measurement grouping

# In[ ]:


def prepare_grouping_star_graph_qcs(qc,
                                    adj_list,
                                    optimization_level=0,
                                    measure_last=False,
                                    initial_layout=None,
                                    method = "tensored",
                                    do_mitigation = False):
    qcs, nums_meas_cal, metadatas, initial_layouts_, n = [], [], [], [], len(adj_list)
    if n <= 1:
        return [], [], [], [], [], []

    # first term
    new_qc1, stabilizer1 = prepare_observables(qc, 0, adj_list, optimization_level, measure_last, initial_layout)

    # second term
    new_qc2 = qc.copy()
    new_qc2 += QuantumCircuit(new_qc2.num_qubits, n)
    new_qc2.h(range(1,n)) # ZXXXXXXXX
    if measure_last:
        new_qc2.barrier()
    new_qc2.measure(range(n), range(n)) # measure all
    # new_qc2 = transpile(new_qc2, basis_gates=['u1','u2','u3','cx'], optimization_level=optimization_level, initial_layout=initial_layout)
    # new_qc2.reverse_bits() # reverse the endian 2021.07.01
    stabilizer2 = "+" + "Z" + "X" * (n-1)

    # for mitigation circuits

    meas_cal_circuits, metadata = None, None
    if method == "CTMP" or method == "ctmp":
        meas_cal_circuits, metadata = mit.expval_meas_mitigator_circuits(n, method=method)
    elif method == "tensored" or method == "tensor":
        qr = qiskit.QuantumRegister(n)
        mit_pattern = [[i] for i in range(n)]
        meas_cal_circuits, metadata = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='mcal') # metadata = state_label
    else:
        meas_cal_circuits, metadata = mit.expval_meas_mitigator_circuits(n, labels=prepare_two_qubit_exitation_labels(n))

    # meas_cal_circuits, metadata = mit.expval_meas_mitigator_circuits(n, labels=prepare_two_qubit_exitation_labels(n))
    nums_meas_cal.append(len(meas_cal_circuits))
    metadatas.append(metadata)

    if do_mitigation:
        qcs += ([new_qc1, new_qc2] + meas_cal_circuits)
        initial_layouts_ += ( [initial_layout, initial_layout] + [[initial_layout[i] for i in [0]+adj_list[0]]] * len(meas_cal_circuits) )
    else:
        qcs += [new_qc1, new_qc2]
        initial_layouts_ += ( [initial_layout, initial_layout] )

    return qcs, [1,1], nums_meas_cal, metadatas, [stabilizer1, stabilizer2], initial_layouts_


# In[ ]:


def prepare_grouping_star_graph_qcs_list(qc_graphs,             # list of qiskit.QuantumCircuit
                     adj_lists,             # list of adjacency list
                     optimization_level=0,  # int
                     measure_last=False, # bool
                     initial_layouts=None,
                     method = "tensored",
                     mitigation_limit=15): # bool
    """
    qcs_list :                   circuits to be executed on the real device
    nums_divide_list :      the information of how I divided a large shots to several circuits
    nums_meas_cal_list : the information of how I prepared the measurement_circuits for CTMP calibration
    metadatas_list :         used in prepare_meas_mitigator_list function
    stabilizers_list :          used in analyze_circuit function
    initial_layouts_list :     used in execute_circuits function
    """
    qcs_list, nums_divide_list, nums_meas_cal_list, stabilizers_list, metadatas_list, initial_layouts_list, = [], [], [], [], [], []
    if initial_layouts == None:
        initial_layouts = [list(range(len(adj_list))) for adj_list in adj_lists]
    for i, (qc, adj_list, initial_layout) in enumerate(zip(qc_graphs, adj_lists, initial_layouts)):
        qcs, nums_divide, nums_meas_cal, metadatas, stabilizers, initial_layouts_ = prepare_grouping_star_graph_qcs(qc, adj_list,
                                                                optimization_level=optimization_level, 
                                                                measure_last=measure_last, 
                                                                initial_layout=initial_layout,
                                                                method = method,
                                                                do_mitigation=len(adj_list) <= mitigation_limit)
        nums_meas_cal_list.append(nums_meas_cal)
        metadatas_list += metadatas
        stabilizers_list += stabilizers
        initial_layouts_list += initial_layouts_
        qcs_list += qcs
        nums_divide_list.append(nums_divide)
    return qcs_list, nums_divide_list, nums_meas_cal_list, metadatas_list, stabilizers_list, initial_layouts_list


# ###  Circuit Efficient Implementation

# In[ ]:


def prepare_grouping_star_graph_reduced_qcs(qc,
                                                        adj_list,
                                                        optimization_level=0,
                                                        measure_last=False,
                                                        initial_layout=None):
    qcs, initial_layouts_, n = [], [], len(adj_list)
    if n <= 1:
        return [], []

    # first term (XZZZ...Z)
    new_qc1, _ = prepare_observables(qc, 0, adj_list, optimization_level, measure_last, initial_layout)

    # second term
    new_qc2 = qc.copy()
    new_qc2 += QuantumCircuit(new_qc2.num_qubits, n)
    new_qc2.h(range(1,n)) # ZXXX...X
    if measure_last:
        new_qc2.barrier()
    new_qc2.measure(range(n), range(n)) # measure all
    new_qc2 = transpile(new_qc2, basis_gates=['u1','u2','u3','cx'], optimization_level=optimization_level, initial_layout=initial_layout)
    new_qc2.reverse_bits() # reverse the endian

    qcs += [new_qc1, new_qc2]
    initial_layouts_ += [initial_layout, initial_layout]

    return qcs, initial_layouts_


# In[ ]:


def prepare_grouping_star_graph_reduced_qcs_list(qc_graphs,             # list of qiskit.QuantumCircuit
                     adj_lists,             # list of adjacency list
                     optimization_level=0,  # int
                     measure_last=False, # bool
                     initial_layouts=None,
                     method=None):
    """
    qcs_list :                   circuits to be executed on the real device
    initial_layouts_list :     used in execute_circuits function
    metadata : 
    """
    qcs_list, initial_layouts_list = [], []
    if initial_layouts == None:
        initial_layouts = [list(range(len(adj_list))) for adj_list in adj_lists]
    for i, (qc, adj_list, initial_layout) in enumerate(zip(qc_graphs, adj_lists, initial_layouts)):
        qcs, initial_layouts_ = prepare_grouping_star_graph_reduced_qcs(qc, adj_list,
                                                                        optimization_level=optimization_level, 
                                                                        measure_last=measure_last, 
                                                                        initial_layout=initial_layout)
        initial_layouts_list += initial_layouts_
        qcs_list += qcs
    
    # for CTMP mitigation
    # n = max([len(adj_list) for adj_list in adj_lists])
    n = len(adj_lists[-1])
    meas_cal_circuits, metadata = None, None
    if method == "CTMP" or method == "ctmp":
        meas_cal_circuits, metadata = mit.expval_meas_mitigator_circuits(n, method=method)
    elif method == "tensored" or method == "tensor":
        qr = qiskit.QuantumRegister(n)
        mit_pattern = [[i] for i in range(n)]
        meas_cal_circuits, metadata = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='mcal') # metadata = state_label
    else:
        meas_cal_circuits, metadata = mit.expval_meas_mitigator_circuits(n, labels=prepare_two_qubit_exitation_labels(n))
    qcs_list += meas_cal_circuits
    initial_layouts_list += ([ [ initial_layouts[-1][i] for i in [0] + adj_lists[-1][0] ] ] * len(meas_cal_circuits) )

    return qcs_list, initial_layouts_list, metadata


# ## Execute Circuits

# In[ ]:


def split_to_lists(lst, n):
    begin, ret = 0, []
    while begin < len(lst):
        ret.append(lst[begin:begin + n])
        begin += n
    return ret


# In[ ]:


def execute_circuits(qcs, 
                     backend = "qasm_simulator", 
                     provider = None, 
                     shots = 8192, 
                     max_credits = 10, 
                     max_experiments = 900,
                     coupling_map=None,
                     basis_gates=None,
                     noise_model=None,
                     optimization_level=0,
                     initial_layout=None):
    if backend == "qasm_simulator":
        print("running on qasm_simulator")
        if initial_layout is None:
            return [execute(qcs, backend=Aer.get_backend('qasm_simulator'), shots=shots, coupling_map=coupling_map, basis_gates=basis_gates, noise_model=noise_model, optimization_level=optimization_level)]
        else:
            return [execute(qcs, backend=Aer.get_backend('qasm_simulator'), shots=shots, coupling_map=coupling_map, basis_gates=basis_gates, noise_model=noise_model, optimization_level=optimization_level, initial_layout=initial_layout)]
    else:
        if provider is None:
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
            print("default provider: ", provider)
        else:
            print("given provider: ", provider)
        print("running on", backend)
        jobs, i = [], 0
        while i < len(qcs):
            if initial_layout is None:
                jobs.append(execute(qcs[i:i + max_experiments], backend=provider.get_backend(backend), shots=shots, max_credits=max_credits,optimization_level=optimization_level))
            else:
                jobs.append(execute(qcs[i:i + max_experiments], backend=provider.get_backend(backend), shots=shots, max_credits=max_credits,optimization_level=optimization_level, initial_layout=initial_layout[i:i + max_experiments]))
            print("circuits from", i, "to", min(i + max_experiments - 1, len(qcs)), "are put on the real device.")
            i += max_experiments
    return jobs


# ## Analyze Results
# 
# jobs_to_counts
# job_ids_to_counts
# 
# ↓
# 
# collect_counts
# 
# ↓
# 
# if error mitigation, then readout_error_mitigation

#  #### Retrieve Jobs and Convert Result Format

# In[ ]:


def jobs_to_result(jobs):
    """
    Input
        jobs                     : list of job
    Output
        result_list : list of dictionary (raw counts of one experiment)
    """
    result_list = []
    for job in jobs:
        result_list.append(job.result())
    return result_list


# In[ ]:


def jobs_to_counts(jobs):
    """
    Input
        jobs                     : list of job
    Output
        counts_dict_list : list of dictionary (raw counts of one experiment)
    """
    counts_dict_list = []
    for job in jobs:
        counts_dict_list += job.result().get_counts()
    return counts_dict_list


# In[ ]:


def job_ids_to_result(job_ids, device):
    """
    Input
        jobs                     : list of job
        devices                  : backend (device information)
    Output
        result_list : list of dictionary (raw counts of one experiment)
    """
    results_jobs_list = []
    for job_id in job_ids:
        results_jobs_list.append(device.retrieve_job(job_id).result())
    return results_jobs_list


# In[ ]:


def job_ids_to_counts(job_ids, device):
    """
    Input
        jobs                     : list of job
        devices                  : backend (device information)
    Output
        counts_dict_list : list of dictionary (raw counts of one experiment)
    """
    counts_dict_list = []
    for job_id in job_ids:
        counts_dict_list += device.retrieve_job(job_id).result().get_counts()
    return counts_dict_list


# In[ ]:


def prepare_meas_mitigator_list(results_meas_cal, metadatas_list):
    meas_mitigator_list = []
    for result_mit_backend, metadata in zip(results_meas_cal, metadatas_list):
        meas_mitigator_list.append(mit.ExpvalMeasMitigatorFitter(result_mit_backend, metadata).fit())
    return meas_mitigator_list


# In[ ]:


def results_list_to_counts_dict_list(results_list):
    counts_dict_list = []
    for results in results_list:
        counts_dict_list.append(results.get_counts())
    return counts_dict_list


# #### Measurement Error Mitigation using the technique proposed by Bravyi et al.

# In[ ]:


def separate_results(one_job_results):
    return [qiskit.result.Result(backend_name=one_job_results.backend_name, 
                                 backend_version=one_job_results.backend_version,
                                 qobj_id=one_job_results.qobj_id, 
                                 job_id=one_job_results.job_id,
                                 success=True,
                                 results=[results]) for results in one_job_results.results]


# In[ ]:


def merge_results(results_list):
    results = []
    for res in results_list:
        results += res.results
    return qiskit.result.Result(backend_name=results_list[0].backend_name, 
                                 backend_version=results_list[0].backend_version,
                                 qobj_id=results_list[0].qobj_id, 
                                 job_id=results_list[0].job_id,
                                 success=True,
                                 results=results)


# In[ ]:


def flatten_results_jobs_list(results_jobs_list):
    results_list = []
    for one_job_results in results_jobs_list:
        results = separate_results(one_job_results)
        results_list += results
    return results_list


# In[ ]:


def arrange_results_list_for_grouping_star_graph(results_list, nums_divide_list, nums_meas_cal_list, limit=100):
    pos = 0
    results_graph_states = []
    results_meas_cal = []
    for i, (nums_divide, nums_meas_cal) in enumerate(zip(nums_divide_list, nums_meas_cal_list)): # graph wise
        for num_divide in nums_divide: # term wise
            results_graph_states += results_list[pos:pos + num_divide]
            pos += num_divide
        if i < limit: # start from 2
            for num_meas_cal in nums_meas_cal: # term wise
                results_meas_cal.append(merge_results(results_list[pos:pos + num_meas_cal]))
                pos += num_meas_cal
    return results_graph_states, results_meas_cal


# In[ ]:


def prepare_meas_mitigator_list(results_meas_cal, metadatas_list):
    meas_mitigator_list = []
    for result_mit_backend, metadata in zip(results_meas_cal, metadatas_list):
        meas_mitigator_list.append(mit.ExpvalMeasMitigatorFitter(result_mit_backend, metadata).fit(method="CTMP"))
    return meas_mitigator_list


# #### Analyze Whole Correlation of All Graphs (star graph version)

# In[ ]:


def extract_two_qubit_counts(counts, pos1, pos2):
    ret_counts = {"00": 0, "01": 0, "10": 0, "11": 0}
    for k, v in counts.items():
        ret_counts[k[pos1] + k[pos2]] += v
    return ret_counts


# In[ ]:


def compute_stddev_of_grouping(stddevs):
    return np.sqrt(sum([stddev ** 2 for stddev in stddevs]))


# In[ ]:


def analyze_circuits_for_star_graph(adj_lists, counts_dict_list, meas_mitigator=None, limit=100):
    """
    Input
        adj_lists         : list of adjacency list
        counts_list       : list of int list (list of counts)
        meas_mitigator : measurement mitigator
    Output
        expval_all_list : list of float (correlation of each graph)
        stddev_all_list : list of float (standard deviation of each graph)
        Es_all_list   : list of list (term-wise correlation of each graph)
        Ds_all_list   : list of list (term-wise stddev of each graph)
    """
    expval_all_list, stddev_all_list, Es_all_list, Ds_all_list = [], [], [], []
    begin = 0
    for adj_list in adj_lists:
        n = len(adj_list)
        if n > limit:
            break
        print("graph size:", n)
        if n <= 1:
            print("skipped\n")
            expval_all_list.append(0)
            stddev_all_list.append(0)
            Es_all_list.append([])
            Ds_all_list.append([])
            continue

        # for the first term
        expval1, stddev1 = mit.expectation_value(counts_dict_list[begin], 
                                                 qubits=range(n),
                                                 clbits=range(n),
                                                 meas_mitigator=meas_mitigator)
        Es_1, Ds_1 = [expval1], [stddev1]
        begin += 1 # update index of counts_dict_list


        # for the second term
        Es_2, Ds_2 = [], []
        sum_expval2, sum_stddev2 = 0, 0
        for pos in range(1, n): # recover the two qubit expectation values
            expval2, stddev2 = mit.expectation_value(extract_two_qubit_counts(counts_dict_list[begin], 0, pos), 
                                                     qubits=[0,pos],
                                                     clbits=[0,pos],
                                                     meas_mitigator=meas_mitigator)
            Es_2.append(expval2)
            Ds_2.append(stddev2)
        sum_stddev2 = compute_stddev_of_grouping(Ds_2)
        begin += 1 # update index of counts_dict_list

        sum_expval = np.sqrt(2) * ((n - 1) * sum(Es_1) + sum(Es_2))
        sum_stddev = np.sqrt(2 * ((stddev1 * (n - 1)) ** 2 +  sum_stddev2 ** 2) )
        Es = [Es_1, Es_2]
        Ds = [Ds_1, Ds_2]

        expval_all_list.append(sum_expval)
        stddev_all_list.append(sum_stddev)
        Es_all_list.append(Es)
        Ds_all_list.append(Ds)
        print("total correlation:", sum_expval, "\n")
    return expval_all_list, stddev_all_list, Es_all_list, Ds_all_list


# ## Additional Processing for Google Colab


# In[3]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'star_bell_ineq.ipynb'])


# In[ ]:




