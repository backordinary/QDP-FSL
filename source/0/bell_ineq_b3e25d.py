# https://github.com/BOBO1997/qip2021_poster549/blob/bf363c03dd14f45a8dc130aa8eb438f40c3ec7d8/experiments/libs/bell_ineq.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[11]:


def remaining_vertices(adj_list, n, F):
    remaining = set(range(n)) - set(F)
    for m in F:
        for v in adj_list[m]:
            remaining.discard(v)
    return list(remaining)


# In[12]:


def prepare_observables(qc, vertex, adj_list, optimization_level, measure_last=False, initial_layout=None):
    new_qc = qc.copy("vertex" + str(vertex))
    new_qc += QuantumCircuit(qc.num_qubits, 1 + len(adj_list[vertex]))
    new_qc.h(vertex) # focus on vertex "m"
    if measure_last:
        new_qc.barrier()
    new_qc.measure([vertex] + adj_list[vertex], range(1 + len(adj_list[vertex]))) # measure all
    new_qc = transpile(new_qc, basis_gates=['u1','u2','u3','cx'], optimization_level=optimization_level, initial_layout=initial_layout)
    new_qc.reverse_bits() # reverse the endian
    return new_qc


# In[13]:


def prepare_qcs(qc,
                adj_list,
                F,
                shots_scale=16,
                shots_per_circuit=8192,
                same_scale=False, # extra??
                optimization_level=0,
                measure_last=False,
                initial_layout=None,
                readout_error_mitigation=False):
    qcs, nums_divide, qubit_sublist, n = [], [], [], len(adj_list)
    if n <= 1:
        if readout_error_mitigation:
            return [], [], []
        return [], []
    remaining = remaining_vertices(adj_list, n, F)
    shift = len(format(shots_scale - 1, "0b")) - len(format(shots_per_circuit - 1, "0b"))
    for m in F:
        num_divide = 1 << max(0, (len(adj_list[m]) + shift))
        qcs += [prepare_observables(qc, m, adj_list, optimization_level, measure_last, initial_layout)] * num_divide
        nums_divide.append(num_divide)
        qubit_sublist.append([m] + adj_list[m])
        for v in adj_list[m]:
            num_divide = 1 << max(0, (len(adj_list[v]) + shift))
            qcs += [prepare_observables(qc, v, adj_list, optimization_level, measure_last, initial_layout)] * num_divide
            nums_divide.append(num_divide)
            qubit_sublist.append([v] + adj_list[v])
    for v in remaining:
        num_divide =1 << max(0, (len(adj_list[v]) + shift))
        qcs += [prepare_observables(qc, v, adj_list,  optimization_level, measure_last, initial_layout)] * num_divide
        nums_divide.append(num_divide)
        qubit_sublist.append([v] + adj_list[v])
    if readout_error_mitigation:
        return qcs, nums_divide, qubit_sublist
    return qcs, nums_divide


# In[14]:


def prepare_qcs_list(qc_graphs,             # list of qiskit.QuantumCircuit
                     adj_lists,             # list of adjacency list
                     Fs,                    # list of list of int (list of selected vertices)
                     shots_per_circuit,     # int 
                     shots_scale_per_graph, # list of int
                     same_scale=False,      # if True, then the number of each divide would be aimed to be the same scale of shots in terms of the measured qubits size
                     optimization_level=0,  # int
                     measure_last=False,
                     initial_layouts=None,
                     readout_error_mitigation=False):   # bool
    assert len(adj_lists) == len(Fs) and len(adj_lists) == len(shots_scale_per_graph)
    qcs_list, nums_divide_list, qubit_sublists = [], [], []
    if initial_layouts == None:
        initial_layouts = [list(range(len(adj_list))) for adj_list in adj_lists]
    for i, (qc, adj_list, F, shots_scale, initial_layout) in enumerate(zip(qc_graphs, adj_lists, Fs, shots_scale_per_graph, initial_layouts)):
        qcs, nums_divide, qubit_sublist = None, None, None
        if readout_error_mitigation:
            qcs, nums_divide, qubit_sublist = prepare_qcs(qc, adj_list, F, shots_scale, shots_per_circuit, 
                                                          same_scale=same_scale, optimization_level=optimization_level, 
                                                          measure_last=measure_last, initial_layout=initial_layout,
                                                          readout_error_mitigation=readout_error_mitigation)
        else:
            qcs, nums_divide = prepare_qcs(qc, adj_list, F, shots_scale, shots_per_circuit, 
                                           same_scale=same_scale, optimization_level=optimization_level, 
                                           measure_last=measure_last, initial_layout=initial_layout)
        qcs_list += qcs
        nums_divide_list.append(nums_divide)
        if readout_error_mitigation:
            qubit_sublists += qubit_sublist
    if readout_error_mitigation:
        return qcs_list, nums_divide_list, qubit_sublists
    return qcs_list, nums_divide_list


# #### Count the number of prepared quantum circuits

# In[15]:


def count_qcs(qc,
              adj_list,
              F,
              shots_scale=16,
              shots_per_circuit=8192,
              same_scale=False, # extra??
              optimization_level=0,
              measure_last=False,
              initial_layout=None):
    nums_divide, n = [], len(adj_list)
    if n <= 1:
        return 0, []
    # print(adj_list, n, F)
    remaining = remaining_vertices(adj_list, n, F)
    for m in F:
        num_divide = 1 << max(0, (len(adj_list[m]) + len(format(shots_scale - 1, "0b")) - len(format(shots_per_circuit - 1, "0b"))))
        nums_divide.append(num_divide)
        for v in adj_list[m]:
            num_divide = 1 << max(0, (len(adj_list[v]) + len(format(shots_scale - 1, "0b")) - len(format(shots_per_circuit - 1, "0b"))))
            nums_divide.append(num_divide)
    for v in remaining:
        num_divide = 1 << max(0, (len(adj_list[v]) + len(format(shots_scale - 1, "0b")) - len(format(shots_per_circuit - 1, "0b"))))
        nums_divide.append(num_divide)
    return sum(nums_divide), nums_divide


# In[16]:


def count_qcs_list(qcs,                   # list of qiskit.QuantumCircuit
                   adj_lists,             # list of adjacency list
                   Fs,                    # list of list of int (list of selected vertices)
                   shots_per_circuit,     # int 
                   shots_scale_per_graph, # list of int
                   same_scale=False,      # if True, then the number of each divide would be aimed to be the same scale of shots in terms of the measured qubits size
                   optimization_level=0,  # int
                   measure_last=False,
                   initial_layouts=None):   # bool
    assert len(adj_lists) == len(Fs) and len(adj_lists) == len(shots_scale_per_graph)
    num_qcs_list, nums_divide_list = 0, []
    for i, (qc, adj_list, F, shots_scale) in enumerate(zip(qcs, adj_lists, Fs, shots_scale_per_graph)):
        num_qcs, nums_divide = count_qcs(qc, adj_list, F, shots_scale, shots_per_circuit, same_scale=same_scale)
        # print(num_qcs)
        num_qcs_list += num_qcs
        nums_divide_list.append(nums_divide)
    return num_qcs_list, nums_divide_list


# ## Execute Circuits

# In[18]:


def split_to_lists(lst, n):
    begin, ret = 0, []
    while begin < len(lst):
        ret.append(lst[begin:begin + n])
        begin += n
    return ret


# In[19]:


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

#  #### Retrieve Jobs

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
        result_list += job.result()
    return result_list


# In[20]:


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
    result_list = []
    for job_id in job_ids:
        result_list.append(device.retrieve_job(job_id).result())
    return result_list


# In[21]:


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


#  #### Readout Error Mitigation (by calibration matrix)

# In[ ]:


def prepare_calibration_matrix(n, qubit_list, backend, initial_layout, scale=3):
    """
    n              : size
    qubit_list     : which virtual qubit to measure
    backend        : backend
    initial_layout : initial_layout
    """
    qr = qiskit.QuantumRegister(n)
    meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr, circlabel='mcal')
    coupling_map = backend.configuration().coupling_map
    noise_model = NoiseModel.from_backend(backend)
    # Execute the calibration circuits
    job = qiskit.execute(meas_calibs, backend=qiskit.Aer.get_backend('qasm_simulator'), shots=1 << (n+scale),
                         noise_model=noise_model, coupling_map=coupling_map, initial_layout=initial_layout)
    cal_results = job.result()
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, qubit_list=qubit_list, circlabel='mcal')
    return meas_fitter


# In[ ]:


def readout_error_mitigation(counts_list, meas_fitter, qubit_sublists):
    """
    Input 
        counts_list    : list or dict
        meas_fitter    : measurement fitter 
        qubit_sublists : qubit sublists for meas_fitter
    Output
        mitigated_counts_dict_list : list or dict according to the type of counts_list
    """
    mitigated_counts_dict_list = []
    for counts, qubit_sublist in zip(counts_list, qubit_sublists):
        meas_fitter_sub = meas_fitter.subset_fitter(qubit_sublist=qubit_sublist)
        meas_filter_sub = meas_fitter_sub.filter
        mitigated_counts_dict_list.append(meas_filter_sub.apply(counts))
    return mitigated_counts_dict_list


# #### Collect counts

# In[22]:


def sum_counts(counts_dicts, ret_type="list"):
    """
    merge divided shots of one graph state
    
    Input
        counts_dicts : list of dictionary
    Output
        : list of int (counts of one experiment)
    """
    n = len(list(counts_dicts[0].keys())[0])
    sum_of_counts = {format(i, '0' + str(n) + 'b') : 0 for i in range(1 << n)} # initialize by 0
    for counts in counts_dicts:
        for key in counts:
            sum_of_counts[key] += counts[key]
    if ret_type == "list":
        return list(sum_of_counts.values())
    elif ret_type == "dict":
        return sum_of_counts
    else:
        return None


# In[23]:


def collect_counts(counts_dict_list, graph_sizes, nums_divide_list, ret_type="list"):
    """
    Input
        counts_dict_list : list of dictionary
        graph_sizes      : list of int
        nums_divide_list : list of int ( == len(graph_sizes) )
    Output
        counts_list : list of int list (list of counts)
    """
    counts_list, sizes, divide_list, begin = [], [], [], 0
    for graph_size, nums_divide in zip(graph_sizes,nums_divide_list):
        if graph_size > 1:
            sizes.append(graph_size)
            divide_list.append(nums_divide)
    for i, (graph_size, nums_divide) in enumerate(zip(sizes, divide_list)):
        for num_divide in nums_divide:
            counts_list.append(sum_counts(counts_dict_list[begin:begin + num_divide], ret_type=ret_type))
            begin += num_divide
    return counts_list


# #### Design Expectation Function

# In[24]:


def make_parity_table(n):
    parity_table = {format(i, '0b') : 0 for i in range(1 << n)}
    parity_table['1'] = 1
    for i in parity_table:
        if len(i) > 1:
            parity_table[i] = parity_table[format(int(i[1:], 2), '0b')] ^ int(i[0])
    return list(parity_table.values())


# In[25]:


def E(shots, counts, parity_table=None):
    """
    Input
        shots        : int (number of shots)
        counts       : list of int (result of measurement)
        parity_table : list of 0 and 1 (map measurement patterns to 0 or 1 (that is, 1 or -1))
    Output
        : float (correlation)
    """
    
    parity_table = make_parity_table(len(str(counts)) - 1) if parity_table is None else parity_table
    corr = 0
    for i in range(len(counts)): # len(counts) == 2 ** (number of observables used in the term)
        if parity_table[i] == 0: # if index has odd number of 1 in binary expression, then the eigenvalue is -1
            corr += counts[i]
        else:
            corr -= counts[i]
            
    return corr / shots


# #### Analyze Whole Correlation

# In[26]:


def analyze_circuits(adj_lists, Fs, counts_list, shots_per_circuit, nums_divide_list, parity_table):
    """
    Input
        adj_lists         : list of adjacency list
        Fs                : list of vertex subset
        counts_list       : list of int list (list of counts)
        shots_per_circuit : list of int
        nums_divide_list  : list of list of int
        parity_table      : list of 0 or 1
    Output
        corr_all_list : list of float (correlation of each graph)
        Es_all_list   : list of list (term-wise correlation of each graph)
    """
    assert len(adj_lists) == len(Fs) and len(adj_lists) == len(nums_divide_list)
    corr_all_list, Es_all_list = [], []
    begin = 0
    for adj_list, F, nums_divide in zip(adj_lists, Fs, nums_divide_list):
        nums_divide = iter(nums_divide)
        print("graph size:", len(adj_list))
        Es_F, corr_F, n = [], 0, len(adj_list)
        if n <= 1:
            print("skipped\n")
            corr_all_list.append(0)
            Es_all_list.append([])
            continue
        remaining = remaining_vertices(adj_list, n, F)
        for m in F:
            corr_itself = E(shots_per_circuit * next(nums_divide), counts_list[begin], parity_table)
            corr_deg, begin = corr_itself * len(adj_list[m]), begin + 1
            Es_m = [corr_itself]                  + [E(shots_per_circuit * next(nums_divide), counts_list[begin:begin + len(adj_list[m])][j], parity_table) for j, v in enumerate(adj_list[m])]
            sum_corr, begin = corr_deg + sum(Es_m[1:]), begin + len(adj_list[m])
            print("correlation on n[", m, "]:", sum_corr)
            corr_F += sum_corr
            Es_F.append(Es_m)

        # remainig part
        Es_R = [E(shots_per_circuit * next(nums_divide), counts_list[begin:begin+len(remaining)][i], parity_table) for i, v in enumerate(remaining)]
        corr_R = sum(Es_R)
        begin += len(remaining)
        print("correlation on remaining vertices:", corr_R)

        corr_F *= np.sqrt(2)
        corr_all = corr_F + corr_R
        corr_all_list.append(corr_all)
        Es_all_list.append([Es_F, Es_R])
        print("total correlation:", corr_all, "\n")
    return corr_all_list, Es_all_list


# In[4]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'bell_ineq.ipynb'])

