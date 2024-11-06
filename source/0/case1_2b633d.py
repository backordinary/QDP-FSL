# https://github.com/rum-yasuhiro/experiments_crosstalk_multitasking/blob/387a8c2563058999d704924722ba9e99f44efdad/physical_distance_layout_pass/execution_time_and_density/case1.py
# python tools
import os
import pandas as pd

# qiskit tools
from qiskit.test.mock import FakeManhattan
from qiskit import Aer
from qiskit.providers.ibmq import ibmqbackend

# self-made tools
from utils.get_backend import get_IBM_backend
from utils.pickle_tools import pickle_load, pickle_dump
from physical_distance_layout_pass.execution_time_and_density.exp_tools import prep_experiments, run_experiments_on_backend, results_experiments
from utils.prep_QASMBench import PrepQASMBench, save_QuantumCircuit_data



def case1_1():

    bench_qc_names = [
        "wstate_n3", 
        "adder_n4", 
        "basis_change_n3",
        "cat_state_n4",
        "deutsch_n2",
        "fredkin_n3",
        "grover_n2",
        "hs4_n4",
        "inverseqft_n4",
        "iswap_n2",
        # "linearsovler_n3",
        # "ipn_n5",
        # "pec_en_n5",
    ]
    qasm_bench_path = os.getcwd() + "/qiskit_qasmbench.pickle"
    queued_qasmbench = PrepQASMBench(qasm_bench_path).qc_list(bench_names=bench_qc_names)
    file_path = os.getcwd() + "/physical_distance_layout_pass/execution_time_and_density/experiments_circuit/case1_1.csv"
    
    backend = FakeManhattan()
    
    df = prep_experiments(
        qc_list=queued_qasmbench,
        backend = backend,
        physical_dist_list=[0, 1],
        save_path=file_path,
        output=True,
    )

    print(df)

def case1_2():

    # Prepare the experiments
    bench_qc_names = [
        "basis_change_n3",
        "cat_state_n4",
        "deutsch_n2",
        "fredkin_n3",
        "grover_n2",
        "hs4_n4",
        "inverseqft_n4",
        "iswap_n2",
        "wstate_n3", 
        "adder_n4", 
        "basis_change_n3",
        "cat_state_n4",
        "deutsch_n2",
        "fredkin_n3",
        "grover_n2",
        "hs4_n4",
        "inverseqft_n4",
        "iswap_n2",
        "wstate_n3", 
        "adder_n4", 
        "basis_change_n3",
        "cat_state_n4",
        "deutsch_n2",
        "fredkin_n3",
        "grover_n2",
        "hs4_n4",
        "inverseqft_n4",
        "iswap_n2",
        "wstate_n3", 
        "adder_n4", 
        "basis_change_n3",
        "cat_state_n4",
        "deutsch_n2",
        "fredkin_n3",
        "grover_n2",
        "hs4_n4",
        "inverseqft_n4",
        "iswap_n2","wstate_n3", 
        "adder_n4", 
        "basis_change_n3",
        "cat_state_n4",
        "deutsch_n2",
        "fredkin_n3",
        "grover_n2",
        "hs4_n4",
        "inverseqft_n4",
        "deutsch_n2",
        "iswap_n2",
        "wstate_n3", 
        "adder_n4", 
        "basis_change_n3",
        "cat_state_n4",
        "deutsch_n2",
        "fredkin_n3",
        "grover_n2",
        "hs4_n4",
        "inverseqft_n4",
        "iswap_n2",
        "wstate_n3", 
        "adder_n4", 
        "basis_change_n3",
        "cat_state_n4",
        "deutsch_n2",
        "fredkin_n3",
        "grover_n2",
        "adder_n4", 
        "hs4_n4",
        "inverseqft_n4",
        "iswap_n2",
        "wstate_n3", 
        "adder_n4", 
        "basis_change_n3",
        "cat_state_n4",
        "deutsch_n2",
        "fredkin_n3",
        "grover_n2",
        "hs4_n4",
        "inverseqft_n4",
        "iswap_n2",
        "wstate_n3", 
        "adder_n4", 
        "basis_change_n3",
        "cat_state_n4",
        "deutsch_n2",
        "fredkin_n3",
        "grover_n2",
        "hs4_n4",
        "inverseqft_n4",
        "iswap_n2",
        "wstate_n3", 
        "adder_n4", 
        "basis_change_n3",
        "cat_state_n4",
        "deutsch_n2",
        "fredkin_n3",
        "grover_n2",
        "hs4_n4",
        "inverseqft_n4",
        "iswap_n2",
        "inverseqft_n4",
        "iswap_n2",
        "inverseqft_n4",        
    ]

    qasm_bench_path = os.getcwd() + "/qiskit_qasmbench.pickle"
    queued_qasmbench = PrepQASMBench(qasm_bench_path).qc_list(bench_names=bench_qc_names)
    exp_info_path = os.getcwd() + "/physical_distance_layout_pass/execution_time_and_density/experiments_circuit/case1_2.pickle"
    
    backend = FakeManhattan()
    
    exp_df = prep_experiments(
        qc_list=queued_qasmbench,
        backend = backend,
        physical_dist_list=[1],
        save_path=exp_info_path,
        output=True,
    )

    exp_df = pickle_load(exp_info_path)
    print(exp_df)

    # Run the experiments on the quantum device
    # prepare backend
    # ibmq_device = get_IBM_backend("ibmq_qasm_simulator")
    # qiskit_simulator = get_IBM_backend("ibmq_qasm_simulator")

    ibmq_device = backend
    qiskit_simulator = backend
    
    job_info_path = os.getcwd() + "/physical_distance_layout_pass/execution_time_and_density/job_information/case1_2.pickle"

    job_df = run_experiments_on_backend(
        backend=ibmq_device, 
        simlator=qiskit_simulator,
        experiments_df=exp_df, 
        save_path=job_info_path, 
        output=True,
    )

    print(job_df)

if __name__ == "__main__":
    # save_path = os.getcwd() + "/qiskit_qasmbench.pickle"
    # basis_gates = ['id', 'rz', 'sx', 'x', 'cx', 'reset']
    # save_QuantumCircuit_data(save_path, basis_gates)

    case1_2()