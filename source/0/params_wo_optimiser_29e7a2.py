# https://github.com/anneriet/efficient_params_code/blob/6c7b2e018bd615d002c370fc5177dba524f0deb6/params_wo_optimiser.py
from openql import openql as ql
from qxelarator import qxelarator
import numpy as np
import os
import scipy.optimize as opt
from math import pi
import networkx as nx
import time

from qiskit import QuantumCircuit, transpile, execute, assemble
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit.circuit import Parameter, ParameterVector

import random as random

def max_cut_openql_noparams(angles, problem_size, edges, pnum, compiler, programname, qsim):
    program = ql.Program(programname, platf, problem_size, 0, 0)
    k = ql.Kernel ("qk", platf, problem_size) 
    for q in range(problem_size):
        k.gate("hadamard", [q])
    for it in range(pnum):
        for e in edges:
            k.gate("cnot", e)
            k.gate("rz", [e[1]], 0, angles[it])
            k.gate("cnot",e)
        for qb in range(problem_size):
            k.gate("rx", [qb], 0, angles[pnnum+it])
    for q in range(problem_size):
        k.gate("measure", [q])
    program.add_kernel(k)
    compiler.compile(program)

def max_cut_gen_kernel(k, problem_size, pnum):
    ql_paramlst = []
    for _ in range(2*pnum):
        ql_paramlst.append(ql.Param("ANGLE"))
    for q in range(problem_size):
        k.gate("hadamard", [q])
        
    for it in range(pnum):
        for e in graph.edges():
            k.gate("cnot", e)
            k.gate("rz", [e[1]], 0, ql_paramlst[it])
            k.gate("cnot",e)
        for qb in range(problem_size):
            k.gate("rx", [qb], 0, ql_paramlst[pnum +it])
    for q in range(problem_size):
        k.gate("measure", [q])
    return [k, ql_paramlst]   

def max_cut_openql_params(c, program, ql_paramlst, anglelst, qsim):
    c.compile(program, ql_paramlst, anglelst) 

def max_cut_circuit(problem_size, edges, steps):
    qc = QuantumCircuit(problem_size, problem_size)
    gammas = ParameterVector("gammas", steps)
    betas = ParameterVector("betas", steps)
    for q in range(problem_size):
        qc.h(q)
        
    for it in range(steps):
        for e in edges:
            qc.cnot(e[0], e[1])
            qc.rz(gammas[it], e[1])
            qc.cnot(e[0], e[1])
        for qb in range(problem_size):
            qc.rx(betas[it], qb)
    for q in range(problem_size):
        qc.measure(q, q)
    return [qc, gammas, betas]

def max_cut_qiskit(qc, anglelst, gammas, betas, qsim):
    beta = anglelst[0:steps]
    gamma = anglelst[steps:2*steps]
     
    compiled_circuit = qc.assign_parameters({gammas : gamma, betas : beta}, inplace=False)
    compiled_circuit.qasm(filename="qiskitfile.qasm")


total_number_of_executions = 2000
problem_size_lst = [15]
for _ in range (total_number_of_executions):
    problem_size = random.choice(problem_size_lst)
    graph = nx.random_regular_graph(4, problem_size)
    iterations = random.choice([1,2,4,6,8,10,12,14,16])

    programlang = random.choice(["qiskit", "openql_w_params", "openql_no_params"])
    steps = 3

    angles = list(list(random.random() for _ in range(2*steps)) for _ in range(iterations))
    start_time = time.time()

    curdir = os.path.dirname(__file__)

    if(programlang == "openql_no_params"):
        output_dir = os.path.join(curdir, 'test_output')
        ql.set_option('output_dir', output_dir)
        config_fn = os.path.join(curdir, 'hardware_config_qx.json')
        c = ql.Compiler("testCompiler", config_fn)
        c.add_pass_alias("Writer", "outputIR")
        c.set_pass_option("outputIR", "write_qasm_files", "yes")
        platf = ql.Platform("myPlatform", config_fn)
        c.add_pass("RotationOptimizer")

        programname = "params_openql"
        qx = qxelarator.QX()
        qx.set(os.path.join(output_dir + "/" + programname + ".qasm"))
        for i in range(iterations):
            max_cut_openql_noparams(angles[i], problem_size, graph.edges(), steps, c, programname, qx)
    if(programlang == "openql_w_params"):
        output_dir = os.path.join(curdir, 'test_output')
        ql.set_option('output_dir', output_dir)
        config_fn = os.path.join(curdir, 'hardware_config_qx.json')
        c = ql.Compiler("testCompiler", config_fn)
        c.add_pass_alias("Writer", "outputIR")
        c.set_pass_option("outputIR", "write_qasm_files", "yes")
        c.add_pass("RotationOptimizer")
        platf = ql.Platform("myPlatform", config_fn)

        programname = "params_openql"
        qx = qxelarator.QX()
        qx.set(os.path.join(output_dir + "/" + programname + ".qasm"))
        program = ql.Program(programname, platf, problem_size, 0, 0)
        k = ql.Kernel ("qk", platf, problem_size) 
        k, ql_paramlst = max_cut_gen_kernel(k, problem_size, steps)
        program.add_kernel(k)
        for i in range(iterations):
            max_cut_openql_params(c, program, ql_paramlst, angles[i], qx)
    if(programlang == "qiskit"):
        qiskit_circuit, gammas, betas = max_cut_circuit(problem_size, graph.edges(), steps)
        qsim = QasmSimulator()
        transpiled_circuit = transpile(qiskit_circuit, optimization_level=1)
        
        for i in range(iterations):
            max_cut_qiskit(qiskit_circuit, angles[i], gammas, betas, qsim)
    total_time = (time.time() - start_time)
    total_compile_time = total_time
    total_simulation_time = 0

    # print(programlang)
    # print("Total execution time without parameters: %s seconds. Total number of gates: %s gates." % (total_time, 2*problem_size+7*problem_size*steps))
    line = (time.strftime("%d/%m/%y %H:%M:%S") + "\t" +  # +"Total time:\t"
                                        programlang + "\t" +
                                        str(total_time) + "\t" +  #Compile time: \t" + 
                                        str(total_compile_time) +"\t" + # Simulation time:\t" + 
                                        str(total_simulation_time) + "\t" +  # problem_size:\t" + 
                                        str(problem_size) + "\t" +  #steps: \t" + 
                                        str(steps) + "\t" + # iterations:\t" + 
                                        str(iterations))
    # open a file to append
    outF = open(os.path.join(curdir, 'params_wo_optimiser.txt'), "a")
    outF.write(line)
    outF.write("\n")
    outF.close()

