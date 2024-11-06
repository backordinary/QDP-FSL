# https://github.com/KernalPanik/QC_Optimizer/blob/e49775ec39526568da6f543d4e1f31d24afcbcd8/megarandom.py
'''
This script generates a lot (100) of random scripts, which are later 
optimized. Complete Test can take up to 6-8 hours.

This test measures average gate count in random circuits, and average gate count
in optimized random circuits.

In the end, a txt file will be generated with test results. Please check megarandom_test_results.txt file
'''
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from Optimizer_main import Adaptive_Optimizer

def mega_random_test():
    try_count = 100
    ao = Adaptive_Optimizer()
    for i in range(3, 17):
        avg_before_optimize = 0
        avg_after_optimize = 0
        for j in range(0, try_count):
            circ = random_circuit(i, i*i, 2, False, False, False, None)

            circ_ops = circ.count_ops()
            circ_op_count = 0
            for op in circ_ops:
                circ_op_count += circ_ops[op]
            
            avg_before_optimize += circ_op_count
            opti_circ = ao.run_optimization_no_qasm(circ)

            opti_circ_ops = opti_circ.count_ops()
            opti_circ_op_count = 0
            for op in opti_circ_ops:
                opti_circ_op_count += opti_circ_ops[op]
            
            avg_after_optimize += opti_circ_op_count

            print("before opti: " + str(circ_op_count))
            print("after opti: " + str(opti_circ_op_count))

        print("Optimizing random " + str(i) + " qubit circuits")
        print("Before optimization avg: " + str(avg_before_optimize/try_count))
        print("After optimization avg: " + str(avg_after_optimize/try_count))

        with open("megarandom_test_results.txt", 'a') as f:
            f.write("Optimizing random " + str(i) + " qubit circuits\n")
            f.write("Before optimization avg: " + str(avg_before_optimize/try_count)+'\n')
            f.write("After optimization avg: " + str(avg_after_optimize/try_count)+'\n')

mega_random_test()