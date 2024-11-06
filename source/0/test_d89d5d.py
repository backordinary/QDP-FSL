# https://github.com/Connor-Mooney/Quantum-Walks/blob/373c76d41fcf7e841a7282617130da9b229dd256/test.py
from GluedTrees import ReducedGluedTree, GluedTree
from QuantumWalk import QuantumWalk
from qiskit import execute
from qiskit.visualization import plot_histogram
from qiskit import IBMQ, Aer
import matplotlib.figure as fig
import numpy as np
import scipy as sp
from heapq import nsmallest
from matplotlib import pyplot as plt


def eff_res(n):
    # returns the effective resistance
    r = 0
    for i in range(1, n):
        r += 2.*2.**(-i)
    r += 2.**(-n)
    return r

print(eff_res(2))
#n = 2
#print(2*(2**n - 1) - 1)
#g = GluedTree(n, {0}, {2*(2**n - 1) - 1})
#print(np.array2string(g.adjacencyMatrix, max_line_width=np.infty))

#qw = QuantumWalk(g, eff_res(2)/2, eff_res(2)/2)
#print(np.array2string(qw.modifiedAdjacencyMatrix, max_line_width= np.infty))
#print(np.array2string(np.round(qw.quantumWalkOperator, 3), max_line_width=np.infty))
#print(sp.linalg.eigvals(qw.quantumWalkOperator))

sim = Aer.get_backend('qasm_simulator')
phase_gaps = []
num_qubits = []
i = 2
redg = ReducedGluedTree(i, {0}, {2*i - 1})
qw = QuantumWalk(redg, eff_res(i)/2, eff_res(i)/2)
phase_estimator = qw.phase_estimation(6, [1, 0, 0, 0, 0])
print(phase_estimator.draw(output="text"))
job = execute(phase_estimator, sim, shots=1000)
res = job.result()
counts = res.get_counts(phase_estimator)
print(counts)
plot_histogram(counts)
plt.show()
# for i in range(2, 100):
#     #print("{}%".format(i*2/3))
#     #si_s = np.zeros((2*i+1, 1))
#     #psi_s[0] = 1
#     redg = ReducedGluedTree(i, {0}, {2*i-1})
#     qw = QuantumWalk(redg, eff_res(i)/2., eff_res(i)/2.)
#     eigvals, eigvects = np.linalg.eig(qw.quantumWalkOperator)
#     phases = np.imag(np.log(eigvals))
#     #if(i == 2):
#         #print(np.array2string(qw.modifiedAdjacencyMatrix, max_line_width= np.infty))
#
#         #print(np.array2string(qw.quantumWalkOperator, max_line_width=np.infty) +"\n")
#
#         #print(phases)
#     #print(phases)
#     phase_gap = nsmallest(2, np.abs(phases))[1]
#    # print(phase_gap)
#     #NOTE: This estimation of overlap seems to be FAULTY!!!!!
#     #np.absolute(eigvects[list(phases).index(phase_gap)].dot(psi_s)))
#     phase_gaps.append(phase_gap)
#     num_qubits.append(np.log2(2*i-1))
#
# with open('phase_gap_list.txt', 'w') as filehandle:
#     for listitem in phase_gaps:
#         filehandle.write('%s\n' % listitem)

# plt.plot(num_qubits, phase_gaps)
# plt.title("Log of phase gap vs. number of qubits")
# plt.ylabel("Log of phase gap of quantum walk operator")
# plt.xlabel("Number of qubits")
# plt.show()

#g = GluedTree(2)
#print(g.adjacencyMatrix)
#qw = QuantumWalk(g)
#print(qw.quantumWalkOperator)
#print("\n")
