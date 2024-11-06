# https://github.com/smml1996/quantum-research/blob/4b7798b0ff00d62ad642c9921ece4512b6dc5b41/reductscript.py
from qiskit import *
from qiskit.tools import visualization
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.aer import QasmSimulator
from qiskit import IBMQ
from math import sqrt
import time
import networkx as nx

def read_graph(name):
    num_vertices = 0
    num_edges = 0
    list_edges = []
    file = open("data/" + name , "r")
    for line in file:
        tokens = line.split()
        if(tokens[0] == "c"):
            #this line is a comment
            continue
        elif(tokens[0] == "p"and tokens[1] == "edge"):
            # save number of vertices and edges
            num_vertices = tokens[2]
            num_edges = tokens[3]
        else:
            # this is a line describing an edge
            temp = [int(tokens[1]), int(tokens[2])]
            list_edges.append(temp)
    file.close()
    return int(num_vertices), int(num_edges), list_edges

def reduce_to_SAT(nv, ne, ledges, k_colors):

    def get_vertex_color_repr(vertex):
        temp = list()
        for i in range (k_colors):
            temp.append(k_colors*vertex - i)
        return temp


    def get_vertex_contraints(vertex):
        constraints = []
        vertex_color_repr = get_vertex_color_repr(vertex)
        temp = []
        # select at least one color
        for c in vertex_color_repr:
            #temp += str(c) + " "
            temp.append(c)

        #temp+="0"
        constraints.append(temp);

        #select at most one color
        for c1 in range(len(vertex_color_repr)):
            for c2 in range(c1+1, len(vertex_color_repr)):
                temp = []
                temp.append(-vertex_color_repr[c1])
                temp.append(-vertex_color_repr[c2])
                constraints.append(temp)
        return constraints


    def get_edge_constraints(v1, v2):
        constraints = []
        v1_color_repr = get_vertex_color_repr(v1)
        v2_color_repr = get_vertex_color_repr(v2)

        for i in range(len(v1_color_repr)):
            temp = []
            temp.append(-v1_color_repr[i])
            temp.append(-v2_color_repr[i])
            #constraints.append("-" + str(v1_color_repr[i]) + " -" + str(v2_color_repr[i]) + " 0")
            constraints.append(temp)
        return constraints

    buffer = []

    #buffer.append("p cnf " + str(nv*k_colors) + " " + str(ne*k_colors + nv*int(1 +( k_colors * (k_colors-1) )/2 ) ))
    num_variables = nv*k_colors
    for i in range(1, nv+1):
        buffer.extend(get_vertex_contraints(i))

    for e in ledges:
        buffer.extend(get_edge_constraints(e[0], e[1]))
    return buffer, num_variables

def n_controlled_X(circuit, controls, target, ancillas,n):
    if(len(controls) < 1):
        pass
    elif len(controls) == 1:
        circuit.cx(controls[0], target)
    elif len(controls) == 2:
        circuit.ccx(controls[0], controls[1], target)
    else:
        # compute
        circuit.ccx(controls[0], controls[1], ancillas[0])
        for i in range(2, n):
            circuit.ccx(controls[i], ancillas[i-2], ancillas[i-1])

        # copy
        circuit.cx(ancillas[n-2], target[0])

        # uncompute
        for i in range(n-1, 1, -1):
            circuit.ccx(controls[i], ancillas[i-2], ancillas[i-1])
        circuit.ccx(controls[0], controls[1], ancillas[0])

def n_controlled_Z(circuit, controls, target, ancillas):
    """Implement a Z gate with multiple controls"""
    #if (len(controls) > 2):
    #    raise ValueError('The controlled Z with more than 2 ' +
    #                     'controls is not implemented')
    n = len(controls)
    if(n == 0):
        circuit.h(target)
        circuit.x(target)
        circuit.h(target)
        circuit.x(target)
    elif (n == 1):
        circuit.h(target)
        circuit.cx(controls[0], target)
        circuit.h(target)
    elif (n == 2):
        circuit.h(target)
        circuit.ccx(controls[0], controls[1], target)
        circuit.h(target)
    else:
        circuit.h(target)
        circuit.ccx(controls[0], controls[1], ancillas[0])
        for i in range(2, n):
            circuit.ccx(controls[i], ancillas[i-2], ancillas[i-1])
        # copy
        circuit.cx(ancillas[n-2], target[0])
        # uncompute
        for i in range(n-1, 1, -1):
            circuit.ccx(controls[i], ancillas[i-2], ancillas[i-1])
        circuit.ccx(controls[0], controls[1], ancillas[0])
        circuit.h(target)

def initialize(circuit, q_in, y, n):
    """(n+1)-qubit input state for Grover search."""
    for j in range(n):
        circuit.h(q_in[j])
    circuit.x(y)
    circuit.h(y)

def oracle(circuit, q_in, q_out, aux, n, formula, ancillas=[]):
    num_clauses = len(formula)
    for (k, clause) in enumerate(formula):
        # This loop ensures aux[k] is 1 if an odd number of literals
        # are true
        q_literals_in_clause = []
        for literal in clause:
            if literal > 0:
                q_literals_in_clause.append(q_in[literal-1])
                circuit.cx(q_in[literal-1], aux[k])
            else:
                q_literals_in_clause.append(q_in[-literal-1])
                circuit.x(q_in[-literal-1])
                circuit.cx(q_in[-literal-1], aux[k])
        # Flip aux[k] if all literals are true, using auxiliary qubit
        # (ancilla) aux[num_clauses]
        if len(q_literals_in_clause) > 1:
            n_controlled_X(circuit, q_literals_in_clause, aux[k], ancillas, len(q_literals_in_clause))
        for literal in clause:
            if literal < 0:
                circuit.x(q_in[-literal-1])
    # The formula is satisfied if and only if all auxiliary qubits
    # except aux[num_clauses] are 1
    n_controlled_X(circuit, aux, q_out, ancillas, num_clauses)
    # Flip back any auxiliary qubits to make sure state is consistent
    # for future executions of this routine; same loop as above.
    for (k, clause) in enumerate(formula):
        # This loop ensures aux[k] is 1 if an odd number of literals
        # are true
        q_literals_in_clause = []
        for literal in clause:
            if literal > 0:
                q_literals_in_clause.append(q_in[literal-1])
                circuit.cx(q_in[literal-1], aux[k])
            else:
                q_literals_in_clause.append(q_in[-literal-1])
                circuit.x(q_in[-literal-1])
                circuit.cx(q_in[-literal-1], aux[k])
        # Flip aux[k] if all literals are true, using auxiliary qubit
        # (ancilla) aux[num_clauses]
        if len(q_literals_in_clause) > 1:
            n_controlled_X(circuit, q_literals_in_clause, aux[k], ancillas, len(q_literals_in_clause))
        for literal in clause:
            if literal < 0:
                circuit.x(q_in[-literal-1])

def inversion_about_average(circuit, q_in, n, ancillas):
    """Apply inversion about the average step of Grover's algorithm."""
    for j in range(n):
        circuit.h(q_in[j])
    for j in range(n):
        circuit.x(q_in[j])
    n_controlled_Z(circuit, [q_in[j] for j in range(n-1)], q_in[n-1], ancillas)
    for j in range(n):
        circuit.x(q_in[j])
    for j in range(n):
        circuit.h(q_in[j])

def make_reads(circuit, q_in, cr, n):
    for i in range(n):
        circuit.measure(q_in[i], cr[i])

def estimate_ancilla_usage(num_variables, clauses):
    return max(num_variables-2, len(clauses)-1)

def grover(clauses, num_variables, simulated,backend_name ='ibmq_16_melbourne', extra_iters = 0):
    circuit = QuantumCircuit()
    q_in = QuantumRegister(num_variables, "var")
    q_out = QuantumRegister(1, "y")
    aux = QuantumRegister(len(clauses), "c")
    ancillas = []
    if(estimate_ancilla_usage(num_variables, clauses) > 0):
        ancillas = QuantumRegister(estimate_ancilla_usage(num_variables, clauses), "a")
        circuit.add_register(ancillas)
    cr = ClassicalRegister(num_variables)
    circuit.add_register(q_in)
    circuit.add_register(q_out)
    circuit.add_register(aux)
    circuit.add_register(cr)
    initialize(circuit,q_in, q_out, num_variables)
    for i in range(int(sqrt(2**num_variables))+extra_iters):
        oracle(circuit, q_in, q_out, aux, num_variables, clauses, ancillas)
        inversion_about_average(circuit, q_in, num_variables, ancillas)
    make_reads(circuit, q_in, cr, num_variables)
    result = ""
    if simulated:
        start_time = time.time()
        result = execute(circuit, QasmSimulator(),shots=1024).result()
        elapsed_time = time.time() - start_time
    else:
        provider = IBMQ.get_provider(hub='ibm-q')
        backend = provider.get_backend(backend_name)
        start_time = time.time()
        result = execute(circuit, backend, shots=1024).result()
        elapsed_time = time.time() - start_time

    print("execution time: ", elapsed_time)

    return result

def search_k_coloring(name, k, simulated=True, backend_name='ibmq_16_melbourne', extra_iters = 0):
    print("processing ", name, "... ")
    nv, ne, le = read_graph(name)
    clauses, num_variables = reduce_to_SAT(nv, ne, le, k)
    return grover(clauses, num_variables,simulated,backend_name=backend_name, extra_iters=extra_iters)


def is_coloring_valid(s, nv, le, k):
    map_colors = {}
    color = 0
    vertex_colors = []

    temp =""
    c = 0
    count_zeros = 0
    for i in range(len(s)-1, -1,-1):
        temp+= s[i]
        if(s[i] == '0'):
            count_zeros+=1
        c+=1
        if(c == k):

            if count_zeros == k:
                return False
            if k-1 != count_zeros :
                return False

            if temp in map_colors:
                vertex_colors.append(map_colors[temp])
            else:
                map_colors[temp] = color
                vertex_colors.append(color)
                color+=1
            c = 0
            temp = ""
            count_zeros = 0

    for e in le:
        if vertex_colors[e[0]-1] == vertex_colors[e[1]-1]:
            return False
    return True

def getAccuracy(k, counts, graph_name):
    nv, ne, le = read_graph(graph_name)
    ans = 0
    for s in counts:
        if is_coloring_valid(s, nv, le, k):
            ans+=counts[s]
    return ans/1024.0

search_k_coloring("graph1.txt", 1, extra_iters=2)
search_k_coloring("graph2.txt", 2, extra_iters=2)
search_k_coloring("graph3.txt", 1)
search_k_coloring("graph4.txt", 2)
answer = search_k_coloring("graph5.txt", 2).get_counts()
print(getAccuracy(2, answer, "graph5.txt"))
answer = search_k_coloring("graph5.txt", 2, extra_iters=2).get_counts()
print(getAccuracy(2, answer, "graph5.txt"))

answer = search_k_coloring("graph6.txt", 2).get_counts()
print(getAccuracy(2, answer, "graph6.txt"))
answer = search_k_coloring("graph6.txt", 2, extra_iters=2).get_counts()
print(getAccuracy(2, answer, "graph6.txt"))
