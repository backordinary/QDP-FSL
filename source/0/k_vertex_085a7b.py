# https://github.com/cook-jeremy/qaoa-kvertex/blob/201d98c5005cc506006a253c7b98f6b1480c3219/old/simulation/k-vertex.py
# Importing standard Qiskit libraries and configuring account
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram
from math import pi
from multiprocessing import Process
import datetime

# Run the quantum circuit on a statevector simulator backend
backend = BasicAer.get_backend('qasm_simulator')

# Global variables
num_nodes = 4
num_shots = 1000
k = 1

c = ClassicalRegister(num_nodes, 'c')
q = QuantumRegister(num_nodes, 'q')

def generate_graph():
    # Generate a random graph
    #G = nx.fast_gnp_random_graph(num_nodes,0.8)
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0,1),(1, 2), (1, 3), (0,3)])
    #nx.draw(G, with_labels=True, font_weight='bold')
    #plt.show()
    return G

def fracAngle(l,n):
    return np.arccos(np.sqrt(l/n))

def blocki(circ, n):
    circ.cx(q[n-2],q[n-1])
    circ.cu3(2.0*fracAngle(1,n),0,0, q[n-1], q[n-2])
    circ.cx(q[n-2],q[n-1])

def blockii(circ, n,l):
    circ.cx(q[n-l-1],q[n-1])
    # CCRy
    circ.cx(q[n-l], q[n-l-1])
    circ.u3(-0.5*fracAngle(l,n),0,0, q[n-l-1])   
    circ.cx(q[n-1], q[n-l-1])
    circ.u3(0.5*fracAngle(l,n),0,0, q[n-l-1])
    circ.cx(q[n-l], q[n-l-1])
    circ.u3(-0.5*fracAngle(l,n),0,0, q[n-l-1])
    circ.cx(q[n-1], q[n-l-1])
    circ.u3(0.5*fracAngle(l,n),0,0, q[n-l-1])
    circ.cx(q[n-l-1],q[n-1])  

def SCS(circ, n, k):
    blocki(circ, n)
    for l in range(2, k+1):
        blockii(circ, n, l)

def U(circ, n, k):
    for i in range(n,1,-1):
        SCS(circ, i, min(i-1,k))

def dicke(circ, n, k):
    # input string
    for i in range(n-k,n):
        circ.x(q[i])
    U(circ, n, k)

def expectation(G, counts):
    total = 0
    for state in counts:
        sa = []
        for i in state:
            sa.append(int(i))
        sa = list(reversed(sa))
        total_cost = 0
        for edge in G.edges:
            cost = 3
            if sa[edge[0]] == 1 or sa[edge[1]] == 1:
                cost = -1
            total_cost += cost
        f = -(1/4)*(total_cost - 3*G.number_of_edges())
        total += f*(counts[state]/num_shots)
    return total

def phase_separator(G, circ, gamma):
    # e^{-i \gamma deg(j) Z_j}}
    for i in range(len(G.nodes)):
        circ.rz(2*gamma*G.degree[i], q[i])
    # e^{-i \gamma Z_u Z_v}
    for edge in G.edges:
        circ.cx(q[edge[0]], q[edge[1]])
        circ.rz(2*gamma, q[edge[1]])
        circ.cx(q[edge[0]], q[edge[1]])

def eix(circ, beta, i, j):
    circ.h(q[i])
    circ.h(q[j])
    circ.cx(q[i], q[j])
    circ.u1(2*beta, q[j])
    circ.cx(q[i], q[j])
    circ.h(q[i])
    circ.h(q[j])

def eiy(circ, beta, i, j):
    circ.u2(0, pi/2, q[i])
    circ.u2(0, pi/2, q[j])
    circ.cx(q[i], q[j])
    circ.u1(2*beta, q[j])
    circ.cx(q[i], q[j])
    circ.u2(pi/2, pi, q[i])
    circ.u2(pi/2, pi, q[j])

def ring_mixer(G, circ, beta):
    # even terms
    for i in range(0, len(G.nodes)-1, 2):
        #print(str(i) + ", " + str((i+1) % len(G.nodes)))
        eix(circ, beta, i, (i+1) % len(G.nodes))
        eiy(circ, beta, i, (i+1) % len(G.nodes))

    # odd terms
    for i in range(1, len(G.nodes), 2):
        #print(str(i) + ", " + str((i+1) % len(G.nodes)))
        eix(circ, beta, i, (i+1) % len(G.nodes))
        eiy(circ, beta, i, (i+1) % len(G.nodes))

    # if number of edges in ring is odd, we have one leftover term
    if len(G.nodes) % 2 != 0:
        #print("special case")
        #print(str(len(G.nodes)-1) + ", 0")
        eix(circ, beta, len(G.nodes)-1, 0)
        eiy(circ, beta, len(G.nodes)-1, 0)

def qaoa(G, circ, gamma, beta, p):
    # prepare equal superposition over Hamming weight k
    dicke(circ, num_nodes, k)
    for i in range(p):
        phase_separator(G, circ, gamma)
        ring_mixer(G, circ, beta)
    # measure
    circ.measure(q,c)

def gamma_beta():
    G = generate_graph()
    num_steps = 100
    gamma = 0
    beta = 0
    p = 1
    g_list = []
    grid = []
    fig, ax = plt.subplots()

    print('0/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))
    for i in range(0, num_steps):
        for j in range(0, num_steps):
            circ = QuantumCircuit(q, c)
            qaoa(G, circ, gamma, beta, p)
            #circ.draw(interactive=True, output='latex')
            job = execute(circ, backend, shots=num_shots)
            result = job.result()
            counts = result.get_counts(circ)
            exp = expectation(G, counts)
            g_list.append(exp)
            #print('g: ' + str(gamma) + ', b: ' + str(beta) + ', exp: ' + str(exp))
            gamma += pi/(num_steps-1)
        beta += pi/(num_steps-1)
        gamma = 0
        grid.append(g_list)
        g_list = []
        print(str(i+1) + '/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))

    grid = list(reversed(grid))
    #print(grid)

    im = ax.imshow(grid, extent=(0, pi, 0, pi), interpolation='None', cmap=cm.inferno_r)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('$\\langle C \\rangle$', rotation=-90, va="bottom")

    plt.xlabel('$\\gamma$')
    plt.ylabel('$\\beta$')
    plt.title('$\\beta \\ vs \\ \\gamma$\nn=' + str(num_nodes) + ', k=' + str(k) + ', p=' + str(p) + ', grid_size=' + str(num_steps) + 'x' + str(num_steps))
    plt.show()

if __name__ == '__main__':
    gamma_beta()
    #test_expectation()
