# https://github.com/Jh0mpis/GrafosAleatoriosUsandoComputacionCuantica/blob/a73e7261cdb751426104bcff38c2f5a207e7e4de/RandomGraph.py
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pandas as pd
import random
from qiskit import QuantumCircuit
import qiskit

np.random.seed(823)

n = 5

G1 = nx.balanced_tree(2,n)
G2 = nx.balanced_tree(2,n)
G3 = nx.union(G1,G2,rename=('1-','2-'))

HaveConnection = pd.Series({i:0 for i in range(2**(n)-1,2**(n+1)-1)})

def GenEdges(HaveConnection):
    return random.sample(list(HaveConnection.where(HaveConnection<2).dropna().index),2)

for i in range(2**n-1,2**(n+1)-1):
    newEdges = GenEdges(HaveConnection)
    for j in newEdges:
        G3.add_edge(f'1-{i}',f'2-{j}')
        HaveConnection[j] += 1

pos = {}
pos[f'1-0'] = (0,0.5)
pos[f'2-0'] = (1,0.5)
k = 0.5
for i in range(1,n+1):
    print(i        /(2*n+1))
    delta = k/2
    for j in range(2**i,2**(i+1)):        
        print(i        /(2*n+1),0.5)


        pos[f'1-{j-1}'] = (i        /(2*n+1),delta+2*delta*(j-2**i))
        pos[f'2-{j-1}'] = ((2*n-i+1)/(2*n+1),delta+2*delta*(j-2**i))
    k = delta
xs = np.linspace(0,1,12)


nx.draw(G3,pos,node_size=15,style='--',node_color='b')

circ = QuantumCircuit(1,1)
circ.h(0)
circ.measure([0],[0])
SimBackend = qiskit.Aer.get_backend('qasm_simulator')

camino = ''
ys = [0.5]
k = 0.5
for i in range(n+1):
    delta = k/2
    Circuito = qiskit.execute(circ,backend=SimBackend,shots=1)
    camino += max(Circuito.result().get_counts(),key=Circuito.result().get_counts().get)
    if i < n:
        dy = delta
        if camino[-1] == '0':dy*=-1
        ys.append(ys[-1]+dy)
    else:
        for nodename in pos:
            if nodename.startswith('1-'):
                position = pos[nodename]
                print(type(position))
                if round(position[0],6) == round(xs[len(ys)-1],6) and round(position[1],6) == round(ys[-1],6):
                    break
        connectingnodes = pd.Series(list(G3[nodename].keys()))
        if camino[-1] == '0': 
            first2 = '2-{}'.format(min(connectingnodes.where(connectingnodes.str.startswith('2-')).dropna().str[2:].astype('int')))
        else:
            first2 = '2-{}'.format(max(connectingnodes.where(connectingnodes.str.startswith('2-')).dropna().str[2:].astype('int')))

        ys.append(pos[first2][1])
        for i in range(5):
            connectingnodes = pd.Series(list(G3[first2].keys()))
            print(connectingnodes.where(connectingnodes.str.startswith('2-')).dropna().str[2:].astype('int'))
            first2 = '2-{}'.format(min(connectingnodes.where(connectingnodes.str.startswith('2-')).dropna().str[2:].astype('int')))

            ys.append(pos[first2][1])
    k = delta
plt.plot(xs[:len(ys)],ys,'r-')
print(camino)
plt.savefig('CaminoAleatorioEnGrafo.pdf',bbox_inches='tight')
plt.show()
