# https://github.com/jiyoonm/quantum-touch-map-main/blob/541b0704c8694de438f510391d7cf8e6d7c212a3/base_qiskit/maps3.py
from qiskit import QuantumCircuit, Aer, execute
from math import pi
from qc_tools import plot_height

from random import random
import matplotlib.pyplot as plt
from matplotlib import cm

L = 15
project_dir = project.folder
file_name ='test3.png'
file_path = '{}/{}'.format(project_dir, file_name)

def get_height(x,y,seed):
    
    qc = QuantumCircuit(1,1)
    
    # get seed positions for this distance
    d = max(abs(x),abs(y))
    (xs,ys) = seed[d]
    
    # perform rotations, whose angles depend on x and y
    qc.x(0)
    # low frequency rotations to create island shape
   

    qc.rx((1/32)*((x+xs)/2)*pi,0) 
    qc.ry((1/32)*((y+ys)/2)*pi,0)
    qc.rx((1/16)*x*pi,0)
    qc.ry((1/16)*y*pi,0)
    
    # perform a z measurement
    qc.measure(0,0)
    
    # determine the probability of a 1
    try:
        p = execute(qc,Aer.get_backend('qasm_simulator'),shots=10000).result().get_counts()['1']/1000
    except:
        p = 0
    
    # return p^2 as the height
    return p**2
def generate_seed(L):
    seed = [(0,0)]
    x,y = 0,0
    d = 0
    while d<L:
        if random()<0.5:
            x += 1
        else:
            y += 1
        d = max(abs(x),abs(y))
        if d>=len(seed):
            seed += [(x,y)]*(d+1-len(seed))    
    return seed
seed = generate_seed(L)

seed = generate_seed(L)
z = [ [get_height(x,y,seed) for x in range(-L,L)] for y in range(-L,L) ]   


    
def plot_z(z,L):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(L/2,L/2)
    cs = ax.contourf(z,25,cmap=cm.get_cmap('gray'))

    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight')

    plt.show()
    op('moviefilein3').par.file =file_path
    op('moviefilein3').par.reloadpulse.pulse()

    
plot_z(z,L)