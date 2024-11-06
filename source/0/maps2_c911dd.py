# https://github.com/jiyoonm/quantum-touch-map-main/blob/541b0704c8694de438f510391d7cf8e6d7c212a3/base_qiskit/maps2.py
from qiskit import QuantumCircuit, Aer, execute
from math import pi
from qc_tools import plot_height

from random import random
import matplotlib.pyplot as plt
from matplotlib import cm

L = 15
project_dir = project.folder
file_name ='test2.png'
file_path = '{}/{}'.format(project_dir, file_name)

def get_height(x,y):
    
    qc = QuantumCircuit(1,1)
    
    # perform rotations, whose angles depend on x and y
    qc.x(0)
    # low frequency rotations
    qc.rx((1/32)*x*pi,0) 
    qc.ry((1/32)*y*pi,0)
    qc.rx((1/16)*x*pi,0)
    qc.ry((1/16)*y*pi,0)
    qc.rx((1/8)*x*pi,0)
    qc.ry((1/8)*y*pi,0)
    
    # perform a z measurement
    qc.measure(0,0)
    
    # determine the probability of a 1
    try:
        p = execute(qc,Aer.get_backend('qasm_simulator'),shots=10000).result().get_counts()['1']/10000
    except:
        p = 0
    
    # return p^2 as the height
    return p**2

z = [ [get_height(x,y) for x in range(-L,L)] for y in range(-L,L) ]   
    
def plot_z(z,L):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(L/2,L/2)
    cs = ax.contourf(z,25,cmap=cm.get_cmap('gray'))

    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight')

    plt.show()
    op('moviefilein2').par.file =file_path
    op('moviefilein2').par.reloadpulse.pulse()

    
plot_z(z,L)