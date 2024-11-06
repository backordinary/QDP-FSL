# https://github.com/Abdullah-R/Q-hackers/blob/afca6bd23ed77a72d615d404bb01b5621d57307a/animate.py
#%%
# some libraries that are useful
from celluloid import Camera
from matplotlib import pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram, plot_bloch_vector
from math import sqrt, pi

# returns array of people who are infected given data from random simulation
def infected(peeps):
    sz = int(np.sum(peeps[4,:]))
    inf = np.zeros([5,sz])
    ind = np.where(peeps[4,:]==1)
    for i in range(sz):
        inf[:,i] = peeps[:,ind[0][i]]
    return inf

# runs a quantum circuit and randomly returns infection value based on probability of infection
def qc_check(p):
    qc = QuantumCircuit(1,1)
    theta = 2*np.arcsin(sqrt(p))
    qc.rx(theta,0)
    qc.measure(0,0)
    qc.draw('mpl')

    counts = execute(qc,Aer.get_backend('qasm_simulator'), shots=1).result().get_counts()

    return(int(list(counts.keys())[0]))
# %%
# number of people simulated in the random experiemnt
pn=150

#Plot and camera setup
fig = plt.figure()
plt.title("Simulated Spread of COVID-19 Via Random Movement")
plt.xlabel("x-coordinate location")
plt.ylabel("y-coordinate location")
camera = Camera(fig)

# rows 1-2 of loc represent positions of people, rows 3-4 represent current velocities, 
# row 5 is whether infected
loc = (np.random.rand(5, pn) - 1/2)
loc[4,:] = 0
# patient zero! 
loc[4, 6] = 1
loc[[2,3],:] = loc[[2,3],:]/100
for i in range(50):

    inf = infected(loc)

    for j in range(pn):
        #plotting
        col = 'black'
        if loc[4,j] == 1: col = 'red'
        plt.plot(loc[0,j], loc[1,j], '.' ,color = col) # new line added here

        #check for infection
        minDist = 10
        if loc[4,j] == 0:
            for k in range(inf[1,].size):
                dist = np.linalg.norm(loc[[0,1],j]-inf[[0,1],k])
                if dist < minDist: minDist = dist
            
            # returns random infection value (0 = not infected, 1 = infected) 
            # if close enough to someone infected
            if minDist < 0.05:
                loc[4,j] = qc_check(0.1)

        #checking velocities
        if np.abs(loc[0,j]) > 0.5:
            loc[2,j] = -loc[2,j]
        if np.abs(loc[1,j]) > 0.5:
            loc[3,j] = -loc[3,j]


    camera.snap()
    mov = (np.random.rand(2, pn) - 1/2)/150
    loc[[2,3],:] = loc[[2,3],:] + mov
    loc[[0,1],:] = loc[[2,3],:] + loc[[0,1],:]

animation = camera.animate()
animation.save('my_animation.gif')