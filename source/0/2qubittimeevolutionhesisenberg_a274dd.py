# https://github.com/Kedree/UMF-QISKIT-EXAMPLES/blob/4476d64823c5436ce31f483470510cbc33a3bb6a/CustomPrograms/2QubitTimeEvolutionHesisenberg.py
from qiskit import *
from qiskit.opflow import X,Y,Z,I,One,Zero
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import *
from qiskit.opflow import *
import time

H = ((X^X)+(Y^Y)+(Z^Z))

def ex(H, t):
    return (H * t).exp_i()

ts = np.linspace(0, np.pi, 100)

df = pd.DataFrame()
tempdf = pd.DataFrame()

states = [Zero^Zero,Zero^One,One^Zero,One^One]
titles = ['00','01','10','11']

for stateNum in range(len(states)):
    state = states[stateNum]
    probs = [np.abs((~state @ ex(H, t) @ (Zero^One)).eval())**2 for t in ts] # Change this
    title = titles[stateNum]
    
    tempdf['data'] = probs
    tempdf['col'] = title[1:] # And these
    tempdf['row'] = title[:1] # 
    tempdf['state'] = title
    
    df = df.append(tempdf)

grid = sns.FacetGrid(df, col='col', row='row', hue='state', palette='hls', height=3, sharex=True, sharey=True, aspect=1, legend_out=True)
grid.map(pyplot.plot, 'data')

for ax,title in zip(grid.axes.flatten(),titles):
    ax.set_title("<"+title+"| e^(-iHt) |01>") # And this

grid.set_axis_labels('Time', 'Probability')
grid.add_legend()

pyplot.show(block=False)

grid.savefig('States.jpg')

