# https://github.com/JaimePSantos/Dissertation-Tex-Code/blob/15544a4334f61e670d1eeee9849fd168c468863d/Coding/Qiskit/CoinedQuantumWalk/cnotDecomp.py
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from qiskit import( ClassicalRegister,
        QuantumRegister,
         QuantumCircuit,
         execute,
         Aer)
from qiskit.visualization import( plot_histogram,
                       plot_state_city)
from math import (log,ceil)
plt.rcParams['figure.figsize'] = 11,8
matplotlib.rcParams.update({'font.size' : 15})
sys.path.append('../Tools')
from IBMTools import(
         simul,
         savefig,
         saveMultipleHist,
         printDict,
         plotMultipleQiskit,
         cnx)

#qreg = QuantumRegister(3)
#creg = ClassicalRegister(3)
#qc = QuantumCircuit(qreg,creg)
#qc = cnx(qc,qreg[0],qreg[1],qreg[2])
#qc.draw(output='mpl')
#plt.show()

theta = np.pi/2
fi = np.pi/2
delta = np.pi/2
ry = np.array([[np.cos(theta/2),-np.sin(theta/2)],[np.sin(theta/2),(np.cos(theta/2))]])
ryDag = np.array([[np.cos(-theta/2),-np.sin(-theta/2)],[np.sin(-theta/2),(np.cos(-theta/2))]])
rz = np.array([[np.exp(1j*fi/2),0],[0,np.exp(-1j*fi/2)]])
rzDag = np.array([[np.exp(1j*(-fi/2)),0],[0,np.exp(-1j*(-fi/2))]])
fiMat = np.array([[np.exp(1j*delta),0],[0,np.exp(1j*delta)]])
x = np.array([[0,1],[1,0]])
gateDecomp = (rz.dot(ry).dot(x).dot(ryDag).dot(x).dot(rzDag)).round(1)
print(gateDecomp)


