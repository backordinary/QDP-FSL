# https://github.com/milanleonard/SCNC3102/blob/4e935f7b4afcff0732d2b6533c283b444c464179/Code/initial_benchmarks.py
#%% Imports
from pennylane import numpy as np # get pennylane's numpy wrapper
import qiskit
import pennylane as qml
from itertools import combinations, groupby
import matplotlib.pyplot as plt
import random
from pennylane import expval, var

#%%
# -----------------
## COMPUTE TRUE MINIMUM
# SECTION 1: SINGLE VQE PROBLEM - QNG, SGD, ADAM etc. GO THROUGH EACH OF THE GRADIENT BASED UPDATE METHODS
""" SETUP VARIATONAL CIRCUIT: TAKEN FROM https://pennylane.ai/qml/demos/tutorial_quantum_natural_gradient.html"""
dev = qml.device("default.qubit", wires=3, analytic=False, shots=100)
@qml.qnode(dev)
def circuit(params):
    # |psi_0>: state preparation
    qml.RY(np.pi / 4, wires=0)
    qml.RY(np.pi / 3, wires=1)
    qml.RY(np.pi / 7, wires=2)

    # V0(theta0, theta1): Parametrized layer 0
    qml.RZ(params[0], wires=0)
    qml.RZ(params[1], wires=1)

    # W1: non-parametrized gates
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

    # V_1(theta2, theta3): Parametrized layer 1
    qml.RY(params[2], wires=1)
    qml.RX(params[3], wires=2)

    # W2: non-parametrized gates
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

    return qml.expval(qml.PauliY(0)) 
# %% GLOBAL PARAMS FOR EACH TEST
steps = 10
init_params = np.array([0.432, -0.123, 0.543, 0.233])

#%%

GradientDescentCost = [circuit(init_params)]
opt = qml.GradientDescentOptimizer(0.01)
thetagd = init_params
gdparams = [thetagd]
for _ in range(steps):
    thetagd = opt.step(circuit, thetagd)
    gdparams.append(thetagd)
    GradientDescentCost.append(circuit(thetagd))
gdparams = np.array(gdparams)
#%% Quantum natural gradient

Quantum_natural_GD_Cost = [circuit(init_params)]

opt = qml.QNGOptimizer(0.01)

thetaqng = init_params
qngparams = [thetaqng]
for _ in range(steps):
    thetaqng = opt.step(circuit, thetaqng)
    qngparams.append(thetaqng)
    Quantum_natural_GD_Cost.append(circuit(thetaqng))
qngparams = np.array(qngparams)

#%% ROTOSOLVE

Rotosolve_Cost = [circuit(init_params)]

opt = qml.RotosolveOptimizer()

thetart = init_params
for _ in range(steps):
    thetart = opt.step(circuit, thetart)
    Rotosolve_Cost.append(circuit(thetart))

#%% ADAM OPTIMIZER
Adam_Cost = [circuit(init_params)]

opt = qml.AdamOptimizer(0.01)

thetaadam = init_params
for _ in range(steps):
    thetaadam = opt.step(circuit, thetaadam)
    Adam_Cost.append(circuit(thetaadam))



#%% Plotting
plt.plot(np.arange(1,402)*2, GradientDescentCost, label='Standard gradient descent')
plt.plot(np.arange(1,201)*4, Quantum_natural_GD_Cost[:200], label='Quantum Natural Gradient')
plt.plot(np.arange(1,101)*7, Rotosolve_Cost[:100], label='Rotosolve analytic minimum')
plt.plot(np.arange(1,402)*2, Adam_Cost, label="Adam optimiser")
plt.legend()
plt.xlabel("Number of circuit evaluations")
plt.ylabel("Cost function")
plt.title("Optimiser performance considering circuit evaluation requirements")
plt.savefig("Images/descentcomparefair.png")

#%% Running the plot

#%% Let's have a look at the cost landscape
from functools import partial
def wrap_circuit(param1,param2):
    def circuit_wrap(theta0,theta1,theta2,theta3):
        return circuit(np.array([theta0,theta1,theta2,theta3]))
    circuitwrapvec = np.vectorize(partial(circuit_wrap,param1,param2))
    return circuitwrapvec
#circuitwrapvec = wrap_circuit(theta[0],theta[1])
circuitwrapvec = wrap_circuit(thetart[0],thetart[1])
grid_size = 30
X, Y = np.meshgrid(np.linspace(-np.pi,np.pi,grid_size),np.linspace(-np.pi,np.pi,grid_size))
Z = circuitwrapvec(X,Y)

#%%
# Plot this stuff
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z,cmap='viridis', edgecolor='none')
ax.set_title('Cost landscape for two parameters')
ax.set_xlabel("theta_1")
ax.set_ylabel("$theta_2$")
ax.set_zlabel("$\mathcal{L}(\mathbb{\theta})")
plt.show()


# %%
import pickle as pkl
toymodel = {"adam":Adam_Cost, "gd":GradientDescentCost, "qng":Quantum_natural_GD_Cost, \
    "roto":Rotosolve_Cost, "meshgrid":[X,Y,Z]}
with open("./datafiles/toymodel.pkl","wb") as f:
    pkl.dump(toymodel, f)
# %%
