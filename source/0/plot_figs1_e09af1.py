# https://github.com/sysbio-bioinf/QuantumBooleanNetworks/blob/5b53fc0159ef8c0c816dc5b35aa9597843fa7f45/suppl_scripts/plot_FigS1.py
from qiskit import *
from functions import *
from datetime import datetime

#Load dictionaries from pkl files to reproduce the plot as shown in the article
#Running this script required 1.3sec on a MacBook Pro with a 2.3GHz Quad-Core processor and 16 GB of RAM
starttime = datetime.now()

toymodelpath = "./toyBNQuantum.txt"

syncToyCirc = synthesizeFullNetworkUpdateCircuit(toymodelpath, update="synchronous", includeClassicalRegister=True, updateorder=None)
print(syncToyCirc)
asyncToyCirc = synthesizeFullNetworkUpdateCircuit(toymodelpath, update="asynchronous", includeClassicalRegister=True, updateorder=[2,0,1])
print(asyncToyCirc)

#Decompose circuit to have X gates shown in plot:
target_basis = ['x', 'rz', 'h', 'cx', 'ccx', 'mcx']
syncdecomposed = transpile(syncToyCirc,
                       basis_gates=target_basis,
                       optimization_level=2)
syncdecomposed.draw(output="mpl", filename="./FigS1/toymodel_synccirc.pdf")

asyncdecomposed = transpile(asyncToyCirc,
                       basis_gates=target_basis,
                       optimization_level=2)
asyncdecomposed.draw(output="mpl", filename="./FigS1/toymodel_asynccirc.pdf")

print("Runtime = " + str(datetime.now() - starttime))
