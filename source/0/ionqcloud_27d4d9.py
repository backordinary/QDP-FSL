# https://github.com/sysbio-bioinf/QuantumBooleanNetworks/blob/5b53fc0159ef8c0c816dc5b35aa9597843fa7f45/main_scripts/Fig4/AWS/ionqCloud.py
from functions import *
import braket.circuits
from qiskit import *
from datetime import datetime
import boto3
from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import Circuit
from functions import ionqDict2qiskitNotation
import re #regex for parsing circuit to ionq

#
#NOTE THAT RUNNING THIS SCRIPT REQUIRES AN AMAZON BRAKET ACCOUNT
#THE IONQ QPU IS NOT AVAILABLE AT ALL TIMES
#RUNNING THIS SCRIPT WILL GENERATE COSTS OF $0.30 PER TASK + $0.01 PER SHOT

GiacoPath = "Giacomantonio2010.txt"
GiacoCirc = synthesizeFullNetworkUpdateCircuit(GiacoPath, update="synchronous")
n=5
InitCircuit = QuantumCircuit(2*n, n)
InitCircuit.h(0)
InitCircuit.h(1)
InitCircuit.h(2)
InitCircuit.h(3)
InitCircuit.h(4)
GiacoCirc = InitCircuit.compose(GiacoCirc, list(range(0, 2 * n)), list(range(0, n)))

IonQCircuit = QiskitQASM2IonQCompiler(GiacoCirc, gatenameMapping=gatenameMapping_Qiskit_IonQ)

#TODO: Specify your account specific data here
my_bucket = "amazon-braket-YOUR-BUCKET-HERE"
my_prefix = "YOUR-FOLDER-NAME-HERE"
s3_folder = (my_bucket, my_prefix)
device = AwsDevice("arn:aws:braket:::device/qpu/ionq/ionQdevice")

print("Circuit depth:")
print(IonQCircuit.depth) #Depth of 140 for single Giacomantonio transition circuit

nrshots = 1000 #How many shots to perform, THIS DETERMINES THE COST OF THE TASK

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
task = device.run(IonQCircuit, s3_folder, shots=nrshots, poll_timeout_seconds=24*60*60)  #1 day polling time
resultCounts = task.result().measurement_counts
print(resultCounts)
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

#IonQ measures all 11 qubits -> Parse output back into same format as was used in Qiskit simulations
qubitindices = [5,6,7,8,9] # Which qubits carry the outputs at t=1 for nodes in order, 0-indexed
resultCounts_qiskitFormat = ionqDict2qiskitNotation(resultCounts, qubitindices=qubitindices, invertEndian=True)
print(resultCounts_qiskitFormat)
