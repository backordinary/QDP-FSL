# https://github.com/seunomonije/quantum/blob/5c82e466f39880bd38dcedb45e48866883da36e9/python/qiskit/main.py
# importing entire package might take a while and may be unnecessary.
# Potential speedup here.
import qiskit

## Basics
from qiskit import QuantumCircuit, execute, QuantumRegister, ClassicalRegister, IBMQ

## IBMQ transpilation
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager

## Repetition codes
from qiskit.ignis.verification.topological_codes import RepetitionCode, lookuptable_decoding, GraphDecoder

## Relaxation/Decoherence
from qiskit.providers.aer.noise.errors.standard_errors import thermal_relaxation_error
from qiskit.ignis.characterization.coherence import T1Fitter, T2StarFitter, T2Fitter
from qiskit.ignis.characterization.coherence import t1_circuits, t2_circuits, t2star_circuits
from qiskit.ignis.characterization.gates import AngleCalFitter, AmpCalFitter
from qiskit.ignis.characterization.gates import ampcal_1Q_circuits, anglecal_1Q_circuits, ampcal_cx_circuits, anglecal_cx_circuits

## My implementations
from functions import * # wildcard import is OK in this setting, we will need all the functions
from exceptions import InvalidDeviceException
from classes import IBMQHandler, SystemInformationGatherer

## Python specifics
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
  # Retrieve API key and use it to access IBMQ
  API_KEY = None
  for line in sys.stdin:
    API_KEY = line.rstrip() # .rstrip() removes all whitespace

  # Start my IBMQ session
  session = IBMQHandler(API_KEY)

  # Select which device you want to use and activate it for this session
  session.select_device('ibmq_athens')
  backend = session.use_real_device()

  # Instantiate information gathering and provide required dependencies
  infoGatherer = SystemInformationGatherer(backend)
  infoGatherer.set_test_qubits([0, 2])
  infoGatherer.set_gate_time(0.1)
  infoGatherer.set_number_of_gates(np.linspace(10, 300, 50))

  # Tell the information gathering what information you want before send to IBM
  infoGatherer.request_T1_data()
  infoGatherer.request_T2_data()

  # Send the info to IBM and fit the data to your needs
  result = infoGatherer.fit_given_circuits()
 
  # Handling the plotting on your own end
  print(result["T1"].time())
  print(result["T1"].time_err())
  print(result["T1"].params)
  print(result["T1"].params_err)

  plt.figure(figsize=(15,6))
  ax = plt.subplot(1, 2, 0+1)
  result["T1"].plot(0, ax=ax)
  ax = plt.subplot(1, 2, 1+1)
  result["T1"].plot(1, ax=ax)
  plt.show()
  return

if __name__ == "__main__":
  main()
