# https://github.com/seunomonije/quantum/blob/5c82e466f39880bd38dcedb45e48866883da36e9/python/qiskit/classes.py
import pickle
import threading

from qiskit import IBMQ, Aer, execute
from exceptions import InvalidDeviceException, DuplicateNameException, InvalidTupleTypeException

from qiskit.ignis.characterization.coherence import t1_circuits, t2_circuits
from qiskit.ignis.characterization.coherence import T1Fitter, T2StarFitter, T2Fitter
from qiskit.ignis.characterization.gates import AngleCalFitter, AmpCalFitter

class IBMQHandler:
  """
    NOTE: Make sure to install python SSL cerfiicates or you will not be able to access IBMQ
  """
  def __init__(self, API_KEY):
    # Connect to the IBMQ Experience to allow use of real quantum computers
    IBMQ.enable_account(API_KEY)

    # Initialize simulator for use as well
    self.simulatorBackend = Aer.get_backend('qasm_simulator')

  """
    Traverses through all providers associated with the current IMBQ account
    and selects the backend object associated with a provided device name.

    PARAMS:
      deviceName : String -> name of the IBMQ backend object
  """
  def select_device(self, deviceName):
    # Searches all providers, in most cases there will only be one
    backend = None
    for provider in IBMQ.providers():
      for potential_backend in provider.backends():
        if potential_backend.name() == deviceName:
          backend = potential_backend

    if backend:
      self.currentDeviceName = deviceName
      self.currentBackend = backend
    else:
      raise InvalidDeviceException("Given device name was unable to be accessed, or is wrong.")
  
  """
    Returns the IBMQ backend object for whenever needed.
  """
  def use_real_device(self):
    return self.currentBackend

  """
    Returns the simulator backend object for whenever needed.
  """
  def use_simulator(self):
    return self.simulatorBackend

  """
    Writes output results to file
  """
  def write_results_to_file(self, results):
    with open(f'results/raw_results_{self.currentDeviceName}.txt', 'w') as file:
      file.write(str(results))

"""
  Gathers various decomposition data from a specified backend
"""
class SystemInformationGatherer:
  def __init__(self, backend):
    self.backend = backend
    self.circuitsWithData = {}
    self.resultDict = {}

  """
    Provides a qubit array. These qubits will be ran through the circuit, 
    and information about throughout the circuit will be gathered.
  """
  def set_test_qubits(self, test_qubits):
    self.test_qubits = test_qubits

  def set_gate_time(self, gateTime):
    self.gateTime = gateTime

  """
    Sets the number of gates using np.linspace()
    Documentation for np.linspace can be found here:
      https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
  """
  def set_number_of_gates(self, numberOfGates):
    self.numberOfGates = numberOfGates.astype(int)

  def request_T1_data(self):
    self._add_circuit_data_tuple("T1", t1_circuits(self.numberOfGates, self.gateTime, self.test_qubits))
  
  def request_T2_data(self):
    self._add_circuit_data_tuple("T2", t2_circuits(self.numberOfGates, self.gateTime, self.test_qubits))
  
  def request_AmpCal_data(self):
    pass

  def request_AngleCal_data(self):
    pass
  
  """
    Adds an entry to the circuitsToRun dict.
    Possible entries for tupleType:
    "T1" -> signifies a T1 circuit entry
    "T2" -> signifies a T2 circuit entry
    "AmpCal" -> signifies single-qubit amplitude error circuits
    "AngleCal" -> signifies single-qubit amplitude error circuits (angle)
    NOTE: Currently only supports one of each type. Change to come later
    NOTE: fit_p0 value 25, fit_bounds value 40 are hard-coded for current use. Change to come later
  """
  def _add_circuit_data_tuple(self, tupleType, givenTuple):
    possibleType = ["T1", "T2", "AmpCal", "AngleCal"]

    if tupleType in self.circuitsWithData.keys():
      raise DuplicateNameException("The name you gave already exists in the dictionary. Enter another.")
    
    if tupleType not in possibleType:
      raise InvalidTupleTypeException("Invalid tuple type provided. Please see the documentation for valid inputs.")
    
    self.circuitsWithData[tupleType] = givenTuple

  """
    Retrieves backend results from the backend, fits them, and sorts into appropriate column
  """
  def _retrieve_and_sort_backend_result(self, key):
    # circuitTuple[0] is circuit, circuitTuple[1] is xdata
    circuitTuple = self.circuitsWithData[key]

    result = execute(circuitTuple[0], self.backend, optimization_level=0).result()
    if key == "T1":
      self.resultDict["T1"] = T1Fitter(result, circuitTuple[1], self.test_qubits, fit_p0=[1, 25, 0], fit_bounds=([0, 0, -1], [2, 40, 1]))
    elif key == "T2":
      self.resultDict["T2"] = T2Fitter(result, circuitTuple[1], self.test_qubits, fit_p0=[1, 25, 0], fit_bounds=([0, 0, -1], [2, 40, 1]))
    elif key == "AmpCal":
      self.resultDict["AmpCal"] = AmpCalFitter(result, circuitTuple[1], self.test_qubits, fit_p0=[1, 25, 0], fit_bounds=([0, 0, -1], [2, 40, 1]))
    elif key == "AngleCal":
      self.resultDict["AngleCal"] = AngleCalFitter(result, circuitTuple[1], self.test_qubits, fit_p0=[1, 25, 0], fit_bounds=([0, 0, -1], [2, 40, 1]))

  """
    Creates corresponding fits for circuit results given.
    Returns dictionary of fits.
  """
  def fit_given_circuits(self):
    threads = []

    for key in self.circuitsWithData:
      thread = threading.Thread(target=self._retrieve_and_sort_backend_result, args=[key])
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

    # Writes to an output file to save result data.
    with open(f'results/SystemInfoDump.txt', 'wb') as file:
      pickle.dump(self.resultDict, file)

    return self.resultDict
