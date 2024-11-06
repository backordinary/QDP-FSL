# https://github.com/quantcert/quantcert.github.io/blob/369089e3319ead9947abb3694f3ad64dac255501/Mermin-hypergraph-states/Qiskit/app/mermin_on_qiskit/run.py
from qiskit import *
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit.providers.ibmq.exceptions import IBMQAccountError

def runCircuit(qc, simulation=True, return_count=True, monitor=False, 
                local=True, shots=1024):
  """ Runs the QuantumCircuit `qc` in IBM Quantum Experience.

  :param QuantumCircuit qc: Quantum circuit to be executed
  :param bool simulation: If `True`, the experience runs on a simulator, which 
    substantially faster than on a quantum processor (due to the demand on
    those). Otherwise, runs on one of the quantum processors.
  :param bool return_count: If the circuit contains measures, and return_count
    is set to `True`, then the count of the result will be returned, otherwise,
    the result will be directly returned.
  :param bool monitor: If `True`, a `job_monitor` will be displayed after the
    job is submitted.

  TODO : add local and shots docs
  
  :returns: dict[str:int] or Result -- Depending on return_count, `runCircuit`
    either returns the result (of type Result) of the run or the count of this
    result, which would be the equivalent of calling `result.get_counts()`.
  """
  n = qc.num_qubits
  if local:
    backend = Aer.get_backend('qasm_simulator')
  else:
    backend = least_busy(IBMQ.get_provider(group='open').backends(
      simulator=simulation, filters=lambda x: x.configuration().n_qubits > 4))
  job_exp = execute(qc, backend=backend, shots=shots)
  if monitor:
    print("Backend name : ", backend.configuration().backend_name)
    print("Job ID : ", job_exp.job_id())
    job_monitor(job_exp)
  result = job_exp.result().get_counts(qc) if return_count else job_exp.result()
  return result


def load_IBMQ_account():
  """ Loads the IMBQ account. If it fails a first time, the IBMQ token will be
  prompted and the account loading will be attempted a second time. If it fails
  a second time. Exits by letting the `Error` be raised.

  Raises:
    IBMQAccountCredentialsInvalidFormat: If the default provider stored on
      disk could not be parsed.
    IBMQAccountCredentialsNotFound: If no IBM Quantum Experience credentials
      can be found.
    IBMQAccountMultipleCredentialsFound: If multiple IBM Quantum Experience
      credentials are found.
    IBMQAccountCredentialsInvalidUrl: If invalid IBM Quantum Experience
      credentials are found.
    IBMQProviderError: If the default provider stored on disk could not
      be found.
  """
  try:
    IBMQ.load_account()
  except IBMQAccountError:
    token = str(input("Account loading failure, please enter your IBM Quantum \
Experience token:\n"))
    IBMQ.save_account(token)
    IBMQ.load_account()
