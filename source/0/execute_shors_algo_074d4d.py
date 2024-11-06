# https://github.com/pkaran57/quantum-computing-final-project/blob/343c7f2c43efe4adcc584dd232914f46579fec6c/src/execute_shors_algo.py
import logging
from qiskit import Aer

from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Shor

logging.basicConfig(format="'%(asctime)s' %(name)s %(message)s'", level=logging.INFO)
logger = logging.getLogger('main')

# provider = IBMQ.load_account()
# device = provider.get_backend('ibmq_qasm_simulator')
device = Aer.get_backend('qasm_simulator')

number_to_factorize = 15

shor = Shor(number_to_factorize)

quantum_instance = QuantumInstance(device)
result = shor.run(quantum_instance)

print("The list of factors of {} as computed by Shor is {}.".format(number_to_factorize, result['factors'][0]))
