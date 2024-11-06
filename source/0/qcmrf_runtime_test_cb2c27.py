# https://github.com/np84/qcmrf/blob/18568942cc60c56fd7794589b36c312e4b48e81a/qcmrf_runtime_test.py
from qiskit import Aer
from qiskit.providers.ibmq.runtime.utils import RuntimeEncoder, RuntimeDecoder
from qiskit.providers.ibmq.runtime import UserMessenger

import json
import qcmrf_runtime as qcmrf

def test():
	backend = Aer.get_backend('qasm_simulator')
	inputs = {
		"graphs": [[[0]]],
		"repetitions": 10,
		"shots": 1024,
		"optimization_level": 3,
		"measurement_error_mitigation": 1,
		"layout": [0,1]
	}
	user_messenger = UserMessenger()
	serialized_inputs = json.dumps(inputs, cls=RuntimeEncoder)
	deserialized_inputs = json.loads(serialized_inputs, cls=RuntimeDecoder)
	qcmrf.main(backend, user_messenger, **deserialized_inputs)

test()
