# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/backends.py
import pickle
from datetime import datetime

import qiskit
from qiskit import IBMQ
from qiskit.providers.models import BackendStatus

qiskit.IBMQ.load_account()
# backends = Aer.backends()
my_provider = IBMQ.get_provider()
backends = my_provider.backends()

backends_status = []
for b in backends:
    if b.configuration().simulator:
        continue
    print(b.name() + " status: ", end='')
    print(b.status())
    st: BackendStatus = b.status()
    backends_status.append({'pending_jobs': st.pending_jobs,
                            'name': st.backend_name,
                            'operational': st.operational,
                            'qubits': b.configuration().n_qubits})

print("Dump to file")
data = {'tms': datetime.utcnow(), 'backends': backends_status}
f = open('ibm_backends.pickle', 'wb')
pickle.dump(data, f)
f.close()

print("Load form file")
f = open('ibm_backends.pickle', 'rb')
dr = pickle.load(f)
print(dr['tms'])
print(dr['backends'])
