# https://github.com/ssenge/Qroestl/blob/98e1c17389a7c728c522ec19e97a986e4a485e7a/qroestl/utils/SaveIBMQToken.py
from qiskit import IBMQ

IBMQ.save_account('put-your-key-here', hub='hub', group='group', project='project', overwrite=True)
