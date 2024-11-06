# https://github.com/CosmicDNA/chsh-inequality/blob/e859ba77a220e13d90759e1b0c6c792d0ac5d883/src/save_account.py
import qiskit
from qiskit import IBMQ
import os

IBMQ.save_account(os.environ['TOKEN'])