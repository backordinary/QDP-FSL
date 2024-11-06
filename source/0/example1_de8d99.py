# https://github.com/MartianSheep/quantum-ATG/blob/d0c7a9c6b4d09edf931356b7b334e0fc0d544cf9/examples/example1.py
import numpy as np
import qiskit.circuit.library as qGate

from qatg import QATG
from qatg import QATGFault

class myUFault(QATGFault):
	def __init__(self, params):
		super(myUFault, self).__init__(qGate.UGate, 0, f"gateType: U, qubits: 0, params: {params}")
		self.params = params
	def createOriginalGate(self):
		return qGate.UGate(*self.params)
	def createFaultyGate(self, faultfreeGate):
		return qGate.UGate(faultfreeGate.params[0] - 0.1*np.pi, faultfreeGate.params[1], faultfreeGate.params[2]) # bias fault on theta

generator = QATG(circuitSize = 1, basisGateSet = [qGate.UGate], circuitInitializedStates = {1: [1, 0]}, minRequiredEffectSize = 2)
configurationList = generator.createTestConfiguration([myUFault([np.pi, np.pi, np.pi])])

for configuration in configurationList:
    print(configuration)
    configuration.circuit.draw('mpl')
input()
