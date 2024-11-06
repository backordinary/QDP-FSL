# https://github.com/Advanced-Research-Centre/QKSA/blob/e6b0452af635dc85e5544f63f1678702cc3ab3fe/legacy_vers/v08/environment.py
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/guessing_game.py
'''
import gym
env = gym.make('GuessingGame-v0')
env.reset()
res = env.step(1000)
print(res)
env.close()
'''

# import openql
from qiskit import QuantumCircuit, qasm, Aer, execute

class environment:

	cQASM = ""		# Filename of OpenQL compiled cQASM that determines environmental dynamics
	OpenQASM = ""	# Filename of OpenQL compiled cQASM that determines environmental dynamics
	allZ = True
	basis = []
	simulator = ""

	def __init__(self, dynamics):

		# self.cQASM = dynamics
		self.OpenQASM = dynamics
		self.simulator = Aer.get_backend('qasm_simulator')

	def setBasis(self, basis):

		self.basis = basis
		self.allZ = False

	def measure(self, neighbours):
		
		circ = QuantumCircuit.from_qasm_file(self.OpenQASM)
		if (not self.allZ):
			print("Add prerotations based on basis (same size as neighbours, else throw error)")
		for n in neighbours:
			circ.measure(n,n)
		result = execute(circ, self.simulator, shots=1, memory=True).result()
		memory = result.get_memory(circ)
		return memory 