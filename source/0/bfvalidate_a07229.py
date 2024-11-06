# https://github.com/M4GNV5/brainfuck-quantum-circuits/blob/892091ceb9d0e83c4c0fced912c47907670e24c9/bfvalidate.py
print("loading libraries...")
import sys, json, numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from qiskit.circuit.library import MCXGate
from qiskit.visualization import plot_histogram, plot_state_city

def generateInput(circuit, qreg):
	circuit.reset(qreg)
	circuit.h(qreg)

canHaveError = None
canHaveErrorI = None
def generateOutput(circuit, qreg):
	global canHaveError, canHaveErrorI
	if canHaveError is None:
		canHaveError = QuantumRegister(2, "err")
		canHaveErrorI = circuit.num_qubits
		circuit.add_register(canHaveError)

	circuit.reset(canHaveError[1])

	# check for >127
	gate = MCXGate(2, ctrl_state="01")
	circuit.append(gate, qargs=[qreg[7]] + canHaveError[:])

	# check for <32
	gate = MCXGate(4, ctrl_state="0000")
	circuit.append(gate, qargs=qreg[5 : 8] + canHaveError[:])

	circuit.cx(canHaveError[1], canHaveError[0])
	#circuit.reset(canHaveError[1])

def generateAddConst(circuit, qreg, value):
	for i in range(7, -1, -1):
		if value & (1 << i):
			for j in range(7, i, -1):
				circuit.mcx(qreg[i : j], qreg[j])
			circuit.x(qreg[i])

def generateSubConst(circuit, qreg, value):
	if value & (1 << 0):
		circuit.x(qreg[0])
		for j in range(0 + 1, 8):
			circuit.mcx(qreg[0 : j], qreg[j])
	if value & (1 << 1):
		circuit.x(qreg[1])
		for j in range(1 + 1, 8):
			circuit.mcx(qreg[1 : j], qreg[j])
	if value & (1 << 2):
		circuit.x(qreg[2])
		for j in range(2 + 1, 8):
			circuit.mcx(qreg[2 : j], qreg[j])
	if value & (1 << 3):
		circuit.x(qreg[3])
		for j in range(3 + 1, 8):
			circuit.mcx(qreg[3 : j], qreg[j])
	if value & (1 << 4):
		circuit.x(qreg[4])
		for j in range(4 + 1, 8):
			circuit.mcx(qreg[4 : j], qreg[j])
	if value & (1 << 5):
		circuit.x(qreg[5])
		for j in range(5 + 1, 8):
			circuit.mcx(qreg[5 : j], qreg[j])
	if value & (1 << 6):
		circuit.x(qreg[6])
		for j in range(6 + 1, 8):
			circuit.mcx(qreg[6 : j], qreg[j])
	if value & (1 << 7):
		circuit.x(qreg[7])
		for j in range(7 + 1, 8):
			circuit.mcx(qreg[7 : j], qreg[j])

def generateAdd(circuit, dest, src):
	for i in range(7, -1, -1):
		for j in range(7, i, -1):
			circuit.mcx([src[i], *dest[i : j]], dest[j])
		circuit.cx(src[i], dest[i])

def generateSetConst(circuit, qreg, value):
	circuit.reset(qreg)
	if value & (1 << 0):
		circuit.x(qreg[0])
	if value & (1 << 1):
		circuit.x(qreg[1])
	if value & (1 << 2):
		circuit.x(qreg[2])
	if value & (1 << 3):
		circuit.x(qreg[3])
	if value & (1 << 4):
		circuit.x(qreg[4])
	if value & (1 << 5):
		circuit.x(qreg[5])
	if value & (1 << 6):
		circuit.x(qreg[6])
	if value & (1 << 7):
		circuit.x(qreg[7])

def generateSet(circuit, dest, src):
	circuit.reset(dest)
	circuit.cx(src[0], dest[0])
	circuit.cx(src[1], dest[1])
	circuit.cx(src[2], dest[2])
	circuit.cx(src[3], dest[3])
	circuit.cx(src[4], dest[4])
	circuit.cx(src[5], dest[5])
	circuit.cx(src[6], dest[6])
	circuit.cx(src[7], dest[7])

def generateLoop(circuit, condQReg, body):
	pass # TODO

def getQreg(circuit, qregs, offset):
	if offset < 0:
		raise Exception("Invalid/Unsupported qreg offset {}".format(offset))

	while len(qregs) <= offset:
		name = "p[{}]".format(len(qregs))
		qreg = QuantumRegister(8, name)
		qregs.append(qreg)
		circuit.add_register(qreg)

	return qregs[offset]

def generateAddExpr(circuit, qregs, qreg, expr):
	if expr["scalar"] < 0:
		generateSubConst(circuit, qreg, -1 * expr["scalar"])
		#circuit.barrier()
	elif expr["scalar"] > 0:
		generateAddConst(circuit, qreg, expr["scalar"])
		#circuit.barrier()

	for summand in expr["summands"]:
		qreg2 = getQreg(circuit, qregs, summand["offset"])
		scale = summand["scale"]
		if scale["denominator"] != 1:
			raise Exception("Unsupported denominator {}".format(scale["denominator"]))
		scale = scale["numerator"]

		for i in range(0, scale):
			generateAdd(circuit, qreg, qreg2)
			#circuit.barrier()

def generateStatement(circuit, qregs, statement):
	typ = statement["type"]
	if typ == "add":
		qreg = getQreg(circuit, qregs, statement["offset"])
		generateAddExpr(circuit, qregs, qreg, statement["value"])
	elif typ == "set":
		qreg = getQreg(circuit, qregs, statement["offset"])
		circuit.reset(qreg)
		generateAddExpr(circuit, qregs, qreg, statement["value"])
	elif typ == "shift":
		raise Exception("Unsupported expression: shift")
	elif typ == "adduntilzero":
		raise Exception("Unsupported expression: adduntilzero")
	elif typ == "loop":
		raise Exception("Unsupported expression: loop")
	elif typ == "input":
		qreg = getQreg(circuit, qregs, statement["offset"])
		generateInput(circuit, qreg)
		#circuit.barrier()
	elif typ == "output":
		value = statement["value"]
		if len(value["summands"]) == 0:
			return # TODO: check scalar for invalid char?
		
		firstScale = value["summands"][0]["scale"]
		if value["scalar"] == 0 and len(value["summands"]) == 1 \
			and firstScale["numerator"] == 1 and firstScale["denominator"] == 1:
			qreg = getQreg(circuit, qregs, value["summands"][0]["offset"])
			generateOutput(circuit, qreg)
		else:
			raise Exception("Unsupported output expression")
	elif typ == "print":
		pass # TODO: check string for invalid char?
	elif typ == "comment":
		pass
	else:
		print("unknown statement type {}".format(typ), file=sys.stderr)

if len(sys.argv) != 2:
	print("Invalid args.", file=sys.stderr)
	exit(1)

with open(sys.argv[1], "r") as fd:
	program = json.load(fd)

print("building circuit...")
circuit = QuantumCircuit(name=sys.argv[1])
qregs = []
for stmt in program:
	generateStatement(circuit, qregs, stmt)

'''
figure = circuit.draw('mpl')
#plt.show()
plt.gca().set_position([0, 0, 1, 1])
plt.savefig("circuit.svg")
exit(0)
'''

successCount = 0
errorCount = 0
if canHaveErrorI is not None:
	print("running simulator...")
	backend = Aer.get_backend('statevector_simulator')
	job = execute(circuit, backend)
	output = job.result().get_counts(circuit)

	print("determining failure rate...")
	errIndex = circuit.num_qubits - canHaveErrorI - 1
	for key in output:
		if key[errIndex] == "0":
			successCount += output[key]
		else:
			errorCount += output[key]
else:
	successCount = 1.0

print("Can fail? {} ({}%)".format(errorCount != 0, int(errorCount * 1000) / 10))
#plot_histogram({"0": successCount, "1": errorCount})
#plt.show()