# https://github.com/FieryRaccoon23/QApple/blob/fbe9b1ac1d73b35bd0a47184d6badf1db10109fa/Config.py
import qiskit
from qiskit import IBMQ

QSystem1Bit = "ibmq_armonk"
QSystem5Bits = "ibmq_belem" #"ibmq_santiago" #"ibmq_manila" #"ibmq_bogota" #"ibmq_quito" #"ibmq_lima"
QSim5000Bits = "simulator_stabilizer"
QSim100Bits = "simulator_mps"
QSim63Bits = "simulator_extended_stabilizer"
QSim32Bits = "ibmq_qasm_simulator" #"simulator_statevector"

singleColorBits = 8
alphaBits = 8
redBits = 8
greenBits = 8
blueBits = 8
rgbBits = redBits + greenBits + blueBits
argbBits = rgbBits + alphaBits

def GetQBitMachine(bits):
    if bits == 1:
        return QSystem1Bit
    elif bits == 5:
        return QSystem5Bits
    elif bits == 5000:
        return QSim5000Bits
    elif bits == 100:
        return QSim100Bits
    elif bits == 63:
        return QSim63Bits
    elif bits == 32:
        return QSim32Bits
    else:
        return ""

def QiskitVersion():
    print(qiskit.__qiskit_version__)

def SetupQiskit():
    IBMQ.save_account('053179eac16528b705aea4ee05bc6d7bab719a16ee027eabe971488ef01ef269724564acf4ea838815ad989345339cae8365f4080068b047afeb2051046d0bf5')
    IBMQ.load_account()