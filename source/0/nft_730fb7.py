# https://github.com/MartinTM/Qunten-Computer-NFT/blob/61d1451856928c5c81aaee45b957a55121ddc965/NFT.py
from qiskit import *
from qiskit.tools.monitor import job_monitor
import turtle
from turtle import *
import math

#IBMQ.save_account('your token here')

qr=QuantumRegister(5)
cr=ClassicalRegister(5)
circuit=QuantumCircuit(qr,cr)

circuit.h(qr[0])
circuit.h(qr[2])
circuit.h(qr[3])
circuit.h(qr[4])

circuit.cx(qr[0],qr[1])

circuit.measure(qr,cr)

IBMQ.load_account()
provider = IBMQ.get_provider('ibm-q')
qcomp = provider.get_backend('ibmq_lima')

job = execute(circuit,backend=qcomp,shots=1, memory=True)
job_monitor(job)
result=job.result()

bin=result.get_memory()[0]

print(bin)
dec=int(bin, base=2)
print(dec)

nft=turtle.Turtle()
nft.color("blue")
nft.speed(10)
title("Quantum NFT")

for i in range((dec+1)*30):
    nft.forward(math.sqrt(i)*20)
    nft.left(dec*10)
    nft.right(dec/2)

turtle.done()