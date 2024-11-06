# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/laurel_or_yanny.py
import qiskit
from pydub import AudioSegment
from pydub.playback import play
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.tools.visualization import plot_histogram

qr = QuantumRegister(1)
cr = ClassicalRegister(1)
qc = QuantumCircuit(qr, cr)

qc.h(qr[0])
qc.measure(qr[0], cr[0])

qiskit.IBMQ.load_accounts()
backend = qiskit.backends.ibmq.least_busy(qiskit.IBMQ.backends(simulator=True))
print("We'll use the least busy device:", backend.name())
job = qiskit.execute(qc, backend, shots=100)
stats = job.result().get_counts()
print(stats)

#Select '0' to represent 'laurel'
if '0' not in stats.keys():
    stats['laurel'] = 0
else:
    stats['laurel'] = stats.pop('0')

#Which leaves '1' to represent 'yanny'
if '1' not in stats.keys():
    stats['yanny'] = 0
else:
    stats['yanny'] = stats.pop('1')
    

plot_histogram(stats)


#Import two tracks
laurel = AudioSegment.from_wav('wav/laurel.wav')
yanny = AudioSegment.from_wav('wav/yanny.wav')
play(laurel)
play(yanny)

mixed = laurel.overlay(yanny)
play(mixed)