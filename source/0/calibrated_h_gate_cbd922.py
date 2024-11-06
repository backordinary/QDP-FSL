# https://github.com/P4ay/Qiskit-Pulse/blob/82aa2a54be7abd0e666c0a348b8c7fcda8d8cc07/Calibrated_H_gate.py
from qiskit import pulse
from qiskit.pulse import Acquire, AcquireChannel, MemorySlot
from qiskit import IBMQ
from qiskit import schedule, transpile, execute
from qiskit.tools.monitor import job_monitor

import numpy as np
with pulse.build(name='my_example') as my_program:
    # Add instructions here
    pass

my_program

from qiskit.pulse import DriveChannel

channel = DriveChannel(0)

IBMQ.save_account('5aa05e965118a4d4c39c15a864c922b90f9beacaa68818452d6d5bdc0a9642e3ebcec552361e81811e90133399c375cdceb726c0eebd8b318a5158259740b2f7')
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend=provider.get_backend('ibmq_armonk')

from qiskit.pulse import library

amp = 0.43573525422528864-0.1091321484397982j
sigma = 80
num_samples = 320


'''with pulse.build(backend=backend, name='backend_aware') as backend_aware_program:
    channel = pulse.drive_channel(0)
    print(pulse.num_qubits())
    # Raises an error as backend only has 5 qubits
    #pulse.drive_channel(100)

with pulse.build(backend) as delay_5dt:
    pulse.delay(5, channel)
    
with pulse.build() as sched:
    pulse.play(pulse, channel)'''
    
h = pulse.library.Drag(num_samples, amp, sigma, -1.4459614505662715,name="H")
m=pulse.library.GaussianSquare(22400, 0.3051214347689275+0.1714669357180885j, 64, 22144)

with pulse.build(backend,name='Hadamard') as h_q0:
    pulse.shift_phase(-1.5707963268, channel)
    pulse.play(h, channel)
    pulse.shift_phase(-1.5707963268, channel)
#    pulse.acquire(m, pulse.acquire_channel(0), MemorySlot(0))

#print(h_q0.instructions)

from qiskit import QuantumCircuit
qc = QuantumCircuit(1,1)
qc.h(0)
qc.measure(0, 0)
    
qc.add_calibration('h', [0], h_q0)
qc.draw('mpl')

job=execute(qc, backend=backend, shots=2000, optimization_level=0)
from qiskit.tools.monitor import job_monitor
job_monitor(job)
result=job.result()
counts = result.get_counts(qc)
print(counts)

from qiskit.tools.visualization import plot_histogram

plot_histogram(result.get_counts(qc))