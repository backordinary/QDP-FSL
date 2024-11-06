# https://github.com/P4ay/Qiskit-Pulse/blob/82aa2a54be7abd0e666c0a348b8c7fcda8d8cc07/Calibrated_bell_state.py
from qiskit import pulse
from qiskit.pulse import Acquire, AcquireChannel, MemorySlot, ControlChannel, DriveChannel
from qiskit import IBMQ
from qiskit import schedule, transpile, execute
from qiskit.tools.monitor import job_monitor
from qiskit.circuit import Gate
import numpy as np
import warnings
warnings.filterwarnings('ignore')


dc0 = DriveChannel(0)
dc1 = DriveChannel(1)
cc0 = ControlChannel(0)
cc1 = ControlChannel(1)

IBMQ.save_account('5aa05e965118a4d4c39c15a864c922b90f9beacaa68818452d6d5bdc0a9642e3ebcec552361e81811e90133399c375cdceb726c0eebd8b318a5158259740b2f7')
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend=provider.get_backend('ibmq_belem')

from qiskit.pulse import library

x90amp = 0.1219116667657439+0.005438173268657675j
x90sigma = 40
x90num_samples = 160
x90beta= -2.4909884222108594

h = pulse.library.Drag(x90num_samples, x90amp, x90sigma, x90beta,name="x90_q0")


y90amp=-0.005438173268657659+0.1219116667657439j
y90sigma=40
y90beta=-2.4909884222108594
y90num_samples=160
y90 = pulse.library.Drag(y90num_samples, y90amp, y90sigma, y90beta,name="y90")

x90q1amp = 0.11970339183692824+0.002259897137120287j
x90q1sigma = 40
x90q1num_samples = 160
x90q1beta= 0.6983177108024056
x90q1 = pulse.library.Drag(x90q1num_samples, x90q1amp, x90q1sigma, x90q1beta,name="x90_q1")

cr90d0duration=1584
cr90d0amp=0.02214868159944392+0.004254749791534447j
cr90d0sigma= 64
cr90d0width= 1328
cr90p_d0_u1=pulse.library.GaussianSquare(cr90d0duration, cr90d0amp, cr90d0sigma, cr90d0width)

cr90duration=1584
cr90amp=-0.13733159784825894+0.04751671131246091j
cr90sigma= 64
cr90width= 1328
cr90p_u1=pulse.library.GaussianSquare(cr90duration, cr90amp, cr90sigma, cr90width)

xpq1amp = 0.24560981188987557+0j
xpq1sigma = 40
xpq1num_samples = 160
xpq1beta= 0.6618063862107643
xpq1 = pulse.library.Drag(xpq1num_samples, xpq1amp, xpq1sigma, xpq1beta,name="xp_q1")

cr90md0amp=-0.02214868159944392-0.004254749791534444j
cr90m_d0_u1=pulse.library.GaussianSquare(cr90d0duration, cr90md0amp, cr90d0sigma, cr90d0width)

cr90mamp=0.13733159784825894-0.04751671131246092j
cr90m_u1=pulse.library.GaussianSquare(cr90duration, cr90mamp, cr90sigma, cr90width)

y90mamp=0.002259897137120292-0.11970339183692824j
y90mbeta=0.6983177108024056
y90m = pulse.library.Drag(y90num_samples, y90mamp, y90sigma, y90mbeta,name="y90")

with pulse.build(backend,name='Bell') as bell_gate_pulse:
    pulse.shift_phase(-1.5707963268, dc0)
    pulse.shift_phase(-1.5707963268, cc1)
    pulse.play(h, dc0)
    pulse.shift_phase(-1.5707963268, dc0)
    pulse.shift_phase(-3.141592653589793, dc0)
    pulse.shift_phase(-1.5707963267948966, dc1)
    pulse.shift_phase(-1.5707963267948966, cc0)  
    pulse.shift_phase(-1.5707963268, cc1)
    pulse.shift_phase(-3.141592653589793, cc1)
    pulse.play(y90,dc0)
    pulse.play(x90q1,dc1)
    pulse.play(cr90p_d0_u1,dc0)
    pulse.play(cr90p_u1, cc1)
    pulse.play(xpq1,dc1)
    pulse.play(cr90m_d0_u1,dc0) 
    pulse.play(cr90m_u1,cc1) 
    pulse.shift_phase(-1.5707963267948966, dc0)  
    pulse.shift_phase(-1.5707963267948966, dc1)
    pulse.play(h,dc0)
    pulse.play(y90m,dc1)
#    pulse.acquire(m, pulse.acquire_channel(0), MemorySlot(0))

#print(h_q0.instructions)
bgate=Gate(name='bell_gate', label='BG', num_qubits=2, params=[])

from qiskit import QuantumCircuit
qc = QuantumCircuit(2,2)
qc.append(bgate, [0,1])
qc.measure(0,0)
qc.measure(1,1)

qc.add_calibration('bell_gate', [0,1], bell_gate_pulse)

qc.draw('mpl')

job=execute(qc, backend=backend, shots=2000, optimization_level=0)
from qiskit.tools.monitor import job_monitor
job_monitor(job)
result=job.result()
counts = result.get_counts(qc)
print(counts)

from qiskit.tools.visualization import plot_histogram

plot_histogram(result.get_counts(qc))


