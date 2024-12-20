# https://github.com/garymooney/qmuvi/blob/996a2da67ba526621a8b0c357a12ffc4a0f1a706/grover_Fmin7.py
import quantum_music
from quantum_music import make_music_video, get_instruments
import qiskit
from qiskit import QuantumCircuit


from qiskit.circuit.library import MCMT
circ = QuantumCircuit(4)

# Equal superposition
circ.h(0)
circ.h(1)
circ.h(2)
circ.h(3)
circ.barrier()

# Fmin7 Oracle
circ.x(0)
circ.x(1)
circ.x(2)
circ.compose(MCMT('z',3,1),inplace=True)
circ.barrier()
circ.x(2)
circ.barrier()
circ.compose(MCMT('z',3,1),inplace=True)
circ.barrier()
circ.x(0)
circ.x(3)
circ.barrier()
circ.compose(MCMT('z',3,1),inplace=True)
circ.barrier()
circ.x(1)
circ.x(2)
circ.barrier()
circ.compose(MCMT('z',3,1),inplace=True)
circ.barrier()
circ.x(2)
circ.x(3)
circ.barrier()

# Inversion
circ.h(0)
circ.h(1)
circ.h(2)
circ.h(3)
circ.barrier()
circ.x(0)
circ.x(1)
circ.x(2)
circ.x(3)
circ.barrier()
circ.compose(MCMT('z',3,1),inplace=True)
circ.barrier()
circ.x(0)
circ.x(1)
circ.x(2)
circ.x(3)
circ.barrier()
circ.h(0)
circ.h(1)
circ.h(2)
circ.h(3)
circ.barrier()

time_list = [[60,0]]*8+[[960,0]]+[[240,0]]*4+[[1920,0]]
make_music_video(circ, "grover_Fmin_seven", time_list, None, [get_instruments("windband")], invert_colours=False, fps=60, smooth_transitions=True)