# https://github.com/garymooney/qmuvi/blob/7de4dd80991faf4bbfee09885857189844744dbb/example_music_video.py
import quantum_music
from quantum_music import make_music_video, get_instruments, chromatic_middle_c, get_depolarising_noise
import qiskit
from qiskit import QuantumCircuit

circ = QuantumCircuit(4)
circ.h(0)
circ.barrier()
circ.cx(0, 1)
circ.barrier()
circ.cx(1, 2)
circ.barrier()
circ.cx(2, 3)
circ.barrier()
circ.h(3)
circ.barrier()
circ.cx(3,2)
circ.barrier()
circ.cx(2,1)
circ.barrier()
circ.cx(1,0)
circ.barrier()

rhythm = [(160, 80)]*8

make_music_video(circ, "example_music_video", rhythm, get_depolarising_noise(0.02, 0.05), [[57], get_instruments('tuned_perc')], note_map=chromatic_middle_c, invert_colours=False, fps=8, smooth_transitions=True, probability_distribution_only = False)