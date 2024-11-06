# https://github.com/scottoshiro2/Qu-Beats/blob/aeeb45c9fc6331408d6ec211f922d3d04e81847e/Qu-Beats-1-Qu-Beats-v2.0/qubeatsv2.0/main.py
from mido import *
from music21 import *

from qc_config import *
from qb_constr import *
from midi_parser import *

import numpy as np
from qiskit import *
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy

from qiskit.providers.aer.noise import NoiseModel
from qiskit import QuantumCircuit, Aer, assemble, transpile, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram, array_to_latex
# import basic plot tools
from qiskit.visualization import plot_histogram


if __name__ == '__main__':
    # This where the main script looks and parses
    mid_dbl = MidiFile('beat1_input.mid')
    mid_lps = MidiFile('beat2_input.mid')

    event_list_dbl, meta_data_dbl = get_messages( mid_dbl.tracks )
    event_list_lps, meta_data_lps = get_messages( mid_lps.tracks )

    org_dbl = note_grouping( event_list_dbl )
    org_lps = note_grouping( event_list_lps )

    bar_len_dbl = bar_length( meta_data_dbl )
    bar_len_lps = bar_length( meta_data_lps )

    svs_dbl = state_vectors( org_dbl, bar_len_dbl )
    svs_lps = state_vectors( org_lps, bar_len_lps )

    qc = quantum_circ( svs_dbl, svs_lps, 8 )
    print("Your Beats Have Been Entangled....and are now communicating....")
    results = exec_qc( qc, 'ibmq_jakarta')

    print("Now Generating MIDI FILES....")
    for i, counts in enumerate( results ):
        gen_midi( beat_construction(org_dbl, org_lps, counts), i )

    print("Successful Generation!")
    ## End of main.py
