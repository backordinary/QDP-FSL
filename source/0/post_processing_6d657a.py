# https://github.com/TobiasRohner/ApproximateReversibleCircuits/blob/0ea8270af169da804593a357954829c89fc0b9e8/post_processing.py
#!/usr/bin/env python3

import sys
import os
import matplotlib.pyplot as plt
import qiskit
import qc_properties
import functions
import copy
import tqdm
import numpy as np
from load_circuits import *
from errors import *
import dill


FNAME = sys.argv[1]
NOISE_FACTOR = 10
func = getattr(functions, 'Func'+FNAME)()


known_circuits = []
parsed = []

for fname in filter(lambda s:  s.startswith(FNAME) and s.endswith('.txt'), os.listdir('.')):
    data = None
    with open(fname, 'r') as f:
        data = [line.strip() for line in f.readlines() if line.strip() != '']
    i = 0
    while i < len(data):
        num_lines, num_gates = map(int, data[i].split())
        parsed.append({})
        parsed[-1]['circuit'] = data[i]
        i += 1
        for k in range(num_gates):
            parsed[-1]['circuit'] += '\n' + data[i+k]
        i += k + 1
        l, d, e, fn, fp, qc = map(float, data[i].split())
        parsed[-1]['l'] = l
        parsed[-1]['d'] = d
        parsed[-1]['e'] = e
        parsed[-1]['fn'] = fn
        parsed[-1]['fp'] = fp
        parsed[-1]['qc'] = qc
        i += 1
for fname in os.listdir('known_circuits/'+FNAME):
    with open('known_circuits/'+FNAME+'/'+fname, 'r') as f:
        source = f.read()
        qc, circuit = tfc2qiskit(source)
        known_circuits.append({'qc':qc, 'circuit':circuit})

# Sort the circuits in increasing order of quantum cost
parsed = sorted(parsed, key = lambda t: t['qc'])

for p in tqdm.tqdm(parsed, desc='Optimized'):
    circuit = ascii2qiskit(p['circuit'], func.output_size)
    e, fn, fp = compute_error_rates(circuit, func, reduce_noise(qc_properties.noise_model, NOISE_FACTOR))
    p['e_noise'] = e
    p['fn_noise'] = fn
    p['fp_noise'] = fp
    e, fn, fp = compute_error_rates(circuit, func, qc_properties.noise_model)
    p['e_noise_melbourne'] = e
    p['fn_noise_melbourne'] = fn
    p['fp_noise_melbourne'] = fp
for known in tqdm.tqdm(known_circuits, desc='Known'):
    e, fn, fp = compute_error_rates(known['circuit'], func, reduce_noise(qc_properties.noise_model, NOISE_FACTOR))
    known['e_noise'] = e
    known['fn_noise'] = fn
    known['fp_noise'] = fp
    e, fn, fp = compute_error_rates(known['circuit'], func, qc_properties.noise_model)
    known['e_noise_melbourne'] = e
    known['fn_noise_melbourne'] = fn
    known['fp_noise_melbourne'] = fp


# Store the data to file
data = {'known_circuits':known_circuits, 'parsed':parsed}
with open(FNAME+'.dat', 'wb') as f:
    dill.dump(data, f)
