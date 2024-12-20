# https://github.com/JQub/quantum_seismic_detection/blob/c560289dc904c0b018e9349006ccfc36fb082fd5/generate_pulse_encoding_state.py
import numpy as np
import qiskit
from qiskit import pulse, transpile
from qiskit.test.mock import FakeValencia,FakeArmonk,FakeLagos,FakeOpenPulse2Q,FakeOpenPulse3Q

from qiskit.visualization import pulse_drawer
from qiskit.providers.aer import PulseSimulator, QasmSimulator
from qiskit.providers.aer.noise import NoiseModel

from qiskit.compiler import assemble
# from utils import get_expectations_from_counts
from qiskit.pulse.macros import measure
# from utils import get_prob_from_counts
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.tools.visualization import circuit_drawer
from qiskit import schedule as build_schedule
from qiskit.providers.aer.pulse import PulseSystemModel
import math
import sys
from qiskit import execute
import torch
from qiskit.extensions import UnitaryGate
from torch.utils.data import Dataset,DataLoader

save_file ='pulse_view/'
F_DTYPE = torch.double
def to_quantum_data(self, tensor,n_wire):
    input_vec = tensor.view(-1)
    vec_len = input_vec.size()[0]
    input_matrix = torch.zeros(vec_len, vec_len, dtype = F_DTYPE)
    input_matrix[0] = input_vec
    input_matrix = input_matrix.transpose(0, 1)
    u, s, v = np.linalg.svd(input_matrix)
    output_matrix = torch.tensor(np.dot(u, v), dtype= F_DTYPE)
    return output_matrix

if __name__ == '__main__':
    # (NOTE): Data Loading
    wave_data = np.load('waveforms1500.npy',allow_pickle=True).item()
    wave_amp = wave_data['data'][:,100:900,:]  
    wave_amp = np.abs(wave_amp)
    wave_label = wave_data['target']


    n_samples = wave_amp.shape[0]
    n_time_steps = wave_amp.shape[1]
    n_channels = wave_amp.shape[2]
    print('n_samples:',n_samples,',n_time_steps:',n_time_steps,',n_channels:',n_channels)


    scaling_factor = 1e5
    cut_times = 1
    l_bound = -1   
    u_bound = 1 
    wave_amp = wave_amp * scaling_factor
    wave_amp = np.clip(wave_amp, -1, 1)  # (BS, t-step, channel)
    state_mats = []


    backend = FakeValencia()
    # sys.exit(0)
    backend.configuration().hamiltonian['qub'] = {'0': 2,'1': 2,'2': 2,'3': 2,'4': 2 }
    print('meas_map:',backend.configuration().meas_map)
  
    armonk_model = PulseSystemModel.from_backend(backend)
    config = backend.configuration()
    # print(config.to_dict())

    for i in range(n_samples):
        state_vectors = []
        for k in range(cut_times):
            wave_times =int(n_time_steps/cut_times)
            waves =  wave_amp[i,k*wave_times:(k+1)*wave_times , :]

            #circuit build 
            circ = QuantumCircuit(3)
            circ_name = 'test_'+ str(i) +'_' +str(k)

            circ.append(Gate('pulse_channel_0', 1, []), [0])
            circ.append(Gate('pulse_channel_1', 1, []), [1])
            circ.append(Gate('pulse_channel_2', 1, []), [2])
            circ.measure_all()

        #vecter build
            for j in range(n_channels): #n_channels
                pulse_sample = pulse.Schedule(name='pulse_sample')  # independent of scheduler
                d_c = pulse.DriveChannel(j)
                wave = waves[:, j]
                # wave =  np.repeat(wave,10)
                wave = pulse.library.Waveform(wave)
                pulse_sample += pulse.Play(wave, d_c)
                circ.add_calibration('pulse_channel_'+str(j), [j], pulse_sample)

        # print(circ)
            circ = transpile(circ, backend)
            schedule = build_schedule(circ, backend)
            if i%50 ==0:
                pulse_drawer(schedule, filename=save_file+'pulse_'+circ_name+'.jpg',plot_range=[0, 1000])
            
            n_shots = 4096
            backend_sim = PulseSimulator(system_model=armonk_model)
            qobj = assemble(schedule,backend=backend_sim, shots=n_shots, meas_level=2, meas_return='avg')
            results = backend_sim.run(qobj).result()
            state_vector = results.get_statevector(schedule)
            state_vector = np.array(state_vector)
            state_vectors.append(state_vector)
            counts = results.get_counts(schedule)    # The basis is now in binary number
            print(i,k,':',counts,'; label:',wave_label[i])
        state_vectors = np.stack(state_vectors)
        state_mats.append(state_vectors)
        if i %100 ==0 :
            state_mats_ = np.stack(state_mats)
            np.save('pulse_input/state_mats_abs1x16_'+ str(i),state_mats_)
    state_mats = np.stack(state_mats)
    np.save('pulse_input/state_mats_abs1x16',state_mats)
    print('state_mats\' shape:',state_mats.shape)
