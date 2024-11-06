# https://github.com/yiiyama/qc_random/blob/028a78de6a579dfa6d9da91dd0a4007e0e8f2561/pulse_experiments/pulse_experiment_utils.py
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Barrier
from qiskit.tools.monitor import job_monitor
from qiskit.ignis.verification.tomography import ProcessTomographyFitter

def get_closest_multiple_of_16(num):
    return int(np.round(num / 16.)) * 16

def remove_barriers(circuit):
    iop = 0
    while iop < len(circuit.data):
        if type(circuit.data[iop][0]) is Barrier:
            del circuit.data[iop]
        else:
            iop += 1

    return circuit

def get_instruction_by_name(schedule, pattern):
    return next(inst for _, inst in schedule.instructions if inst.name is not None and re.match(pattern, inst.name))

def submit_schedules(schedules, backend, shots=1024, meas_level=1, meas_return='avg', monitor=True):
    job = backend.run(schedules, meas_level=meas_level, meas_return=meas_return, shots=shots)

    if monitor:
        job_monitor(job)
        return job.result(timeout=120)
    else:
        return job

def plot_result(result, slot, x=None, indices=None, part=np.real, color='black', title='', xlim=None, ylim=None, xlabel='', ylabel=''):
    if indices is None:
        if x is not None:
            indices = range(x.shape[0])
        else:
            indices = range(len(result.results))
        
    y = np.array([part(result.get_memory(int(im))[slot]) for im in indices])

    if x is None:
        x = np.arange(y.shape[0])
        
    plt.scatter(x, y, color=color)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
    return y
    
def plot_counts(result, strings, x=None, indices=None, color='black', title='', xlim=None, ylim=None, xlabel='', ylabel=''):
    if indices is None:
        if x is not None:
            indices = range(x.shape[0])
        else:
            indices = range(len(result.results))

    counts = result.get_counts()

    totals = np.array([sum(counts[im].values()) for im in indices])
    y = np.array([sum(counts[im].get(s, 0) for s in strings) for im in indices]) / totals
    
    if x is None:
        x = np.arange(y.shape[0])
     
    plt.scatter(x, y, color=color)
    #plt.errorbar(x, y, yerr=np.sqrt(y * (1. - y)), color=color)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
    return y
    
def set_initial(state, schedule, backend, qubit, value_dict=None):
    backend_defaults = backend.defaults()
    sx_pulse = backend_defaults.instruction_schedule_map.get('sx', [qubit])[0][1].pulse
    x_pulse = backend_defaults.instruction_schedule_map.get('x', [qubit])[0][1].pulse
    
    qubit_drive = backend.configuration().drive(qubit)
    
    if value_dict is not None:
        schedule = schedule.assign_parameters(value_dict, inplace=False)
    
    with pulse.build(backend=backend) as full_schedule:            
        if state == '1':
            pulse.play(x_pulse, qubit_drive)
        elif state == '+':
            pulse.shift_phase(-np.pi * 0.5, qubit_drive)
            pulse.play(sx_pulse, qubit_drive)
            pulse.shift_phase(-np.pi * 0.5, qubit_drive)
        elif state == 'i':
            p = sx_pulse
            pulse.play(pulse.Drag(duration=p.duration, amp=-p.amp, sigma=p.sigma, beta=p.beta))
            
        pulse.call(schedule)
        
    return full_schedule    
    
def projection(axis, schedule, backend, qubit, spectators=[], value_dict=None, exclude=None):
    backend_defaults = backend.defaults()
    sx_pulse = backend_defaults.instruction_schedule_map.get('sx', [qubit])[0][1].pulse
    x_pulse = backend_defaults.instruction_schedule_map.get('x', [qubit])[0][1].pulse
    
    qubit_drive = backend.configuration().drive(qubit)
    
    if value_dict is not None:
        schedule = schedule.assign_parameters(value_dict, inplace=False)
    
    with pulse.build(backend=backend) as full_schedule:            
        pulse.call(schedule)
        
        if axis == 'x':
            pulse.shift_phase(-np.pi * 0.5, qubit_drive)
            pulse.play(sx_pulse, qubit_drive)
            pulse.shift_phase(-np.pi * 0.5, qubit_drive)
        elif axis == 'y':
            pulse.play(sx_pulse, qubit_drive)
        
        qubits = [qubit] + spectators
        registers = [pulse.MemorySlot(iq) for iq in range(len(qubits))]
        pulse.measure(qubits=qubits, registers=registers)
        
    if exclude is not None:
        return full_schedule.exclude(exclude)
    else:
        return full_schedule
    
class DummyResult(object):
    def __init__(self, counts):
        self.counts = counts
        
    def get_counts(self, label):
        counts_arr = self.counts[label][:]
        num_qubits = int(np.log2(counts_arr.shape[0]))
        template = '{:0%db}' % num_qubits
        
        return dict((template.format(key), value) for key, value in enumerate(counts_arr))
    
def run_process_tomography(counts):
    """Run the process tomography given a counts dict {label: np.array((4,))} (label: e.g. '(("Zp", "Zm"), ("Y", "Z"))')
    """
    
    labels = []
    for key in counts.keys():
        try:
            if type(eval(key)) is tuple:
                labels.append(key)
        except NameError:
            continue
            
    result = DummyResult(counts)

    fitter = ProcessTomographyFitter([result], labels)
    return fitter.fit()

def readout_error_mitigation(counts, error_matrix):
    nshots = np.sum(counts)

    fun = lambda x: np.sum(np.square(counts - np.dot(error_matrix, x)))

    x0 = np.random.rand(counts.shape[0])
    x0 /= np.sum(x0)
    cons = ({'type': 'eq', 'fun': lambda x: nshots - np.sum(x)})
    bnds = tuple((0, nshots) for x in x0)
    res = minimize(fun, x0, method='SLSQP',
                   constraints=cons, bounds=bnds, tol=1e-6)
    return res.x
