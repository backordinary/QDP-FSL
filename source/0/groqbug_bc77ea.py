# https://github.com/rakytap/qgd_DFE/blob/e8be268ac99c5045b4a10fc0a342af0cd7e72244/groq/groqbug.py
import sys, os

sys.setrecursionlimit(sys.getrecursionlimit()*2*4)

import numpy as np
from numba import njit
from numba.np.unsafe.ndarray import to_fixed_tuple
from functools import lru_cache
from scipy.stats import unitary_group

import groq.api as g
import groq.api.instruction as inst
import groq.api.nn as nn
import groq.tensor as tensor
import groq.runner.tsp as tsp
from groq.common import print_utils
try:
    import groq.runtime as runtime
except ImportError:
    # raise ModuleNotFoundError("groq.runtime")
    print('Error: ModuleNotFoundError("groq.runtime")')

def qiskit_oracle(unitary, qbit_num, parameters, target_qbits, control_qbits, usefloat=True):
    from qiskit import Aer
    from qiskit import QuantumCircuit, execute
    backend = Aer.get_backend('unitary_simulator')
    if usefloat: backend.set_option("precision", "single")
    circuit = QuantumCircuit(qbit_num)
    circuit.unitary(unitary, [i for i in range(qbit_num)])
    for param, target_qbit, control_qbit in zip(parameters, target_qbits, control_qbits):
        if control_qbit is None or target_qbit == control_qbit:
            circuit.u(param[0]*2, param[1], param[2], target_qbit)
        else:
            circuit.cry(param[0]*2, control_qbit, target_qbit)
    job = execute(circuit, backend)
    result=job.result()
    U3_qiskit = result.get_unitary(circuit)
    U3_qiskit = np.asarray(U3_qiskit)
    return U3_qiskit
@njit
def make_u3(parameters):
    return np.array(
        [[np.cos(parameters[0]*2/2), -np.exp(parameters[2]*1j)*np.sin(parameters[0]*2/2)],
         [np.exp(parameters[1]*1j)*np.sin(parameters[0]*2/2), np.exp((parameters[1]+parameters[2])*1j)*np.cos(parameters[0]*2/2)]])
@njit
def make_ry(parameters):
    return make_u3([parameters[0], 0, 0])
    #return np.array(
    #    [[np.cos(parameters[0]*2/2), -np.sin(parameters[0]*2/2)],
    #     [np.sin(parameters[0]*2/2), np.cos(parameters[0]*2/2)]])
@njit
def make_controlled(gate):
    return np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), gate]]) #[np.ix_(*([[0,2,1,3]]*2))]
@njit
def make_cry(parameters):
    return make_ry(parameters) #make_controlled(make_ry(parameters))
@njit
def twoByTwoFloat(A, B):
    res = np.empty(B.shape, dtype=B.dtype)
    for i in range(B.shape[1]):
        res[0,i] = (np.real(A[0,0])*np.real(B[0,i])-np.imag(A[0,0])*np.imag(B[0,i])) + (np.real(A[0,1])*np.real(B[1,i])-np.imag(A[0,1])*np.imag(B[1,i]))
        res[0,i] += ((np.real(A[0,0])*np.imag(B[0,i])+np.imag(A[0,0])*np.real(B[0,i])) + (np.real(A[0,1])*np.imag(B[1,i])+np.imag(A[0,1])*np.real(B[1,i]))) * 1j
        #((np.real(A[j,0])*np.imag(B[0,i])+np.real(A[j,1])*np.imag(B[1,i])) + (np.imag(A[j,0])*np.real(B[0,i])+np.imag(A[j,1])*np.real(B[1,i]))) * 1j
    for i in range(B.shape[1]):
        res[1,i] = (np.real(A[1,0])*np.real(B[0,i])-np.imag(A[1,0])*np.imag(B[0,i])) + (np.real(A[1,1])*np.real(B[1,i])-np.imag(A[1,1])*np.imag(B[1,i]))
        res[1,i] += ((np.real(A[1,0])*np.imag(B[0,i])+np.imag(A[1,0])*np.real(B[0,i])) + (np.real(A[1,1])*np.imag(B[1,i])+np.imag(A[1,1])*np.real(B[1,i]))) * 1j
        #((np.real(A[j,0])*np.imag(B[0,i])+np.real(A[j,1])*np.imag(B[1,i])) + (np.imag(A[j,0])*np.real(B[0,i])+np.imag(A[j,1])*np.real(B[1,i]))) * 1j
    return res
#@njit
def apply_to_qbit(unitary, num_qbits, target_qbit, control_qbit, gate):
    pow2qb = 1 << num_qbits
    t = np.arange(num_qbits)
    if not control_qbit is None:
        t[:-1] = np.roll(t[:-1], (target_qbit - control_qbit) % num_qbits)
        gate = make_controlled(gate)
    t = np.roll(t, -target_qbit)
    idxs = np.arange(pow2qb).reshape(*((2,)*num_qbits)).transpose(t).flatten().tolist()
    return np.kron(np.eye(pow2qb>>(1 if control_qbit is None else 2), dtype=np.bool_), gate)[np.ix_(idxs, idxs)].astype(unitary.dtype) @ unitary
@lru_cache
def make_apply_to_qbit_loop(num_qbits):
    twos = tuple([2]*num_qbits)
    @njit
    def apply_to_qbit_loop(unitary, _, target_qbit, control_qbit, gate):
        pow2qb = 1 << num_qbits
        t = np.roll(np.arange(num_qbits), target_qbit)
        idxs = np.arange(pow2qb).reshape(twos).transpose(to_fixed_tuple(t, num_qbits)).copy().reshape(-1, 2) #.reshape(*([2]*num_qbits)).transpose(t).reshape(-1, 2)
        for pair in (idxs if control_qbit is None else idxs[(idxs[:,0] & (1<<control_qbit)) != 0,:]):
            unitary[pair,:] = twoByTwoFloat(gate, unitary[pair,:])
            #unitary[pair,:] = gate @ unitary[pair,:]
        return unitary
    return apply_to_qbit_loop
def process_gates32(unitary, num_qbits, parameters, target_qbits, control_qbits):
    return process_gates(unitary.astype(np.complex64), num_qbits, parameters, target_qbits, control_qbits).astype(np.complex128)
def process_gates(unitary, num_qbits, parameters, target_qbits, control_qbits):
    if unitary.dtype == np.dtype(np.complex128): unitary = np.copy(unitary)
    return process_gates_loop(unitary, num_qbits, parameters, target_qbits, control_qbits, make_apply_to_qbit_loop(num_qbits)) #apply_to_qbit
@njit
def process_gates_loop(unitary, num_qbits, parameters, target_qbits, control_qbits, apply_to_qbit_func):
    for param, target_qbit, control_qbit in zip(parameters, target_qbits, control_qbits):
        unitary = apply_to_qbit_func(unitary, num_qbits, target_qbit, None if control_qbit == target_qbit else control_qbit, (make_u3(param) if control_qbit is None or control_qbit==target_qbit else make_cry(param)).astype(unitary.dtype))
    return unitary
def trace_corrections(result, num_qbits):
    import math
    pow2qb = 1 << num_qbits
    return np.array([np.trace(np.real(result)),
        np.sum(np.real(result[[c ^ (1<<i) for i in range(num_qbits) for c in range(pow2qb)], list(range(pow2qb))*num_qbits])),
        np.sum(np.real(result[[c ^ ((1<<i)+(1<<j)) for i in range(num_qbits-1) for j in range(i+1, num_qbits) for c in range(pow2qb)], list(range(pow2qb))*math.comb(num_qbits, 2)]))])
def test():
    num_qbits, num_gates = 10, 20
    oracles = [process_gates(np.eye(1 << num_qbits) + 0j, num_qbits, np.array([[(25+i+d)%64, (55+i)%64, (50+i)%64] for i in range(num_gates)]), np.array([i % num_qbits for i in range(num_gates)]), np.array([(i*2+d+1) % num_qbits for i in range(num_gates)])) for d in range(4)]
    #print([(oracle, trace_corrections(oracle, num_qbits)) for oracle in oracles]); assert False
    #pi = np.pi
    #parameters = np.array( [pi/2*0.32, pi*1.2, pi/2*0.89])
    num_qbits, use_identity = 5, False
    num_gates = 10
    target_qbits = np.array([np.random.randint(num_qbits) for _ in range(num_gates)], dtype=np.uint8)
    control_qbits = np.array([np.random.randint(num_qbits) for _ in range(num_gates)], dtype=np.uint8)
    parameters = np.random.random((num_gates, 3))
    pow2qb = 1 << num_qbits
    unitary = np.eye(pow2qb) + 0j if use_identity else unitary_group.rvs(pow2qb)
    assert np.allclose(myoracle, qiskit_oracle(unitary, num_qbits, gateparams, target_qbits, control_qbits))
    for i in range(num_qbits):
        for j in range(num_qbits):
            target_qbits, control_qbits = np.array([i, (i+1)%num_qbits, i]), np.array([i, i, j])
            gateparams = np.repeat(parameters.reshape(1,3), 3, axis=0)
            actual, oracle = qiskit_oracle(unitary, num_qbits, gateparams, target_qbits, control_qbits), process_gates(unitary, num_qbits, gateparams, target_qbits, control_qbits)
            assert np.allclose(actual, oracle), (i, j, actual, oracle)
#test()
WEST, EAST = 0, 1
s16rangeW = list(range(25, 27+1))+list(range(29, 37+1))+list(range(39,42+1))
s16rangeE = list(range(26, 27+1))+list(range(29,42+1))
s16rangeW2 = list(range(6, 15+1))+list(range(17, 19+1))+list(range(21, 23+1))
s16rangeE2 = list(range(7, 15+1))+list(range(17, 19+1))+list(range(21, 23+1))+[25]
s8range = list(range(17, 19+1))+list(range(21, 23+1))+list(range(25, 26+1))
s8range2 = [27]+list(range(29, 35+1))
def rev_alu(x, do_rev): return (x//4*4)+3-x%4 if do_rev else x
def get_slice1(drctn, start, bank=0):
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S1(" + str(start) + "), B1(" + str(bank) + ")"
def get_slice2(drctn, start, end, bank=0):
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S2(" + str(start) + "-" + str(end) + "), B1(" + str(bank) + ")"
def get_slice4(drctn, start, end, bank=0):
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S4(" + str(start) + "-" + str(end) + "), B1(" + str(bank) + ")"
def get_slice8(drctn, start, end, bank=0):
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S8(" + str(start) + "-" + str(end) + "), B1(" + str(bank) + ")"
def get_slice16(drctn, slices, bank=0):
    #return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S16(" + ",".join(str(x) for x in slices) + "), B1(" + str(bank) + ")"
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S16(" + str(min(slices)) + "-" + str(max(slices)) + "), B1(" + str(bank) + ")"
def compile_unit_test(name):
    print_utils.infoc("\nCompiling model ...")
    # Compile program to generate IOP. Also generate groqview JSON dump file and
    # check for potential stream conflicts.
    iop_file = g.compile(
        base_name=name, gen_vis_data=True, check_stream_conflicts=True, skip_assembler=False #tree_conflicts=True, inspect_raw=True
    )
    g.write_visualizer_data(name)
    json_file = name + "/visdata.json"
    print_utils.cprint("Have a GroqView:\n    % " + print_utils.Colors.GREEN + "groqview --port 8888 " + json_file + print_utils.Colors.RESET, "")
    return iop_file, json_file
def invoke(devices, iop, pgm_num, ep_num, tensors, lastouts=None, buffers=None):
    """Low level interface to the device driver to access multiple programs. A higher level abstraction
    will be provided in a future release that offers access to multiple programs and entry points."""
    pgm = iop[pgm_num]
    ep = pgm.entry_points[ep_num]
    input_buffers, output_buffers = [], []
    for i, device in enumerate(devices):
        input_buffers.append(runtime.BufferArray(ep.input, 1)[0] if buffers is None else buffers[0][i])
        output_buffers.append(runtime.BufferArray(ep.output, 1)[0] if buffers is None else buffers[1][i])
        if ep.input.tensors:
            for input_tensor in ep.input.tensors:
                if input_tensor.name not in tensors[i]:
                    raise ValueError(f"Missing input tensor named {input_tensor.name}")
                input_tensor.from_host(tensors[i][input_tensor.name], input_buffers[i])
        device.invoke_nonblocking(input_buffers[i], output_buffers[i])
    l = len(devices)
    outs = [{} for _ in range(l)]
    i, checks = -1, list(range(l))
    while l != 0:
        i = (i + 1) % l
        idx = checks[i]
        if not output_buffers[idx].ready(): continue
        if devices[idx].check_faults(): print(devices[idx].dump_faults())
        del checks[i]; l -= 1
        if ep.output.tensors:
            for output_tensor in ep.output.tensors:
                result_tensor = lastouts[idx][output_tensor.name] if not lastouts is None else output_tensor.allocate_numpy_array()
                output_tensor.to_host(output_buffers[idx], result_tensor)
                outs[idx][output_tensor.name] = result_tensor
    return outs, [input_buffers, output_buffers]

class UnitarySimulator(g.Component):
    def __init__(self, num_qbits, reversedir=False, lastus=None, **kwargs):
        super().__init__(**kwargs)
        self.num_qbits, self.rev = num_qbits, reversedir
        #more efficient to just directly copy on the controlled rotation rather than doing unnecessary identity gate computations
        #self.identity2x2 = [g.from_data(np.zeros((1), dtype=np.float32), layout=get_slice1(WEST, 0, 0), broadcast=True),
        #                    g.from_data(np.ones((1), dtype=np.float32), layout=get_slice1(WEST, 0, 0), broadcast=True)]
        self.otherinit, self.copystore = [], []
        pow2qb = 1 << num_qbits
        num_inner_splits = (pow2qb+320-1)//320 #handle inner splits for >=9 qbits
        for hemi in (EAST, WEST) if reversedir else (WEST, EAST):
            self.otherinit.append(lastus.otherinit[hemi] if not lastus is None else tensor.create_storage_request(layout=get_slice8(hemi, s8range[0], s8range[-1], 0).replace(", S8", ", A" + str(pow2qb*num_inner_splits) + "(0-" + str(pow2qb*num_inner_splits-1) + "), S8")))
            self.copystore.append(lastus.copystore[hemi] if not lastus is None else tensor.create_storage_request(layout=get_slice8(hemi, s8range2[0], s8range2[-1], 0).replace(", S8", ", A" + str(pow2qb*num_inner_splits) + "(0-" + str(pow2qb*num_inner_splits-1) + "), S8")))
    def copymatrix(self, unitaryinit):
        unitaryinit = unitaryinit.read(streams=g.SG8[0], time=0)
        unitary = unitaryinit.write(name="unitary", storage_req=self.otherinit[WEST])
        copy = unitaryinit.write(name="initcopy", storage_req=self.copystore[WEST])
        resultother = self.create_memory_tensor(name="result", storage_req=self.otherinit[EAST], tensor_type=unitary.tensor_type)
        copyother = self.create_memory_tensor(name="copy", storage_req=self.copystore[EAST], tensor_type=copy.tensor_type)    
        return unitary, copy, resultother, copyother
    def cmppairs(a, b):
        return a[0].tolist() == b[0].tolist() and ((a[1] is None) and (b[1] is None) or a[1].tolist() == b[1].tolist())
    def idxmapgather(num_qbits):
        pow2qb = 1 << num_qbits
        idxmap = [np.arange(pow2qb).reshape(*([2]*num_qbits)).transpose(np.roll(np.arange(num_qbits), target_qbit)).reshape(-1, 2) for target_qbit in range(num_qbits)]
        idxmapsort = [x[x[:,0].argsort()] for x in idxmap]
        #idxmapm1 = [np.arange(1 << (num_qbits-1)).reshape(*([2]*(num_qbits-1))).transpose(np.roll(np.arange(num_qbits-1), target_qbit)).reshape(-1, 2) for target_qbit in range(num_qbits-1)]
        #idxmapm1 = [x[x[:,0].argsort()] for x in idxmapm1]
        idxmapm1 = [idxmapsort[i][:pow2qb//4,:] for i in range(num_qbits-1)]
        for target_qbit in range(num_qbits):
            for control_qbit in range(num_qbits):
                if target_qbit == control_qbit: assert UnitarySimulator.cmppairs(UnitarySimulator.idxmap(num_qbits, target_qbit, None), (idxmap[target_qbit], None))
                else:
                    idxs = idxmap[target_qbit]
                    oracle = UnitarySimulator.idxmap(num_qbits, target_qbit, control_qbit)
                    assert UnitarySimulator.cmppairs(oracle, (idxs[(idxs[:,0] & (1<<control_qbit)) != 0,:], idxs[(idxs[:,0] & (1<<control_qbit)) == 0,:]))
                    idxs = idxmapsort[target_qbit]
                    oracle = (oracle[0][oracle[0][:,0].argsort()], oracle[1][oracle[1][:,0].argsort()])
                    actual = (np.array(idxs[idxmapm1[(control_qbit - (control_qbit > target_qbit)) % (num_qbits-1)][:,1]]), np.array(idxs[idxmapm1[(control_qbit - (control_qbit > target_qbit)) % (num_qbits-1)][:,0]]))
                    assert UnitarySimulator.cmppairs(oracle, actual), (target_qbit, control_qbit, actual, oracle)
        return idxmapsort, idxmapm1 #target_qbit and control_qbit gather maps
    def idxmap(num_qbits, target_qbit, control_qbit):
        pow2qb = 1 << num_qbits
        t = np.roll(np.arange(num_qbits), target_qbit)
        idxs = np.arange(pow2qb).reshape(*([2]*num_qbits)).transpose(t).reshape(-1, 2)
        pairs = idxs if control_qbit is None else idxs[(idxs[:,0] & (1<<control_qbit)) != 0,:]
        if not control_qbit is None: bypasspairs = idxs[(idxs[:,0] & (1<<control_qbit)) == 0,:]
        else: bypasspairs = None
        #print(pairs, bypasspairs)
        return pairs, bypasspairs
    def intersection_range(r1, r2):
        mx, mn = max(r1[0], r2[0]), min(r1[-1], r2[-1])
        return (mx, mn) if mx < mn else None
    def difference_range(r1, r2):
        return None if r1 == r2 else (max(r2[0], r1[1]), r2[1])
    def difference_ranges(r, ranges):
        return [(None, None) if x is None else (UnitarySimulator.intersection_range(r, x), UnitarySimulator.difference_range(r, x)) for x in ranges]
    def smallest_contig_range(ranges):
        if all(r is None for r in ranges): return None
        ranges = list(filter(lambda k: not k is None, ranges))
        m = min(ranges)[0]
        return (m, min(map(lambda k: k[1] if k[0] == m else k[0], ranges))) 
    def transpose_null_share(tensors, schedules, gaps, time, tpose_shared):
        Allsplits = [[] for _ in tensors]
        while True:
            r = UnitarySimulator.smallest_contig_range([x[0] for x in schedules])
            if r is None: break
            diffs = UnitarySimulator.difference_ranges(r, [x[0] for x in schedules])
            for i, x in enumerate(diffs):
                Allsplits[i].append(None if x[0] is None else x[0][1] - x[0][0])
                if x[1] is None:
                    del schedules[i][0]
                    if len(schedules[i]) == 0: schedules[i].append(None) 
                else: schedules[i][0] = x[1]
        tensors = [y.split_vectors([x for x in Allsplits[i] if not x is None]) for i, y in enumerate(tensors)]
        t, Allindexes = 0, [0 for _ in tensors]
        for i, splits in enumerate(zip(*Allsplits)):
            with g.ResourceScope(name="dist" + str(i), is_buffered=False, time=time+t) as innerpred:
                res = tpose_shared([None if x is None else tensors[j][Allindexes[j]] for j, x in enumerate(splits)])
                for j, x in enumerate(splits):
                    if not x is None: tensors[j][Allindexes[j]] = res[j]
                Allindexes = [x + (0 if splits[j] is None else 1) for j, x in enumerate(Allindexes)]
            t += next(filter(lambda k: not k is None, splits))
            if t in gaps: t += gaps[t]
        return [g.concat(x, 0) for x in tensors] 
    def build(self, unitary, copy, target_qbit, control_qbit, gate, gatesel=None, tcqbitsel=None, derivdistro=None, inittime=0):
        if copy is None:
            with g.ResourceScope(name="initcopy", is_buffered=True, time=0) as pred:
                unitary, copy, _, _ = self.copymatrix(unitary)
        else: pred = None
        pow2qb = 1 << self.num_qbits
        num_inner_splits = 1 if gatesel is None else (pow2qb+320-1)//320
        innerdim = pow2qb if gatesel is None else 320
        usplit = np.array(g.split_vectors(unitary, [1] * (2*pow2qb*num_inner_splits))).reshape(pow2qb*num_inner_splits, 2)
        ucopysplit = np.array(g.split_vectors(copy, [1] * (2*pow2qb*num_inner_splits))).reshape(pow2qb*num_inner_splits, 2)
        if tcqbitsel is None:
            pairs, bypasspairs = UnitarySimulator.idxmap(self.num_qbits, target_qbit, control_qbit)
            u = [usplit[pairs[:,0],0], usplit[pairs[:,0],1], ucopysplit[pairs[:,1],0], ucopysplit[pairs[:,1],1]]
            ub = [np.array([])]*4 if control_qbit is None else [usplit[bypasspairs[:,0],0], usplit[bypasspairs[:,0],1], ucopysplit[bypasspairs[:,1],0], ucopysplit[bypasspairs[:,1],1]]
            revidx = np.argsort((pairs if control_qbit is None else np.hstack([bypasspairs, pairs])).transpose().flatten()).tolist()         
        r = 1 if control_qbit is None else 2
        with g.ResourceScope(name="rungate", is_buffered=True, time=0 if pred is None else None, predecessors=None if pred is None else [pred]) as pred:
            #(a+bi)*(c+di)=(ac-bd)+(ad+bc)i
            #gate[0] * p[0] - gate[1] * p[1] + gate[2] * p[2] - gate[3] * p[3]
            #gate[0] * p[1] + gate[1] * p[0] + gate[2] * p[3] + gate[3] * p[2]
            if gatesel is None:
                gatevals = g.split_vectors(gate, [1]*(2*2*2))
                gs = [g.concat_vectors([gatevals[i]]*(pow2qb//2*num_inner_splits//r)+[gatevals[i+4]]*(pow2qb//2*num_inner_splits//r), (pow2qb//r, pow2qb)).read(streams=g.SG4[2*i]) for i in range(4)] #, time=0 if i == 0 else None
            else:
                #gate = g.from_addresses(np.array(gate.addrs).reshape(-1, g.float32.size), pow2qb, g.float32, "gatedim")
                gatevals = np.array(g.split_vectors(gate, [1]*(gate.shape[0]))).reshape(gate.shape[0]//8, 2*2*2)
                #gatesel_st = g.concat_vectors([gatesel[i].reshape(1,innerdim) for i in range(len(gatesel)) for _ in range(pow2qb//2*num_inner_splits//r)], (pow2qb*len(gatesel)//2*num_inner_splits//r, innerdim)).read(streams=g.SG4[1])
                #gs = [g.mem_gather(g.concat_vectors(gatevals[:,i], (gate.shape[0]//8, innerdim)), gatesel_st, output_streams=[g.SG4[2*i if i==3 else 2*i]]) for i in range(4)]
                gatesel_st = [gatesel[j].read(streams=g.SG4[1]) for j in range(4)]
                gs = [[g.mem_gather(g.concat_vectors(gatevals[:,i], (gate.shape[0]//8, innerdim)), gatesel_st[j], output_streams=[g.SG4[2*i if i==3 else 2*i]]) for i in range(4)] for j in range(4)]
            with g.ResourceScope(name="ident", is_buffered=False, time=0) as innerpred:
                if tcqbitsel is None:
                    us = [g.concat_vectors((ub[i%2].flatten().tolist() + ub[i%2+2].flatten().tolist() if i in [0,3] else []) + u[i].flatten().tolist()*2, (pow2qb*num_inner_splits if control_qbit is None or i in [0,3] else pow2qb//2*num_inner_splits, innerdim)).read(streams=g.SG4[2*i+1]) for i in range(4)]
                else:
                    if len(tcqbitsel) == 6 or len(tcqbitsel) == 8:
                        if len(tcqbitsel) == 8:
                            tqbitdistro, tqbitpairs0, tqbitpairs1, cqbitdistro, cqbitpairs0, cqbitpairs1, tcqbitdistro, cqbithighsel = tcqbitsel
                            cqbithighsel1 = g.split(cqbithighsel[1], num_splits=2)
                            tchighgather = g.distribute_8(g.concat([cqbithighsel1[0]]*2+[cqbithighsel1[1]]*2, 0).read(streams=g.SG1[24]), tcqbitdistro[1].read(streams=g.SG1[25]), bypass8=0b11111110, distributor_req=3+(4 if self.rev else 0), time=0)
                            cdistro = g.stack(pow2qb*num_inner_splits*[cqbitdistro[1]], 0).read(streams=g.SG1[16+1]) #, time=1 #slice 14->slice 41 = 44//4-14//4=8
                            lasttensor = [None]
                            def shared_tpose(tensors):
                                if tensors[1] is None:
                                    lasttensor[0] = g.transpose_null(tensors[0], transposer_req=3 if self.rev else 1, stream_order=[8], time=0)
                                    return lasttensor[0], None
                                if lasttensor[0].shape != tensors[1].shape: lt, lasttensor[0] = g.split(lasttensor[0], splits=[tensors[1].shape[0], lasttensor[0].shape[0] - tensors[1].shape[0]])
                                else: lt, lasttensor[0] = lasttensor[0], None
                                tchigh = g.mem_gather(cqbitpairs0[1], lt, output_streams=[g.SG1[16]]) #, time=-7
                                readcontrols = g.distribute_8(tchigh, tensors[1], bypass8=0b11111110, distributor_req=2+(4 if self.rev else 0)) #, time=-1
                                if tensors[0] is None: return None, g.transpose_null(readcontrols, transposer_req=3 if self.rev else 1, stream_order=[0], time=0)
                                else:
                                    readcontrols, lt = g.split(g.transpose_null(g.stack((readcontrols, tensors[0]), 1), transposer_req=3 if self.rev else 1, stream_order=[0, 8], time=0), num_splits=2, dim=1)
                                    lt = lt.reshape(lt.shape[0], lt.shape[-1])
                                    if lasttensor[0] is None: lasttensor[0] = lt
                                    else: lasttensor[0] = g.concat((lasttensor[0], lt), 0)
                                    return lt, readcontrols.reshape(readcontrols.shape[0], readcontrols.shape[-1])
                            _, readcontrols = UnitarySimulator.transpose_null_share([tchighgather, cdistro],
                                [[(x, min(x+15, tchighgather.shape[0])) for x in range(0, tchighgather.shape[0], 15)], [(15, 15+cdistro.shape[0])]], {}, 1, shared_tpose)
                            #tr = tensor.create_transposer_request(trans=[3 if self.rev else 1])
                            #tchighgather = g.transpose_null(tchighgather, transposer_req=tr, stream_order=[8])
                            #tchigh = g.mem_gather(cqbitpairs0[1], tchighgather, output_streams=[g.SG1[16]])
                            #readcontrols = g.distribute_8(tchigh, cdistro, bypass8=0b11111110, distributor_req=2+(4 if self.rev else 0))
                            #readcontrols = g.transpose_null(readcontrols, transposer_req=tr, stream_order=[0])
                        else:
                            tqbitdistro, tqbitpairs0, tqbitpairs1, cqbitdistro, cqbitpairs0, cqbitpairs1 = tcqbitsel
                            if self.num_qbits > 8:
                                for x in (tqbitpairs0, tqbitpairs1, cqbitpairs0, cqbitpairs1):
                                    for i in range(2): x[i] = g.split(x[i], num_splits=2)[target_qbit//8]
                            if self.num_qbits > 9:
                                for x in (cqbitpairs0, cqbitpairs1):
                                    for i in range(2): x[i] = g.split(x[i], num_splits=2)[control_qbit//8]
                            cdistro = cqbitdistro[1].read(streams=g.SG1[16+1])
                            readcontrols = g.distribute_8(g.stack([cqbitpairs0[1]]*2 + [cqbitpairs1[1]]*2, 0).reshape(pow2qb*num_inner_splits, innerdim).read(streams=g.SG1[16]), cdistro, bypass8=0b11111110, distributor_req=2+(4 if self.rev else 0))
                            readcontrols = g.transpose_null(readcontrols, transposer_req=3 if self.rev else 1, stream_order=[0], time=0)
                        tqb = g.mem_gather(tqbitpairs0[1], readcontrols, output_streams=[g.SG1[0]])
                        tqbp = g.mem_gather(tqbitpairs1[1], readcontrols, output_streams=[g.SG1[8]])
                    else:
                        tqbitdistro, tqbitpairs0, tqbitpairs1 = tcqbitsel
                        if self.num_qbits > 8:
                            for x in (tqbitpairs0, tqbitpairs1):
                                for i in range(2): x[i] = g.split(x[i], num_splits=2)[target_qbit//8]
                        tqb = g.concat_vectors([tqbitpairs0[1]]*2, (pow2qb*num_inner_splits, innerdim)).read(streams=g.SG1[0])
                        tqbp = g.concat_vectors([tqbitpairs1[1]]*2, (pow2qb*num_inner_splits, innerdim)).read(streams=g.SG1[8])
                    distro = tqbitdistro[1].read(streams=g.SG1[18])
                    readaddrs = g.distribute_lowest(tqb, distro, bypass8=0b11110000, distributor_req=0+(4 if self.rev else 0)) #.reinterpret(g.uint32)
                    readaddrpairs = g.distribute_lowest(tqbp, distro, bypass8=0b11110000, distributor_req=1+(4 if self.rev else 0)) #.reinterpret(g.uint32)
                    readaddrs, readaddrpairs = g.split(g.transpose_null(g.stack([readaddrs, readaddrpairs], 1), transposer_req=2 if self.rev else 0, stream_order=[0, 1, 2, 3, 8, 9, 10, 11]), dim=1, num_splits=2)

                    if len(gatesel) != 4 and len(tcqbitsel) == 6:
                        readaddrs = readaddrs.split(dim=0, num_splits=pow2qb*num_inner_splits)
                        readaddrpairs = readaddrpairs.split(dim=0, num_splits=pow2qb*num_inner_splits)
                        readaddrs, readaddrpairs = g.concat_vectors([(readaddrs if (i & (pow2qb*num_inner_splits//4)) == 0 else readaddrpairs)[i] for i in range(pow2qb//2*num_inner_splits)] + readaddrs[pow2qb//2*num_inner_splits:], (pow2qb*num_inner_splits, 1, 4, innerdim)), g.concat_vectors([(readaddrs if (i & (pow2qb*num_inner_splits//4)) == 0 else readaddrpairs)[i] for i in range(pow2qb//2*num_inner_splits)] + readaddrpairs[pow2qb//2*num_inner_splits:], (pow2qb*num_inner_splits, 1, 4, innerdim))
                    readaddrs, readaddrpairs = [x.reshape(pow2qb*num_inner_splits, innerdim) for x in g.split(readaddrs, dim=2, num_splits=4)], [x.reshape(pow2qb*num_inner_splits, innerdim) for x in g.split(readaddrpairs, dim=2, num_splits=4)]
                    #s8range
                    us = [g.stack([g.mem_gather(g.split_vectors(g.concat_vectors(x, (pow2qb*num_inner_splits, innerdim)).reinterpret(g.uint8).transpose(1, 0, 2), [pow2qb*num_inner_splits]*4)[j],
                                    *[z if control_qbit is None or i in [0,3] or len(gatesel)==4 else g.split_vectors(z, [pow2qb//2*num_inner_splits]*2)[1] for z in (readaddrs[j] if i<2 else readaddrpairs[j],)], output_streams=[g.SG1[4*(2*i+1 if (i&1)!=0 else 2*i+1)+j]]) for j in range(4)], 1).reinterpret(g.float32)
                            for i, x in enumerate((usplit[:,0], usplit[:,1], ucopysplit[:,0], ucopysplit[:,1]))]
                    if inittime == 10:
                        test = us[3].reinterpret(g.uint8).split(dim=-2, num_splits=4)[3].write(name="r" + str(inittime), program_output=True, layout="-1, H1(" + ("E" if self.rev else "W") + "), S1(" + str(DUMP_AT_SLICE) + "), B1(1), A4(" + str(inittime*4) + "-" + str((inittime+1)*4-1) + ")") #25->23 corruption on 3rd value, 21->19 corruption on 2nd value
                        g.add_mem_constraints([test], [unitary], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                usb = [[]]*2
                if not control_qbit is None and (gatesel is None or len(gatesel) == 2):
                    for i in [0,3]:
                        usb[i%2], us[i] = g.split_vectors(us[i], [pow2qb//2*num_inner_splits, pow2qb//2*num_inner_splits])
                    #usb = [g.vxm_identity(usb[i], alus=[[rev_alu(13, self.rev),rev_alu(14, self.rev)][i]], time=0, output_streams=g.SG4[[1,7][i]]) for i in range(2)]
                    if derivdistro is None:
                        usb = [g.vxm_identity(usb[i], alus=[[rev_alu(15, self.rev),rev_alu(11, self.rev)][i]], time=0 if tcqbitsel is None or control_qbit is None else None, output_streams=g.SG4[[1,5][i]]) for i in range(2)]
                    else:
                        readddistro = g.concat_vectors([derivdistro.reshape(1, 320)]*(pow2qb//2*num_inner_splits), (pow2qb//2*num_inner_splits, 320)).read(streams=g.SG4[6])
                        usb = [g.mul(usb[i], readddistro, alus=[[rev_alu(15, self.rev),rev_alu(11, self.rev)][i]], time=0 if tcqbitsel is None or control_qbit is None else None, output_streams=g.SG4[[1,5][i]]) for i in range(2)]
us[0] = us[0].split(num_splits=4)us[1] = us[1].split(num_splits=4)us[2] = us[2].split(num_splits=4)us[3] = us[3].split(num_splits=4)            m1reqs = [tensor.create_alu_request(rev_alu(x, self.rev)) for x in (0, 4, 8, 12)]
            m2reqs = [tensor.create_alu_request(rev_alu(x, self.rev)) for x in (2, 3, 10, 11)]
            m1, m2 = [], []
            m1.append([g.mul(gs[0][i], us[i][0], alus=m1reqs[i], output_streams=g.SG4[[0,2,4,6][i]], time=(0 if control_qbit is None else pow2qb*num_inner_splits) if i==0 and (gatesel is None or len(gatesel) != 4) and (tcqbitsel is None or control_qbit is None) else None) for i in range(4)])
            m2.append([g.mul(gs[0][i], us[i^1][0], alus=m2reqs[i], output_streams=g.SG4[[3,3,5,5][i]]) for i in range(4)])
            m1.append([g.mul(gs[1][i], us[i][1], alus=m1reqs[i], output_streams=g.SG4[[0,2,4,6][i]], time=(0 if control_qbit is None else pow2qb*num_inner_splits) if i==0 and (gatesel is None or len(gatesel) != 4) and (tcqbitsel is None or control_qbit is None) else None) for i in range(4)])
            m2.append([g.mul(gs[1][i], us[i^1][1], alus=m2reqs[i], output_streams=g.SG4[[3,3,5,5][i]]) for i in range(4)])
            m1.append([g.mul(gs[2][i], us[i][2], alus=m1reqs[i], output_streams=g.SG4[[0,2,4,6][i]], time=(0 if control_qbit is None else pow2qb*num_inner_splits) if i==0 and (gatesel is None or len(gatesel) != 4) and (tcqbitsel is None or control_qbit is None) else None) for i in range(4)])
            m2.append([g.mul(gs[2][i], us[i^1][2], alus=m2reqs[i], output_streams=g.SG4[[3,3,5,5][i]]) for i in range(4)])
            m1.append([g.mul(gs[3][i], us[i][3], alus=m1reqs[i], output_streams=g.SG4[[0,2,4,6][i]], time=(0 if control_qbit is None else pow2qb*num_inner_splits) if i==0 and (gatesel is None or len(gatesel) != 4) and (tcqbitsel is None or control_qbit is None) else None) for i in range(4)])
            m2.append([g.mul(gs[3][i], us[i^1][3], alus=m2reqs[i], output_streams=g.SG4[[3,3,5,5][i]]) for i in range(4)])
            m1 = [g.stack([m1[i][j] for i in range(4)], 0) for j in range(4)]
            m2 = [g.stack([m2[i][j] for i in range(4)], 0) for j in range(4)]
            a1 = [g.sub(m1[2*i], m1[2*i+1], alus=[[rev_alu(1, self.rev),rev_alu(9, self.rev)][i]], output_streams=g.SG4[[0,6][i]]) for i in range(2)]
            a2 = [g.add(m2[i], m2[2+i], alus=[[rev_alu(5, self.rev),rev_alu(6, self.rev)][i]], output_streams=g.SG4[[4,3][i]]) for i in range(2)]
            ri = [g.add(a1[0], a1[1], alus=[rev_alu(15, self.rev)], output_streams=g.SG4[1]),
                  g.add(a2[0], a2[1], alus=[rev_alu(7, self.rev)], output_streams=g.SG4[2])]
            if tcqbitsel is None:
                ri = g.concat_vectors(np.hstack([np.array(g.split_vectors(ri[i] if control_qbit is None else g.concat_vectors([usb[i], ri[i]], (pow2qb*num_inner_splits, innerdim)), [1]*(pow2qb*num_inner_splits)))[revidx].reshape(pow2qb*num_inner_splits, 1) for i in range(2)]).flatten().tolist(), (pow2qb*num_inner_splits*2, innerdim))
                result = ri.write(name="result", storage_req=self.otherinit[EAST])
                copy = ri.write(name="copy", storage_req=self.copystore[EAST])
            else:
                if len(tcqbitsel) == 6 or len(tcqbitsel) == 8:
                    if len(gatesel) != 4: ri = [g.concat_vectors([usb[i], ri[i]], (pow2qb*num_inner_splits, innerdim)) for i in range(2)]
                    rigap, delay = 0 if len(gatesel) == 4 else 3*2, 4+4+1+4+2 #3 cycle ALU time and transposer time=4, IO crossing time=4, gather time=1, IO crossing time=4, distributor crossing time=2 (1 pre-entry, 1 for distributor operation)
                    tposediff = 4+4+44//4+5+3*3+44//4+4+2 #50, the ALU entry delay is 5 cycles -or- the read delay is 1 cycle + ALU entry delay of 4 cycles, but there is no ALU exit delay 
                    if len(tcqbitsel) == 8:
                        cdistro = g.stack(pow2qb*num_inner_splits*[cqbitdistro[0]], 0).read(streams=g.SG1[16+1])
                        cqbithighsel0 = g.split(cqbithighsel[0], num_splits=2)
                        tchighgather = g.distribute_8(g.concat([cqbithighsel0[0]]*2+[cqbithighsel0[1]]*2, 0).read(streams=g.SG1[24]), tcqbitdistro[0].read(streams=g.SG1[25]), bypass8=0b11111110, distributor_req=3+(0 if self.rev else 4), time=tposediff)
                        #tchighgather = g.transpose_null(tchighgather, transposer_req=3 if self.rev else 1, stream_order=[8], time=0)
                        #tchigh = g.mem_gather(cqbitpairs0[0], tchighgather, output_streams=[g.SG1[16]])
                        lasttensor = [None]
                        def shared_tpose(tensors):
                            if tensors[1] is None:
                                lasttensor[0] = g.transpose_null(tensors[0], transposer_req=1 if self.rev else 3, stream_order=[8], time=0)
                                return lasttensor[0], None
                            if lasttensor[0].shape != tensors[1].shape: lt, lasttensor[0] = g.split(lasttensor[0], splits=[tensors[1].shape[0], lasttensor[0].shape[0] - tensors[1].shape[0]])
                            else: lt, lasttensor[0] = lasttensor[0], None
                            tchigh = g.mem_gather(cqbitpairs0[0], lt, output_streams=[g.SG1[16]]) #, time=-7
                            writecontrols = g.distribute_8(tchigh, tensors[1], bypass8=0b11111110, distributor_req=2+(0 if self.rev else 4)) #, time=-1
                            if tensors[0] is None:
                                writecontrols = g.transpose_null(writecontrols, transposer_req=1 if self.rev else 3, stream_order=[0], time=0)
                                return None, writecontrols.reshape(writecontrols.shape[0], innerdim)                                  
                            else:                                
                                writecontrols, lt = g.split(g.transpose_null(g.stack((writecontrols, tensors[0]), 1), transposer_req=1 if self.rev else 3, stream_order=[0, 8], time=0), num_splits=2, dim=1)
                                lt = lt.reshape(lt.shape[0], lt.shape[-1])
                                if lasttensor[0] is None: lasttensor[0] = lt
                                else: lasttensor[0] = g.concat((lasttensor[0], lt), 0)
                                return lt, writecontrols.reshape(writecontrols.shape[0], writecontrols.shape[-1])                                
                        #ri[1], _, writecontrols = UnitarySimulator.transpose_null_share([ri[1], tchighgather, cdistro], [[(delay+delay, delay+delay+pow2qb*num_inner_splits)], [(x, min(x+delay, pow2qb*num_inner_splits)) for x in range(0, pow2qb*num_inner_splits, delay)], [(delay, delay+pow2qb*num_inner_splits)]], {}, 1+tposediff, shared_tpose)
                        _, writecontrols = UnitarySimulator.transpose_null_share([tchighgather, cdistro], [[(x, min(x+delay, pow2qb*num_inner_splits)) for x in range(0, pow2qb*num_inner_splits, delay)], [(delay, delay+pow2qb*num_inner_splits)]], {}, 1+tposediff, shared_tpose)
                        #tr = tensor.create_transposer_request(trans=[1 if self.rev else 3])
                        #ri[1] = g.transpose_null(ri[1], transposer_req=tr, stream_order=[4, 5, 6, 7])
                        #tchighgather = g.transpose_null(tchighgather, transposer_req=tr, stream_order=[8])
                        #tchigh = g.mem_gather(cqbitpairs0[0], tchighgather, output_streams=[g.SG1[16]])
                        #writecontrols = g.distribute_8(tchigh, cdistro, bypass8=0b11111110, distributor_req=2+(0 if self.rev else 4))  
                        #writecontrols = g.transpose_null(writecontrols, transposer_req=tr, stream_order=[0])
                    else:
                        cdistro = cqbitdistro[0].read(streams=g.SG1[16+1])
                        tchigh = g.stack([cqbitpairs0[0]]*2 + [cqbitpairs1[0]]*2, 0).read(streams=g.SG1[16])
                        writecontrols = g.distribute_8(tchigh, cdistro, bypass8=0b11111110, distributor_req=2 if self.rev else 6)
                        scheduleri = [(delay, delay+pow2qb*num_inner_splits)]
                        schedulewrite = [(0, pow2qb*num_inner_splits)]
                        gaps = {} if pow2qb*num_inner_splits >= delay else {pow2qb*num_inner_splits: delay-pow2qb*num_inner_splits}
                        #gaps = {pow2qb//2*num_inner_splits: rigap, delay+pow2qb//2*num_inner_splits: rigap} if pow2qb//2*num_inner_splits <= delay else {}
                        def shared_tpose(tensors):
                            if tensors[0] is None: return None, g.transpose_null(tensors[1], transposer_req=1 if self.rev else 3, stream_order=[0], time=0)
                            elif tensors[1] is None: return g.transpose_null(tensors[0], transposer_req=1 if self.rev else 3, stream_order=[4, 5, 6, 7], time=0), None
                            tensors[1], tensors[0] = g.split(g.transpose_null(g.concat([tensors[1].reshape(tensors[1].shape[0], 1, innerdim), tensors[0].reinterpret(g.uint8)], 1), transposer_req=1 if self.rev else 3, stream_order=[0, 4, 5, 6, 7], time=0), dim=1, splits=[1, 4])
                            return tensors[0].reinterpret(g.float32).reshape(tensors[0].shape[0], innerdim), tensors[1].reshape(tensors[1].shape[0], innerdim)                    
                        #ri[1], writecontrols = UnitarySimulator.transpose_null_share([ri[1], writecontrols], [scheduleri, schedulewrite], gaps, tposediff if len(gatesel)==4 else tposediff-3*2, shared_tpose) #t=51 when gather transpose_null resource scope bases from but we are relative again to parent here
                        tr = tensor.create_transposer_request(trans=[1 if self.rev else 3])
                        #ri[1] = g.transpose_null(ri[1], transposer_req=tr, stream_order=[4, 5, 6, 7])
                        writecontrols = g.transpose_null(writecontrols, transposer_req=tr, stream_order=[0], time=tposediff if len(gatesel)==4 else tposediff-3*2)
                        
                    #writecontrols, ri[1] = g.split(g.transpose_null(g.concat([writecontrols.reshape(-1, 1, innerdim), ri[1].reinterpret(g.uint8)], 1), transposer_req=1 if self.rev else 3, stream_order=[0, 4, 5, 6, 7]), dim=1, splits=[1, 4])
                    #ri[1] = ri[1].reinterpret(g.float32).reshape(pow2qb*num_inner_splits, innerdim)
                    #writecontrols = writecontrols.reshape(pow2qb*num_inner_splits, innerdim)
                    dist_st = g.distribute_lowest(g.concat_vectors([g.mem_gather((tqbitpairs0 if (i & (pow2qb*num_inner_splits//4)) == 0 else tqbitpairs1)[0], x, output_streams=[g.SG1[0]]) for i, x in enumerate(writecontrols.split_vectors([1]*pow2qb*num_inner_splits))], (pow2qb*num_inner_splits, innerdim)), tqbitdistro[0].read(streams=g.SG1[12]), bypass8=0b11110000, distributor_req=0 if self.rev else 4)
                    #dist_st = g.distribute_lowest(g.mem_gather(g.stack([tqbitpairs0[0], tqbitpairs1[0]], dim=0).reshape(2, pow2qb//4*num_inner_splits, 2, innerdim).transpose(0,2,1,3), writecontrols, output_streams=[g.SG1[8]]), tqbitdistro[0].read(streams=g.SG1[12]), bypass8=0b11110000, distributor_req=1 if self.rev else 5)
                else:
                    dist_st = g.distribute_lowest(g.concat_vectors([tqbitpairs0[0], tqbitpairs1[0]], (pow2qb*num_inner_splits, innerdim)), tqbitdistro[0].read(streams=g.SG1[12]), bypass8=0b11110000, distributor_req=0 if self.rev else 4)
                    #ri[1] = g.transpose_null(ri[1], transposer_req=1 if self.rev else 3, stream_order=[4, 5, 6, 7])
                #ri[1] = g.distribute_8(ri[1], bypass8=0b11110000, distributor_req=1 if self.rev else 5, use_identity_map=True)
                writeaddrs, ri[0], ri[1] = g.split(g.transpose_null(g.stack([dist_st, ri[0].reinterpret(g.uint8), ri[1].reinterpret(g.uint8)], 1), transposer_req=0 if self.rev else 2, stream_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), dim=1, num_splits=3)
                ri[0] = ri[0].reinterpret(g.float32).reshape(pow2qb*num_inner_splits, innerdim)
                ri[1] = ri[1].reinterpret(g.float32).reshape(pow2qb*num_inner_splits, innerdim)
                result = g.from_addresses(np.array(self.otherinit[EAST].addresses).reshape(-1, g.float32.size), innerdim, g.float32, "result")
                copy = g.from_addresses(np.array(self.copystore[EAST].addresses).reshape(-1, g.float32.size), innerdim, g.float32, "copy")
                writeaddrs = [x.reshape(pow2qb*num_inner_splits, innerdim) for x in g.split(writeaddrs, dim=2, num_splits=4)]
                ri = [[x.reshape(pow2qb*num_inner_splits, innerdim) for x in g.split(ri[i].reinterpret(g.uint8), dim=1, num_splits=4)] for i in range(2)]
                g.mem_scatter(ri[0][0], g.split(g.split(copy.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[0].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[0], index_tensor=writeaddrs[0])
                g.mem_scatter(ri[0][0], g.split(g.split(result.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[0].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[0], index_tensor=writeaddrs[0])
                g.mem_scatter(ri[0][1], g.split(g.split(copy.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[0].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[1], index_tensor=writeaddrs[1])
                g.mem_scatter(ri[0][1], g.split(g.split(result.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[0].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[1], index_tensor=writeaddrs[1])
                g.mem_scatter(ri[0][2], g.split(g.split(copy.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[0].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[2], index_tensor=writeaddrs[2])
                g.mem_scatter(ri[0][2], g.split(g.split(result.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[0].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[2], index_tensor=writeaddrs[2])
                g.mem_scatter(ri[0][3], g.split(g.split(copy.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[0].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[3], index_tensor=writeaddrs[3])
                g.mem_scatter(ri[0][3], g.split(g.split(result.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[0].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[3], index_tensor=writeaddrs[3])
                g.mem_scatter(ri[1][0], g.split(g.split(copy.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[1].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[0], index_tensor=writeaddrs[0])
                g.mem_scatter(ri[1][0], g.split(g.split(result.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[1].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[0], index_tensor=writeaddrs[0])
                g.mem_scatter(ri[1][1], g.split(g.split(copy.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[1].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[1], index_tensor=writeaddrs[1])
                g.mem_scatter(ri[1][1], g.split(g.split(result.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[1].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[1], index_tensor=writeaddrs[1])
                g.mem_scatter(ri[1][2], g.split(g.split(copy.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[1].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[2], index_tensor=writeaddrs[2])
                g.mem_scatter(ri[1][2], g.split(g.split(result.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[1].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[2], index_tensor=writeaddrs[2])
                g.mem_scatter(ri[1][3], g.split(g.split(copy.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[1].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[3], index_tensor=writeaddrs[3])
                g.mem_scatter(ri[1][3], g.split(g.split(result.reshape(pow2qb*num_inner_splits, 2, innerdim), dim=1, num_splits=2)[1].reinterpret(g.uint8).reshape(pow2qb*num_inner_splits, 4, innerdim), dim=1, num_splits=4)[3], index_tensor=writeaddrs[3])
        return result, copy
    def unpack_broadcast(tensor, distmaps, temp_store, inner_splits, reversedir, width, writefn):
        #720=320+360 extra cost due to shifter cycles = 22 (first read delay)+19*19+17 (final write delay)
        inpstream, outpstream, bypass8, mask_bitmap = (g.SG4[1] if width == 4 else (g.SG2[2] if width == 2 else g.SG1[4]),
            g.SG4[0] if width == 4 else (g.SG2[0] if width == 2 else g.SG1[0]),
                0b11110000 if width == 4 else (0b11111100 if width == 2 else 0b11111110),
                0b0000 if width == 4 else (0b1100 if width == 2 else 0b1110)) 
        pred, cur_mt, outp = None, tensor, []        
        for i in range(20):
            if i != 0:
                with g.ResourceScope(name="shift" + str(i), is_buffered=True, predecessors=[pred], time=None) as pred: #time=16*inner_splits+(16*inner_splits+19*inner_splits)*(i-1)) as pred:
                    cur_mt = g.shift(cur_mt, 16, permutor_id=1 if reversedir else 0, shift_src=[inst.NEW_SRC]*width, dispatch_set=inst.DispatchSet.SET_0, input_streams=inpstream, output_streams=outpstream, time=0).write(name="outshift" + str(i), storage_req=temp_store)
            with g.ResourceScope(name="bcast" + str(i), is_buffered=True, predecessors=None if pred is None else [pred], time=0 if pred is None else None) as pred: #time=(16*inner_splits+19*inner_splits)*i) as pred:
                tobcast = g.distribute_8(g.stack([cur_mt]*16, 0), g.concat([distmaps]*inner_splits, 0), distributor_req=4 if reversedir else 0, bypass8=bypass8, map_stream_req=g.SG1[5])
                writefn(g.broadcast_lane_0(tobcast, permutor_req=1 if reversedir else 0, dispatch_set=inst.DispatchSet.SET_0, old_bitmap=[inst.NEW_SRC]*width, mask_bitmap=mask_bitmap, input_streams=outpstream, output_streams=outpstream, time=0))
    def get_correction_masks(num_qbits, second=False):
        pow2qb = 1 << num_qbits
        num_inner_splits = (pow2qb+320-1)//320
        if not second: m = [(c ^ (1<<i), c) for i in range(num_qbits) for c in range(pow2qb)]
        else: m = [(((1<<i)+(1<<j)) ^ c, c) for i in range(num_qbits-1) for j in range(i+1, num_qbits) for c in range(pow2qb)]
        d = {x: [[] for _ in range(num_inner_splits)] for x in range(1<<num_qbits)}
        for x in m: d[x[0]][x[1]//min(pow2qb, 256)].append(x[1]%min(pow2qb, 256)) 
        masks = {frozenset(d[x][i]) for x in d for i in range(num_inner_splits) if len(d[x][i]) != 0}
        md = {len(x): [] for x in masks}
        for x in masks: md[len(x)].append(x)
        sortkeys = list(sorted(md))
        for x in md: md[x] = (sortkeys.index(x), {x: i for i, x in enumerate(sorted(md[x]))})
        return d, md
    def compute_correction(unitary, num_qbits, ident, outp_storage):
        pow2qb = 1 << num_qbits
        num_inner_splits = (pow2qb+320-1)//320
        d, md = UnitarySimulator.get_correction_masks(num_qbits, second=False)
        allrows = g.split_vectors(unitary, [1]*num_inner_splits*pow2qb*2)
        rows = g.concat([allrows[j*pow2qb*2+i*2] for i in d for j, x in enumerate(d[i]) if len(x) != 0], 0)
        ident = g.split_vectors(ident, [1]*ident.shape[0])
        with g.ResourceScope(name="correction", is_buffered=True, time=0) as pred:
            rows = g.mul(rows, g.concat([ident[md[len(x)][0]*min(256, pow2qb)+md[len(x)][1][frozenset(x)]] for i in d for x in d[i] if len(x) != 0], 0), alus=[0], time=0)
            rows = g.sum(rows, dims=[0], alus=[1])
            rows = rows.write(name="correction", storage_req=outp_storage)
        return rows
    def compute_correction2(unitary, num_qbits, ident, outp_storage):
        pow2qb = 1 << num_qbits
        num_inner_splits = (pow2qb+320-1)//320
        d, md = UnitarySimulator.get_correction_masks(num_qbits, second=True)
        allrows = g.split_vectors(unitary, [1]*num_inner_splits*pow2qb*2)
        rows = g.concat([allrows[j*pow2qb*2+i*2] for i in d for j, x in enumerate(d[i]) if len(x) != 0], 0)
        ident = g.split_vectors(ident, [1]*ident.shape[0])
        with g.ResourceScope(name="correction2", is_buffered=True, time=0) as pred:
            rows = g.mul(rows, g.concat([ident[md[len(x)][0]*min(256, pow2qb)+md[len(x)][1][frozenset(x)]] for i in d for x in d[i] if len(x) != 0], 0), alus=[4], output_streams=g.SG4[4], time=0)
            rows = g.sum(rows, dims=[0], alus=[5], output_streams=g.SG4[4])
            rows = rows.write(name="correction2", storage_req=outp_storage)
        return rows
    def compute_trace_real(unitary, num_qbits, ident, outp_storage): #using the copy could improve cycles
        pow2qb = 1 << num_qbits
        num_inner_splits = (pow2qb+320-1)//320
        #effective shape is address order so (num_inner_splits, pow2qb, 2, min(256, pow2qb))
        rows = g.concat_vectors([y
            for i, x in enumerate(g.split_vectors(unitary, [pow2qb*2]*num_inner_splits))
            for j, y in enumerate(g.split_vectors(x, [1]*(pow2qb*2)))
                if (j & 1) == 0 and j//2>=i*min(256, pow2qb) and j//2<(i+1)*min(256, pow2qb)], (pow2qb, min(256, pow2qb)))
        with g.ResourceScope(name="mask", is_buffered=True, time=0) as pred:
            rows = g.mul(rows, g.concat_vectors([ident]*num_inner_splits, (pow2qb, min(256, pow2qb))), time=0)
            rows = g.sum(rows, dims=[0])
            rows = rows.write(name="singledim", storage_req=outp_storage)
        #with g.ResourceScope(name="innerred", is_buffered=True, time=None, predecessors=[pred]) as pred:
        #    #rows = g.sum(g.concat_vectors([rows.reshape(pow2qb, min(256, pow2qb)), *([g.zeros((3, min(256, pow2qb)), dtype=g.float32, layout="-1, S12")]*pow2qb)], (4, pow2qb, min(256, pow2qb))).transpose(1,0,2), dims=None, time=0).write(name="trace", layout="-1, S4")
        #    rows = g.sum(g.concat_vectors([rows.reshape(1, min(256, pow2qb)), g.zeros((3, min(256, pow2qb)), dtype=g.float32, layout="-1, S12")], (4, min(256, pow2qb))), dims=[0,1], time=0).write(name="trace", layout="-1, S4, H1(W)")
        return rows
    def build_chain(num_qbits, max_gates, output_unitary=False):
        pow2qb = 1 << num_qbits
        debug = False
        pgm_pkg = g.ProgramPackage(name="us" + ("unit" if output_unitary else "") + str(num_qbits) + "-" + str(max_gates), output_dir="usiop", inspect_raw=debug, gen_vis_data=debug, check_stream_conflicts=debug, check_tensor_timing_conflicts=debug)
        num_inner_splits = (pow2qb+320-1)//320 #handle inner splits for >=9 qbits
        chainsize = 26 if num_qbits == 2 else (10 if num_qbits == 10 else (16 if num_qbits >= 5 else 20)) #min(max_gates, int(np.sqrt(6000*max_gates/(pow2qb*num_inner_splits/2)))) #6000*gates/chainsize == chainsize*pow2qb*num_inner_splits/2
        #if (chainsize & 1) != 0: chainsize += 1
        print("Number of qbits:", num_qbits, "Maximum gates:", max_gates, "Chain size:", chainsize)
        with pgm_pkg.create_program_context("init_us") as pcinitunitary:
            unitaryinit = g.input_tensor(shape=(pow2qb*2, pow2qb), dtype=g.float32, name="unitaryinit", layout="-1, H1(W), B1(1), A" + str(pow2qb*num_inner_splits) + "(0-" + str(pow2qb*num_inner_splits-1) + "), S8(0-8)") #get_slice8(WEST, 0, 7, 0)
            identaddr = (max_gates+1)//2*2; derivaddr = identaddr + 2
            gateidentderiv = [[g.from_data(np.repeat(np.eye(2, dtype=np.complex64).view(np.float32).flatten(), 320).reshape(2*2*2, 320), layout="-1, A2(" + str(identaddr) + "-" + str(identaddr+2-1) + "), S16(0-15), B1(0), H1(" + ("W" if hemi == WEST else "E") + ")"),
                g.zeros((2*2*2, 320), g.float32, layout="-1, A2(" + str(derivaddr) + "-" + str(derivaddr+2-1) + "), S16(0-15), B1(0), H1(" + ("W" if hemi == WEST else "E") + ")")] for hemi in (EAST, WEST)]
            identderivaddr = [g.from_data(np.array(([identaddr & 255, identaddr >> 8]+[0]*14)*20, dtype=np.uint8), name="identaddr0", layout=get_slice1(WEST, 0, 0) + ", A1(4085)"),
                              g.from_data(np.array(([(identaddr+1) & 255, (identaddr+1) >> 8]+[0]*14)*20, dtype=np.uint8), name="identaddr1", layout=get_slice1(WEST, 0, 0) + ", A1(4086)"),
                              g.from_data(np.array(([derivaddr & 255, derivaddr >> 8]+[0]*14)*20, dtype=np.uint8), name="derivaddr0", layout=get_slice1(WEST, 1, 0) + ", A1(4085)"),
                              g.from_data(np.array(([(derivaddr+1) & 255, (derivaddr+1) >> 8]+[0]*14)*20, dtype=np.uint8), name="derivaddr1", layout=get_slice1(WEST, 1, 0) + ", A1(4086)")]

            realgatemap = [[g.zeros((320,), g.uint32, layout=get_slice4(hemi, 17, 21, 1), name="realgatemap" + ("W" if hemi==WEST else "E")) for i in range(4)] for hemi in (EAST, WEST)]
            gatemap = [[g.zeros((320,), g.uint8, layout=get_slice1(hemi, 2, 0) + ", A1(4089)", name="gatemap0" + ("W" if hemi==WEST else "E")),
                        g.from_data(np.array(([1]+[0]*15)*20, dtype=np.uint8), layout=get_slice1(hemi, 2, 0) + ", A1(4090)", name="gatemap1" + ("W" if hemi==WEST else "E"))] for hemi in (EAST, WEST)]
            #gatemap = [[g.address_map(g.split_vectors(gates if hemi==EAST else othergates, [4]*((max_gates+1)//2*2))[0], np.array([0]*20), index_map_layout=get_slice4(hemi, 17, 21, 1)),
            #            g.address_map(g.split_vectors(gates if hemi==EAST else othergates, [4]*((max_gates+1)//2*2))[1], np.array([0]*20), index_map_layout=get_slice4(hemi, 17, 21, 1))] for hemi in (EAST, WEST)]
            
            gateinc = [g.from_data(np.array(([1*2]+[0]*15)*20, dtype=np.uint8), layout=get_slice1(hemi, 0, 0) + ", A1(4088)", name="gateinc" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)] #gather map is little endian byte order
            gateinc256 = [g.zeros((320,), layout=get_slice1(hemi, 1, 0) + ", A1(4088)", dtype=g.uint8, name="gateinc256" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]
            gateinccount = [g.from_data(np.array(([0, 1*2]+[0]*14)*20, dtype=np.uint8), layout=get_slice1(hemi, 2, 0) + ", A1(4088)", name="gateinccount" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]
            gateincmask = [g.from_data(np.array(([0, 1]+[0]*14)*20, dtype=np.uint8), layout=get_slice1(hemi, 3, 0) + ", A1(4088)", name="gateincmask" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]

            targetqbitdistro = [[g.zeros((320,), dtype=g.uint8, layout=get_slice1(hemi, 15 if i==1 else 35, 1) + ", A1(4095)", name="targetqbitdistro" + ("W" if hemi==WEST else "E")) for i in range(2)] for hemi in (EAST, WEST)]
            controlqbitdistro = [[g.zeros((320,), dtype=g.uint8, layout=get_slice1(hemi, 14 if i==1 else 34, 1) + ", A1(4095)", name="controlqbitdistro" + ("W" if hemi==WEST else "E")) for i in range(2)] for hemi in (EAST, WEST)]
            if num_qbits >= 9: hightcqdistro = [[g.zeros((320,), dtype=g.uint8, layout=get_slice1(hemi, 13 if i==1 else 33, 1) + ", A1(4095)", name="tcqdistro" + ("W" if hemi==WEST else "E")) for i in range(2)] for hemi in (EAST, WEST)]
            #derivatedistro = g.zeros((320,), dtype=g.float32, layout=get_slice4(WEST, 0, 3, 0) + ", A1(4092)", name="derivatedistro")

            idxmapsort, idxmapm1 = UnitarySimulator.idxmapgather(num_qbits)
            
            idxmapsort = (np.repeat(np.stack(idxmapsort), num_inner_splits, axis=1).reshape(num_qbits, -1, num_inner_splits, 2) + (np.arange(num_inner_splits)*pow2qb).reshape(1, 1, num_inner_splits, 1)).reshape(num_qbits, -1, 2) 
            idxmapm1 = (np.repeat(np.stack(idxmapm1), num_inner_splits, axis=1).reshape(num_qbits-1, -1, num_inner_splits, 2)*num_inner_splits + (np.arange(num_inner_splits)).reshape(1, 1, num_inner_splits, 1)).reshape(num_qbits-1, -1, 2)
            
            idxmapsort = np.stack(((idxmapsort & 255).astype(np.uint8), (idxmapsort >> 8).astype(np.uint8))).transpose(3, 2, 1, 0).reshape(2, -1, num_qbits*2)
            if num_qbits % 8 != 0: idxmapsort = np.concatenate((idxmapsort, np.zeros((2, idxmapsort.shape[-2], 2*(8-num_qbits % 8)), dtype=np.uint8)), axis=2)
            idxmapsort = np.repeat(idxmapsort, 20, axis=1).reshape(2, -1, 320)
            idxmapm1 = np.stack(((idxmapm1 & 255).astype(np.uint8), (idxmapm1 >> 8).astype(np.uint8))).transpose(3, 2, 1, 0).reshape(2, -1, (num_qbits-1)*2)
            if (num_qbits-1) % 8 != 0: idxmapm1 = np.concatenate((idxmapm1, np.zeros((2, idxmapm1.shape[-2], 2*(8-(num_qbits-1) % 8)), dtype=np.uint8)), axis=2)
            idxmapm1 = np.repeat(idxmapm1, 20, axis=1).reshape(2, -1, 320)
            
            idxmapsort = idxmapsort.reshape(2, -1, 20, 2 if num_qbits > 8 else 1, 16).transpose(0, 3, 1, 2, 4).reshape(2, -1, 320)
            idxmapm1 = idxmapm1.reshape(2, -1, 20, 2 if num_qbits > 9 else 1, 16).transpose(0, 3, 1, 2, 4).reshape(2, -1, 320)
            if num_qbits > 8: idxmapm1 = np.stack((idxmapm1, (idxmapm1.reshape(2, 2 if num_qbits > 9 else 1, -1, 20, 8, 2) + np.array((0, pow2qb*num_inner_splits//2//256), dtype=np.uint8)).reshape(2, -1, 320)), axis=1).reshape(2, -1, 320) #must address idxmapsort again for target qbits >=8
            
            tseldim = pow2qb*(2 if num_qbits > 8 else 1)*num_inner_splits//2
            cseldim = pow2qb*(2 if num_qbits > 9 else 1)*(2 if num_qbits > 8 else 1)*num_inner_splits//4
            targetqbitpairs0 = [g.from_data(idxmapsort[0,:], layout=get_slice1(hemi, 43, 0) + ", A" + str(tseldim) + "(0-" + str(tseldim-1) + ")", name="targetqbitpairs0" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]
            targetqbitpairs1 = [g.from_data(idxmapsort[1,:], layout=get_slice1(hemi, 42, 0) + ", A" + str(tseldim) + "(0-" + str(tseldim-1) + ")", name="targetqbitpairs1" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]
            controlqbitpairs0 = [g.from_data(idxmapm1[0,:], layout=get_slice1(hemi, 41, 0) + ", A" + str(cseldim) + "(0-" + str(cseldim-1) + ")", name="controlqbitpairs0" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]
            controlqbitpairs1 = [g.from_data(idxmapm1[1,:], layout=get_slice1(hemi, 41, 1) + ", A" + str(cseldim) + "(0-" + str(cseldim-1) + ")", name="controlqbitpairs1" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]
            
            if num_qbits >= 9: #selecting from 2 at 9 qbits or 4 at 10 or more qbits possibilities and making a shared distributor mapping
                controlqbithighsel = [g.from_data(np.repeat(np.concatenate((np.concatenate((np.arange(cseldim, dtype=np.uint16).reshape(2 if num_qbits==9 else 4, -1).T.reshape(2 if num_qbits==9 else 4, -1).view(np.uint8).reshape(-1, 4 if num_qbits==9 else 8), np.zeros((cseldim//(2 if num_qbits==9 else 4), 12 if num_qbits==9 else 8), dtype=np.uint8)), axis=1), np.concatenate((np.arange(4096, 4096+cseldim, dtype=np.uint16).reshape(2 if num_qbits==9 else 4, -1).T.reshape(2 if num_qbits==9 else 4, -1).view(np.uint8).reshape(-1, 4 if num_qbits==9 else 8), np.zeros((cseldim//(2 if num_qbits==9 else 4), 12 if num_qbits==9 else 8), dtype=np.uint8)), axis=1))), 20, axis=0).reshape(-1,320), layout=get_slice1(hemi, 37, 1), name="controlqbithighsel" + ("W" if hemi==WEST else "E")) for hemi in (EAST, WEST)]

            qbitinc = g.from_data(np.array(([1]+[0]*15)*20, dtype=np.uint8), layout=get_slice1(WEST, 0, 0) + ", A1(4095)", name="qbitinc") #gather map is little endian byte order
            qbitinc256 = g.zeros((320,), layout=get_slice1(WEST, 1, 0) + ", A1(4094)", dtype=g.uint8, name="qbitinc256")
            qbitinccount = g.from_data(np.array(([0, 1]+[0]*14)*20, dtype=np.uint8), layout=get_slice1(WEST, 2, 0) + ", A1(4093)", name="qbitinccount")

            qbitmap = g.zeros((320,), g.uint8, layout=get_slice1(WEST, 1, 0) + ", A1(4095)", name="qbitmap") #g.address_map(targetqbits, np.array([0]*20), index_map_layout=get_slice1(WEST, 1, 0) + ", A1(4095)")

            resetqbitmaporig = g.zeros((320,), g.uint8, layout=get_slice1(WEST, 2, 0) + ", A1(4094)", name="resetqbitmaporig") #g.address_map(targetqbits, np.array([0]*20), index_map_layout=get_slice1(WEST, 2, 0) + ", A1(4094)")
            resetgatemapsorig = [[g.zeros((320,), g.uint8, layout=get_slice1(hemi, 22, 1), name="resetgatemapsorig0" + ("W" if hemi==WEST else "E")),
                    g.from_data(np.array(([1]+[0]*15)*20, dtype=np.uint8), layout=get_slice1(hemi, 23, 1), name="resetgatemapsorig1" + ("W" if hemi==WEST else "E"))] for hemi in (EAST, WEST)]
            #resetgatemaps = [[g.address_map(g.split_vectors(gates if hemi==EAST else othergates, [4]*((max_gates+1)//2*2))[0].reinterpret(g.uint8).split(dim=1, num_splits=4)[0], np.array([0]*20), index_map_layout=get_slice1(hemi, 22, 1)),
            #        g.address_map(g.split_vectors(gates if hemi==EAST else othergates, [4]*((max_gates+1)//2*2))[1].reinterpret(g.uint8).split(dim=1, num_splits=4)[0], np.array([0]*20), index_map_layout=get_slice1(hemi, 23, 1))] for hemi in (EAST, WEST)]

            onepoint = g.full((320,), 1.0, dtype=g.float32, layout=get_slice4(WEST, 4, 7, 0) + ", A1(4095)", name="onepoint")
            zeropads = [g.zeros((1, 320), dtype=g.uint8, layout=get_slice1(WEST, 17+z, 1) + ", A1(4095)", name="zeropads" + str(z)) for z in range(3)]
            if not output_unitary:
                identmat = g.eye(min(256, pow2qb), dtype=g.float32, layout=get_slice4(WEST, 4, 7, 0).replace(", S4", ", A" + str(min(256, pow2qb)) + "(" + str(4095-min(256, pow2qb)) + "-4094), S4"), name="identmat")
                cormat1 = UnitarySimulator.get_correction_masks(num_qbits, False)[1]
                cormat2 = UnitarySimulator.get_correction_masks(num_qbits, True)[1]
                cormatlen1 = sum(len(cormat1[k][1]) for k in cormat1)
                cormatlen2 = sum(len(cormat2[k][1]) for k in cormat2)
                correctionmat1 = g.from_data(np.array([[1.0 if y in x else 0.0 for y in range(min(256, pow2qb))] for k in sorted(cormat1) for x in cormat1[k][1].keys()], dtype=np.float32), layout=get_slice4(WEST, 4, 7, 0).replace(", S4", ", A" + str(cormatlen1) + "(" + str(4095-cormatlen1-min(256, pow2qb)) + "-" + str(4094-min(256, pow2qb)) + "), S4"), name="correctionmat1")
                correctionmat2 = g.from_data(np.array([[1.0 if y in x else 0.0 for y in range(min(256, pow2qb))] for k in sorted(cormat2) for x in cormat2[k][1].keys()], dtype=np.float32), layout=get_slice4(EAST, 4, 7, 0).replace(", S4", ", A" + str(cormatlen2) + "(" + str(4095-cormatlen2) + "-4094), S4"), name="correctionmat2")
                outptrace = g.zeros((320,), dtype=g.float32, layout=get_slice4(WEST, 0, 3, 0) + ", A1(4091)", name="outptrace")
                outpcorrection = g.zeros((320,), dtype=g.float32, layout=get_slice4(WEST, 0, 3, 0) + ", A1(4084)", name="outpcorrection")
                outpcorrection2 = g.zeros((320,), dtype=g.float32, layout=get_slice4(WEST, 0, 3, 0) + ", A1(4083)", name="outpcorrection2")
                g.add_mem_constraints([identmat, correctionmat1], [identmat, correctionmat1, onepoint], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            else: cormatlen1 = 0
            resetzerosorig = g.zeros((320,), dtype=g.uint8, layout=get_slice1(WEST, 2, 0) + ", A1(4095)", name="resetzerosorig")
            distmaps = [g.from_data(np.array([[i] + [16]*319 for i in range(16)], dtype=np.uint8), name="distmaps", layout=get_slice1(hemi, 37, 1) + ", A16(4080-4095)") for hemi in (WEST, EAST)]
            lowmask = g.from_data(np.array(([7]*2+[0]*14)*20, dtype=np.uint8), name="lowmask", layout=get_slice1(WEST, 0, 0) + ", A1(4087)")
            midmask = g.from_data(np.array(([0x38]*2+[0]*14)*20, dtype=np.uint8), name="midmask", layout=get_slice1(WEST, 1, 0) + ", A1(4087)")
            if num_qbits >= 9: highmask = g.from_data(np.array(([0xC0]*2+[0]*14)*20, dtype=np.uint8), name="highmask", layout=get_slice1(WEST, 6, 0) + ", A1(" + str(4094-cormatlen1-min(256, pow2qb)) + ")")
            shl1 = g.full((320,), 1, name="shl1", dtype=g.uint8, layout=get_slice1(WEST, 4, 0) + ", A1(" + str(4094-cormatlen1-min(256, pow2qb)) + ")")
            shr2 = g.full((320,), 2, name="shr2", dtype=g.uint8, layout=get_slice1(WEST, 3, 0) + ", A1(4087)")
            if num_qbits >= 9: shr5 = g.full((320,), 5, name="shr5", dtype=g.uint8, layout=get_slice1(WEST, 2, 0) + ", A1(4087)")
            adjmap = g.from_data(np.array(([0, 1] + [16]*14)*20, dtype=np.uint8), name="adjmap", layout=get_slice1(WEST, 5, 0) + ", A1(" + str(4094-cormatlen1-min(256, pow2qb)) + ")")
            
        with pgm_pkg.create_program_context("init_gates") as pcinit:
            us = UnitarySimulator(num_qbits)
            #unitary = g.input_tensor(shape=(pow2qb*2, pow2qb), dtype=g.float32, name="unitary", layout=get_slice8(WEST, s8range[0], s8range[-1], 0))
            #physical shapes for 9, 10, 11 qbits are (2, 1024, 4, (256, 256)), (4, 2048, 4, (256, 256, 256, 256)), (7, 4096, 4, (304, 304, 304, 304, 304, 304, 224))
            g.reserve_tensor(pcinitunitary, pcinit, unitaryinit)
            unitaryinitctxt = g.from_addresses(np.array(unitaryinit.storage_request.addresses.reshape(-1, g.float32.size), dtype=object), min(256, pow2qb), g.float32, "unitaryinitreset")
            unitaryinitctxt = g.concat_vectors([g.concat_inner_splits(g.split_vectors(x, [1]*num_inner_splits)) for x in g.split_vectors(unitaryinitctxt.reshape(num_inner_splits, pow2qb*2, min(256, pow2qb)).transpose(1, 0, 2), [num_inner_splits]*(pow2qb*2))], (pow2qb*2, pow2qb))
            with g.ResourceScope(name="makecopy", is_buffered=True, time=0) as pred:
                unitary, copy, otherunitary, othercopy = us.copymatrix(unitaryinitctxt)
            #gatescomb = g.input_tensor(shape=(max_gates+1)//2*2, 2*2*2, pow2qb), dtype=g.float32, name="gate", layout="-1, H2, S16(" + str(min(slices)) + "-" + str(max(slices)) + ")")
            #gates = g.input_tensor(shape=((max_gates+1)//2, 2*2*2, min(256, pow2qb)), dtype=g.float32, name="gate", layout="-1, A" + str((max_gates+1)//2*2) + "(0-" + str((max_gates+1)//2*2-1) + "), S16(0-15), B1(0), H1(E)") #get_slice16(EAST, list(range(16)), 0)) #, broadcast=True)
            #othergates = g.input_tensor(shape=((max_gates+1)//2, 2*2*2, min(256, pow2qb)), dtype=g.float32, name="othergate", layout="-1, A" + str((max_gates+1)//2*2) + "(0-" + str((max_gates+1)//2*2-1) + "), S16(0-15), B1(0), H1(W)") #get_slice16(WEST, list(range(16)), 0) #, broadcast=True)
            
            num_inner_gates = (2*2*2*((max_gates+1)//2)+320-1)//320 #(num_inner_gates, 320) or (num_inner_gates, 20, 16) order
            num_inner_qbits = (max_gates+320-1)//320
            gatespack = g.input_tensor(shape=(num_inner_gates*320,), dtype=g.float32, name="gate", layout=get_slice4(EAST, 32, 35, 1)) 
            othergatespack = g.input_tensor(shape=(num_inner_gates*320,), dtype=g.float32, name="othergate", layout=get_slice4(WEST, 32, 35, 1))
            dmaps = [tensor.shared_memory_tensor(mem_tensor=distmaps[hemi], name="distmaps" + str(hemi)) for hemi in (WEST, EAST)]
            with g.ResourceScope(name="initgates", is_buffered=True, time=0) as pred:
                for hemi in (WEST, EAST):
                    outp = []
                    def writefn(st):
                        i = len(outp) // num_inner_gates #order will be (20, num_inner_gates, 4, 4)
                        for j, x in enumerate(g.split_inner_splits(st)):
                            outp.append(x.write(name="initgate" + str(i) + str(j), storage_req=tensor.create_storage_request(layout="-1, A4(" + str(i*4+80*j) + "-" + str(i*4+80*j+3) + "), S16(0-15), B1(0), H1(" + ("W" if hemi==WEST else "E") + ")")))
                    temp_store = tensor.create_storage_request(layout=("-1, H1(" + ("W" if hemi==WEST else "E") + "), S4(39,40,42,43), B1(1)" if num_qbits == 10 else get_slice4(hemi, 40, 43, 1)).replace(", S4", ", A" + str(num_inner_gates) + "(" + str(4096-num_inner_gates) + "-" + "4095), S4"))
                    pack = othergatespack if hemi==WEST else gatespack
                    UnitarySimulator.unpack_broadcast(pack, dmaps[hemi], temp_store, num_inner_gates, hemi, 4, writefn)
                    g.add_mem_constraints(outp, outp, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                    if hemi == WEST: othergates = g.split(g.concat(outp, 0).reshape(20, num_inner_gates, 16, 320).transpose(1, 0, 2, 3).reshape(20*num_inner_gates*16, 320), [(max_gates+1)//2*2*2*2, num_inner_gates*320-(max_gates+1)//2*2*2*2])[0]
                    else: gates = g.split(g.concat(outp, 0).reshape(20, num_inner_gates, 16, 320).transpose(1, 0, 2, 3).reshape(20*num_inner_gates*16, 320), [(max_gates+1)//2*2*2*2, num_inner_gates*320-(max_gates+1)//2*2*2*2])[0]
            qbitinfo = g.input_tensor(shape=(num_inner_qbits*320,), dtype=g.uint16, name="qbits", layout=get_slice2(WEST, 39, 40, 1))
            with g.ResourceScope(name="initqbits", is_buffered=True, predecessors=[pred], time=None) as pred:
                outpt, outpc, outpd, outph = [], [], [], []
                lmask = tensor.shared_memory_tensor(mem_tensor=lowmask, name=lowmask.name + "init")
                mmask = tensor.shared_memory_tensor(mem_tensor=midmask, name=midmask.name + "init")
                if num_qbits >= 9: hmask = tensor.shared_memory_tensor(mem_tensor=highmask, name=highmask.name + "init")
                s1 = tensor.shared_memory_tensor(mem_tensor=shl1, name=shl1.name + "init")
                s2 = tensor.shared_memory_tensor(mem_tensor=shr2, name=shr2.name + "init")
                if num_qbits >= 9: s5 = tensor.shared_memory_tensor(mem_tensor=shr5, name=shr5.name + "init")
                amap = tensor.shared_memory_tensor(mem_tensor=adjmap, name=adjmap.name + "init")
                def writefn(st):
                    #3 target qbits, 3 control qbits with one subtracted if > target_qbit, 0 if control qbit 
                    #target qbit formula: & 0x7 mask, <<1, +1/16 (3 ALU)
                    #control qbit formula: & 0x3f mask, >>2 (>>3, <<1), +1/16 (3 ALU)
                    #derivate formula: write only (0 for target, 1 for control, 2 for derivate)
                    #hightcqbit formula: & 0xC0 mask >>5 (>>6, <<1), +1/16 (3 ALU)
                    i = len(outpt) // num_inner_qbits
                    for j, x in enumerate(g.split_inner_splits(st)):
                        x = x.reinterpret(g.uint8).split(num_splits=2, dim=-2)
                        with g.ResourceScope(name="innersplitsalus", is_buffered=False, time=0):
                            am = amap.read(streams=g.SG1[7*4])
                            outpt.append(x[0].bitwise_and(lmask.read(streams=g.SG1[2*4]), alus=[0], output_streams=g.SG4[2]).left_shift(s1.read(streams=g.SG1[5*4]), alus=[5], output_streams=g.SG4[4]).add(am, alus=[10])
                                .write(name="target_qbits" + str(i) + str(j), storage_req=tensor.create_storage_request(layout=get_slice1(WEST, 37, 0) + ", A16(" + str(i*16+320*j) + "-" + str(i*16+320*j+15) + ")")))
                            outpc.append(x[0].bitwise_and(mmask.read(streams=g.SG1[6*4]), alus=[12], output_streams=g.SG4[6]).right_shift(s2.read(streams=g.SG1[4*4]), alus=[9], output_streams=g.SG4[6]).add(am, alus=[14])
                                .write(name="control_qbits" + str(i) + str(j), storage_req=tensor.create_storage_request(layout=get_slice1(WEST, 36, 0) + ", A16(" + str(i*16+320*j) + "-" + str(i*16+320*j+15) + ")")))
                            outpd.append(x[1].write(name="derivates" + str(i) + str(j), storage_req=tensor.create_storage_request(layout=get_slice1(WEST, 39, 0) + ", A16(" + str(i*16+320*j) + "-" + str(i*16+320*j+15) + ")")))
                            if num_qbits >= 9: outph.append(x[0].bitwise_and(hmask.read(streams=g.SG1[3*4]), alus=[1], output_streams=g.SG4[3]).right_shift(s5.read(streams=g.SG1[1*4]), alus=[2], output_streams=g.SG4[1]).add(am, alus=[15])
                                .write(name="high_tcqbits" + str(i) + str(j), storage_req=tensor.create_storage_request(layout=get_slice1(EAST, 36, 0) + ", A16(" + str(i*16+320*j) + "-" + str(i*16+320*j+15) + ")")))
                temp_store = tensor.create_storage_request(layout=get_slice2(hemi, 42, 43, 1).replace(", S2", ", A" + str(num_inner_qbits) + "(" + str(4096-num_inner_qbits-num_inner_gates) + "-" + str(4095-num_inner_gates) + "), S2"))
                UnitarySimulator.unpack_broadcast(qbitinfo, dmaps[WEST], temp_store, num_inner_qbits, WEST, 2, writefn)
                g.add_mem_constraints(outpt, outpt, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                g.add_mem_constraints(outpc, outpc, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                g.add_mem_constraints(outpd, outpd, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                if num_qbits >= 9: g.add_mem_constraints(outph, outph, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                targetqbits = g.split(g.concat(outpt, 0).reshape(20, num_inner_qbits, 16, 320).transpose(1, 0, 2, 3).reshape(20*num_inner_qbits*16, 320), [max_gates, num_inner_qbits*320-max_gates])[0]
                controlqbits = g.split(g.concat(outpc, 0).reshape(20, num_inner_qbits, 16, 320).transpose(1, 0, 2, 3).reshape(20*num_inner_qbits*16, 320), [max_gates, num_inner_qbits*320-max_gates])[0]
                derivates = g.split(g.concat(outpd, 0).reshape(20, num_inner_qbits, 16, 320).transpose(1, 0, 2, 3).reshape(20*num_inner_qbits*16, 320), [max_gates, num_inner_qbits*320-max_gates])[0]
                if num_qbits >= 9: hightcqbits = g.split(g.concat(outph, 0).reshape(20, num_inner_qbits, 16, 320).transpose(1, 0, 2, 3).reshape(20*num_inner_qbits*16, 320), [max_gates, num_inner_qbits*320-max_gates])[0]
            #targetqbits = g.input_tensor(shape=(max_gates, 320), dtype=g.uint8, name="target_qbits", layout=get_slice1(WEST, 37, 0) + ", A" + str(max_gates) + "(0-" + str(max_gates-1) + ")")
            #controlqbits = g.input_tensor(shape=(max_gates, 320), dtype=g.uint8, name="control_qbits", layout=get_slice1(WEST, 36, 0) + ", A" + str(max_gates) + "(0-" + str(max_gates-1) + ")")
            #derivates = g.input_tensor(shape=(max_gates, 320), dtype=g.uint8, name="derivates", layout=get_slice1(WEST, 39, 0) + ", A" + str(max_gates) + "(0-" + str(max_gates-1) + ")")
            #if num_qbits >= 9: hightcqbits = g.input_tensor(shape=(max_gates, 320), dtype=g.uint8, name="high_tcqbits", layout=get_slice1(EAST, 36, 0) + ", A" + str(max_gates) + "(0-" + str(max_gates-1) + ")")

            for x in (gateinc, gateinccount, gateincmask, targetqbitpairs0, targetqbitpairs1, controlqbitpairs0, controlqbitpairs1, zeropads) + ((controlqbithighsel,) if num_qbits >= 9 else ()):
                for y in x: tensor.shared_memory_tensor(mem_tensor=y, name=y.name + "init")
            for x in (qbitinc, qbitinccount, onepoint) + (() if output_unitary else (identmat,)): tensor.shared_memory_tensor(mem_tensor=x, name=x.name + "fin")        
            resetzeros = tensor.shared_memory_tensor(mem_tensor=resetzerosorig, name="resetzeros")
            resetqbitmap = tensor.shared_memory_tensor(mem_tensor=resetqbitmaporig, name="resetqbitmap")
            resetgatemaps = [[tensor.shared_memory_tensor(mem_tensor=resetgatemapsorig[hemi][i], name="resetgatemaps" + str(i) + ("W" if hemi==WEST else "E")) for i in range(2)] for hemi in (EAST, WEST)]
            with g.ResourceScope(name="resetgathercounts", is_buffered=True, time=None, predecessors=[pred]) as pred:
                #must reset ginc256, gatemap, qbitinc256, qbitmap or GFAULTs will occur due to bad addresses gathered/scattered
                tsrs = [resetzeros.read(streams=g.SG1[0], time=1).write(storage_req=qbitinc256.storage_request),
                    resetqbitmap.read(streams=g.SG1[0], time=2).write(storage_req=qbitmap.storage_request)]
                z = resetzeros.read(streams=g.SG1[0], time=0)
                tsrs += [z.write(storage_req=gateinc256[i].storage_request) for i in range(2)]
                tsrs += [g.concat([resetgatemaps[hemi][i].read(streams=g.SG1[0], time=i*4)]*4, 0).write(storage_req=gatemap[hemi][i].storage_request) for hemi in (EAST, WEST) for i in range(2)]
                g.add_mem_constraints(tsrs, tsrs, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            g.add_mem_constraints(gateinc + gateinc256 + gateinccount + [resetzeros, resetqbitmap], [gates, othergates, resetzeros, resetqbitmap], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        
        #for reversedir in (False, True):
        #for target_qbit, control_qbit in ((0, None), (0, 1)) + (((8, None), (8, 1)) if num_qbits >= 9 else ()) + (((0, 9), (8, 9)) if num_qbits >= 10 else ()):
        #for target_qbit, control_qbit in ((0, 1),): # + (((8, 1),) if num_qbits >= 9 else ()) + (((0, 9), (8, 9)) if num_qbits >= 10 else ()):
        suffix = "universal" #("rev" if reversedir else "") + str(target_qbit) + "_" + str(control_qbit)
        with pgm_pkg.create_program_context("us_gate"+suffix) as pc:
            target_qbit, control_qbit = 0, 1
            #if not reversedir and target_qbit == 0 and control_qbit is None: print(gatemap[0].data, gatemap[1].data)            
            g.reserve_tensor(pcinitunitary, pcinit, unitaryinit)
            g.reserve_tensor(pcinit, pc, unitary)
            g.reserve_tensor(pcinit, pc, otherunitary)
            g.reserve_tensor(pcinit, pc, copy)
            g.reserve_tensor(pcinit, pc, othercopy)
            g.reserve_tensor(pcinit, pc, gates)
            g.reserve_tensor(pcinit, pc, othergates)
            g.reserve_tensor(pcinit, pc, targetqbits)
            g.reserve_tensor(pcinit, pc, controlqbits)
            if num_qbits >= 9: g.reserve_tensor(pcinit, pc, hightcqbits)
            g.reserve_tensor(pcinit, pc, derivates)
            onep = tensor.shared_memory_tensor(mem_tensor=onepoint, name="onep"+suffix)
            zs = [tensor.shared_memory_tensor(mem_tensor=zeropads[i], name="zs"+str(i)+suffix) for i in range(len(zeropads))]
            gmap = [[tensor.shared_memory_tensor(mem_tensor=gatemap[reversedir][i], name="gatemap" + str(reversedir) + str(i) + suffix) for i in range(2)] for reversedir in range(2)]
            realgmap = [[tensor.shared_memory_tensor(mem_tensor=realgatemap[reversedir][i], name="realgatemap" + str(reversedir) + str(i) + suffix) for i in range(4)] for reversedir in range(2)]
            ginc = [tensor.shared_memory_tensor(mem_tensor=gateinc[i], name="gateinc" + str(i) + suffix) for i in range(2)]
            ginc256 = [tensor.shared_memory_tensor(mem_tensor=gateinc256[i], name="gateinc256" + str(i) + suffix) for i in range(2)]
            ginccount = [tensor.shared_memory_tensor(mem_tensor=gateinccount[i], name="gateinccount" + str(i) +suffix) for i in range(2)]
            gincmask = [tensor.shared_memory_tensor(mem_tensor=gateincmask[i], name="gateincmask" + str(i) + suffix) for i in range(2)]
            tqbitdistro = [[tensor.shared_memory_tensor(mem_tensor=targetqbitdistro[i][(i+reversedir) % 2], name="tqbitdistro" + str(reversedir) + str(i) + suffix) for i in range(2)] for reversedir in range(2)]
            tqbitpairs0 = [tensor.shared_memory_tensor(mem_tensor=targetqbitpairs0[i], name="tqbitpairs0" + str(i) + suffix) for i in range(2)]
            tqbitpairs1 = [tensor.shared_memory_tensor(mem_tensor=targetqbitpairs1[i], name="tqbitpairs1" + str(i) + suffix) for i in range(2)]
            if not control_qbit is None:
                cqbitdistro = [[tensor.shared_memory_tensor(mem_tensor=controlqbitdistro[i][(i+reversedir) % 2], name="cqbitdistro" + str(reversedir) + str(i) + suffix) for i in range(2)] for reversedir in range(2)]
                cqbitpairs0 = [tensor.shared_memory_tensor(mem_tensor=controlqbitpairs0[i], name="cqbitpairs0" + str(i) + suffix) for i in range(2)]
                cqbitpairs1 = [tensor.shared_memory_tensor(mem_tensor=controlqbitpairs1[i], name="cqbitpairs1" + str(i) + suffix) for i in range(2)]
                if num_qbits >= 9:
                    tcqbitdistro = [[tensor.shared_memory_tensor(mem_tensor=hightcqdistro[i][(i+reversedir) % 2], name="tcqbitdistro" + str(reversedir) + str(i) + suffix) for i in range(2)] for reversedir in range(2)]
                    cqbithighsel = [tensor.shared_memory_tensor(mem_tensor=controlqbithighsel[i], name="controlqbithighsel" + str(i) + suffix) for i in range(2)]
            else:
                tensor.shared_memory_tensor(mem_tensor=controlqbitpairs0[0], name="cqbitpairs0" + str(0) + suffix)
                tensor.shared_memory_tensor(mem_tensor=controlqbitpairs1[0], name="cqbitpairs1" + str(0) + suffix)
                tensor.shared_memory_tensor(mem_tensor=controlqbitpairs0[1], name="cqbitpairs0" + str(1) + suffix)
                tensor.shared_memory_tensor(mem_tensor=controlqbitpairs1[1], name="cqbitpairs1" + str(1) + suffix)
            g.add_mem_constraints([ginc[0], ginc256[0], ginccount[0]], [othergates if 0 else gates], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            g.add_mem_constraints([ginc[1], ginc256[1], ginccount[1]], [othergates if 1 else gates], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            
            qmap = tensor.shared_memory_tensor(mem_tensor=qbitmap, name="qmap" + suffix)
            qinc = tensor.shared_memory_tensor(mem_tensor=qbitinc, name="qinc" + suffix)
            qinc256 = tensor.shared_memory_tensor(mem_tensor=qbitinc256, name="qinc256" + suffix)
            qinccount = tensor.shared_memory_tensor(mem_tensor=qbitinccount, name="qinccount" + suffix)

            for x in resetgatemapsorig:
                for y in x: tensor.shared_memory_tensor(mem_tensor=y, name=y.name + "calc")
            for x in (resetqbitmaporig, resetzerosorig) + (() if output_unitary else (identmat,)): tensor.shared_memory_tensor(mem_tensor=x, name=x.name + "calc")
            
            unitaryctxt = [g.from_addresses(np.array((otherunitary if reversedir else unitary).storage_request.addresses.reshape(-1, g.float32.size), dtype=object), 320, g.float32, "unitary" + str(reversedir) + suffix) for reversedir in range(2)]
            copyctxt = [g.from_addresses(np.array((othercopy if reversedir else copy).storage_request.addresses.reshape(-1, g.float32.size), dtype=object), 320, g.float32, "copy" + str(reversedir) + suffix) for reversedir in range(2)]
            gatesctxt = [g.from_addresses(np.array((othergates if reversedir else gates).storage_request.addresses.reshape(-1, g.float32.size), dtype=object), 320, g.float32, "gates" + str(reversedir) + suffix) for reversedir in range(2)]
            tqbits = g.from_addresses(np.array(targetqbits.storage_request.addresses.reshape(-1, g.uint8.size), dtype=object), 320, g.uint8, "targetqbits" + suffix)
            cqbits = g.from_addresses(np.array(controlqbits.storage_request.addresses.reshape(-1, g.uint8.size), dtype=object), 320, g.uint8, "controlqbits" + suffix)
            if num_qbits >= 9: htcqbits = g.from_addresses(np.array(hightcqbits.storage_request.addresses.reshape(-1, g.uint8.size), dtype=object), 320, g.uint8, "hightcqbits" + suffix)
            derivs = g.from_addresses(np.array(derivates.storage_request.addresses.reshape(-1, g.uint8.size), dtype=object), 320, g.uint8, "derivates" + suffix)
            idaddr = [tensor.shared_memory_tensor(mem_tensor=identderivaddr[i], name=identderivaddr[i].name + suffix) for i in range(4)]
            one, two = tensor.shared_memory_tensor(mem_tensor=shl1, name=shl1.name + suffix), tensor.shared_memory_tensor(mem_tensor=shr2, name=shr2.name + suffix)
            pred, reversedir = None, False
            for c in range(chainsize):
                with g.ResourceScope(name="setgatherdistros" + str(c), is_buffered=True, time=0 if pred is None else None, predecessors=None if pred is None else [pred]) as pred:
                    qmapW_st = g.split(g.stack([qmap]*(1+1+(1+1 if not control_qbit is None else 0)+8), 0).read(streams=g.SG1_W[0], time=0), splits=[1, 1] + ([1, 1] if not control_qbit is None else []) + [8])
                    if not control_qbit is None and num_qbits >=9: qmapE_st = g.split(g.stack([qmap]*(1+1), 0).read(streams=g.SG1_E[0], time=1+1+(1+1)+8), splits=[1, 1])
                    for i in range(2):
                        g.mem_gather(tqbits, qmapW_st[0+i], output_streams=[g.SG1_E[1]]).write(name="targetqbitdistro" + str(i) + suffix, storage_req=tqbitdistro[reversedir][i].storage_request)
                        if not control_qbit is None:
                            g.mem_gather(cqbits, qmapW_st[2+i], output_streams=[g.SG1_E[2]]).write(name="controlqbitdistro" + str(i) + suffix, storage_req=cqbitdistro[reversedir][i].storage_request)
                            if num_qbits >= 9: g.mem_gather(htcqbits, qmapE_st[0+i], output_streams=[g.SG1_W[3]]).write(name="hightcqbitdistro" + str(i) + suffix, storage_req=tcqbitdistro[reversedir][i].storage_request)

                    d = g.mem_gather(derivs, qmapW_st[-1], output_streams=[g.SG1_E[12]])
                    updmap = g.split(g.bitwise_xor(g.mask(g.equal(d, two.read(streams=g.SG1[2*4]), alus=[5], output_streams=g.SG4_E[4]), g.stack([idaddr[2]]*4+[idaddr[3]]*4, 0).read(streams=g.SG1[7*4]), alus=[10], output_streams=g.SG4[6]).vxm_identity(alus=[13], output_streams=g.SG4[6]),
                        g.bitwise_xor(g.mask(g.equal(d, one.read(streams=g.SG1[0]), alus=[0], output_streams=g.SG4_E[0]), g.stack([idaddr[0]]*4+[idaddr[1]]*4, 0), alus=[1], output_streams=g.SG4[2]), 
                        g.mask_bar(d.vxm_identity(alus=[4], output_streams=g.SG4[4]), g.stack([gmap[reversedir][0]]*4+[gmap[reversedir][1]]*4, 0), alus=[9], output_streams=g.SG4[5]), alus=[6], output_streams=g.SG4[5]), alus=[11]), 0, num_splits=2)          
                    #updmap = g.split(g.bitwise_xor(g.mem_gather(derivs, qmapW_st[-1], output_streams=[g.SG1_E[12]]), g.stack([gmap[reversedir][0]]*4+[gmap[reversedir][1]]*4, 0)), 0, num_splits=2)
                    for i in range(2): updmap[i].reinterpret(g.uint32).write(name="realgatemap" + str(i) + suffix, storage_req=realgmap[reversedir][i].storage_request)
                    
                    updmap = g.split(g.stack([gmap[reversedir][0]]*4+[gmap[reversedir][1]]*4, 0).read(streams=g.SG1[8], time=4*i), 0, num_splits=2)
                    for i in range(2): updmap[i].reinterpret(g.uint32).write(name="realgatemap" + str(i+2) + suffix, storage_req=realgmap[reversedir][i+2].storage_request)
                tcmap = [list(reversed(x)) if reversedir else x for x in ((tqbitdistro[reversedir], tqbitpairs0, tqbitpairs1, cqbitdistro[reversedir], cqbitpairs0, cqbitpairs1) + ((tcqbitdistro[reversedir], cqbithighsel) if num_qbits >= 9 else ()) if not control_qbit is None else (tqbitdistro[reversedir], tqbitpairs0, tqbitpairs1))]
                with g.ResourceScope(name="rungate" + str(c), is_buffered=True, time=None, predecessors=[pred]) as pred:
                    newus = UnitarySimulator(num_qbits, reversedir, us)
                    newus.build(unitaryctxt[reversedir], copyctxt[reversedir], target_qbit, control_qbit, gatesctxt[reversedir], realgmap[reversedir], tcmap, None, inittime=c)
                with g.ResourceScope(name="incgate" + str(c), is_buffered=True, time=None, predecessors=[pred]) as pred:
                    updinc = g.stack([ginc256[reversedir]]*2, 0).add(g.stack([ginccount[reversedir]]*len(gmap[reversedir]), 0), time=0, alus=[3 if reversedir else 0], overflow_mode=g.OverflowMode.MODULAR)
                    updmap = g.split(g.stack(gmap[reversedir], 0).add(g.stack([ginc[reversedir]]*2, 0), alus=[7 if reversedir else 4], overflow_mode=g.OverflowMode.MODULAR).add(g.mask_bar(updinc, g.stack([gincmask[reversedir]]*2, 0))), 0, num_splits=2)
                    for i in range(2):
                        updmap[i].write(storage_req=gmap[reversedir][i].storage_request, name="nextgatemap" + str(i) + suffix)
                    g.split(updinc, 0, num_splits=2)[0].vxm_identity().write(storage_req=ginc256[reversedir].storage_request, name="nextgateinc256" + suffix)
                with g.ResourceScope(name="incqbit" + str(c), is_buffered=True, time=None, predecessors=[pred]) as pred:
                    updinc = qinc256.add(qinccount, time=0, alus=[0], overflow_mode=g.OverflowMode.MODULAR)
                    qmap.add(qinc, alus=[4], overflow_mode=g.OverflowMode.MODULAR).add(g.mask_bar(updinc, gincmask[reversedir])).write(storage_req=qmap.storage_request, name="nextqmap" + suffix)
                    updinc.vxm_identity().write(storage_req=qinc256.storage_request, name="nextqinc256" + suffix)
                reversedir = not reversedir                
            #g.from_addresses(np.array([g.Address('W', 35, x) for x in range(8192)]).reshape(8192, 1), 320, g.uint8, "sliceW").set_program_output()
            #g.from_addresses(np.array([g.Address('E', 35, x) for x in range(8192)]).reshape(8192, 1), 320, g.uint8, "sliceE").set_program_output()
            
        #must validate all addresses are contiguous, and gather/scatter addresses are all on 0-address alignment by checking storage requests, should likely malloc to avoid
        assert {(x.hemi, x.slice, x.offset) for x in unitary.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST, x, i) for x in s8range for i in range(pow2qb*num_inner_splits)}
        assert {(x.hemi, x.slice, x.offset) for x in otherunitary.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.EAST, x, i) for x in s8range for i in range(pow2qb*num_inner_splits)}
        assert {(x.hemi, x.slice, x.offset) for x in copy.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST, x, i) for x in s8range2 for i in range(pow2qb*num_inner_splits)}
        assert {(x.hemi, x.slice, x.offset) for x in othercopy.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.EAST, x, i) for x in s8range2 for i in range(pow2qb*num_inner_splits)}
        
        assert {(x.hemi, x.slice, x.offset) for x in targetqbits.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST, x, i) for x in (37,) for i in range(max_gates)}
        assert {(x.hemi, x.slice, x.offset) for x in controlqbits.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST, x, i) for x in (36,) for i in range(max_gates)}
        assert {(x.hemi, x.slice, x.offset) for x in derivates.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST, x, i) for x in (39,) for i in range(max_gates)}
        assert {(x.hemi, x.slice, x.offset) for x in gates.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.EAST, x, i) for x in range(16) for i in range((max_gates+1)//2*2)}
        assert {(x.hemi, x.slice, x.offset) for x in othergates.storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST, x, i) for x in range(16) for i in range((max_gates+1)//2*2)}
        for i, hemi in enumerate((EAST, WEST)):
            assert {(x.hemi, x.slice, x.offset) for x in targetqbitpairs0[i].storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST if hemi==WEST else g.Hemisphere.EAST, x, i) for x in (43,) for i in range(pow2qb//2*num_inner_splits*(2 if num_qbits > 8 else 1))}
            assert {(x.hemi, x.slice, x.offset) for x in targetqbitpairs1[i].storage_request.addresses.reshape(-1).tolist()} == {(g.Hemisphere.WEST if hemi==WEST else g.Hemisphere.EAST, x, i) for x in (42,) for i in range(pow2qb//2*num_inner_splits*(2 if num_qbits > 8 else 1))}
        
        #we return in raw address format, so the inner splits will be on the outer dimension!
        with pgm_pkg.create_program_context("final_us") as pcfinal:
            g.reserve_tensor(pcinitunitary, pcfinal, unitaryinit)
            g.reserve_tensor(pcinit, pcfinal, unitary)
            for x in (gateinc, gateinccount, gateincmask, targetqbitpairs0, targetqbitpairs1, controlqbitpairs0, controlqbitpairs1, zeropads):
                for y in x: tensor.shared_memory_tensor(mem_tensor=y, name=y.name + "fin")
            for x in resetgatemapsorig:
                for y in x: tensor.shared_memory_tensor(mem_tensor=y, name=y.name + "fin")
            for x in (qbitinc, qbitinccount, resetqbitmaporig, onepoint, resetzerosorig): tensor.shared_memory_tensor(mem_tensor=x, name=x.name + "fin")        
            unitaryres = g.from_addresses(np.array(unitary.storage_request.addresses.reshape(-1, g.float32.size), dtype=object), min(256, pow2qb), g.float32, "unitaryfin")
            if not output_unitary:
                g.reserve_tensor(pcinit, pcfinal, copy)
                copyres = g.from_addresses(np.array(copy.storage_request.addresses.reshape(-1, g.float32.size), dtype=object), min(256, pow2qb), g.float32, "copyfin")
                ident = tensor.shared_memory_tensor(mem_tensor=identmat, name="ident")
                cm1 = tensor.shared_memory_tensor(mem_tensor=correctionmat1, name="cm1")
                cm2 = tensor.shared_memory_tensor(mem_tensor=correctionmat2, name="cm2")
                with g.ResourceScope(name="trace", is_buffered=True, time=0) as pred:
                    trace = UnitarySimulator.compute_trace_real(unitaryres, num_qbits, ident,
                            tensor.shared_memory_tensor(mem_tensor=outptrace, name="outp").storage_request)
                with g.ResourceScope(name="corrections", is_buffered=True, time=None, predecessors=[pred]) as pred:
                    correction = UnitarySimulator.compute_correction(unitaryres, num_qbits, cm1,
                            tensor.shared_memory_tensor(mem_tensor=outpcorrection, name="outpc").storage_request)
                with g.ResourceScope(name="corrections2", is_buffered=True, time=1):
                    unitaryres = g.stack(
                        [trace, correction,
                        UnitarySimulator.compute_correction2(copyres, num_qbits, cm2,
                            tensor.shared_memory_tensor(mem_tensor=outpcorrection2, name="outpc2").storage_request)], 0)
                    unitaryres.name = "traces"                        
            unitaryres.set_program_output()
        """
        with pgm_pkg.create_program_context("finalrev_us") as pcfinal:
            g.reserve_tensor(pcinitunitary, pcfinal, unitaryinit)
            g.reserve_tensor(pcinit, pcfinal, otherunitary)
            for x in (gateinc, gateinccount, gateincmask, targetqbitpairs0, targetqbitpairs1, controlqbitpairs0, controlqbitpairs1, zeropads):
                for y in x: tensor.shared_memory_tensor(mem_tensor=y, name=y.name + "revfin")
            for x in resetgatemapsorig:
                for y in x: tensor.shared_memory_tensor(mem_tensor=y, name=y.name + "revfin")
            for x in (qbitinc, qbitinccount, resetqbitmaporig, onepoint, resetzerosorig): tensor.shared_memory_tensor(mem_tensor=x, name=x.name + "revfin")        
            unitaryrevres = g.from_addresses(np.array(otherunitary.storage_request.addresses.reshape(-1, g.float32.size), dtype=object), min(256, pow2qb), g.float32, "unitaryrevfin")
            if not output_unitary:
                unitaryrevres = UnitarySimulator.compute_trace_real(unitaryrevres, num_qbits,
                    tensor.shared_memory_tensor(mem_tensor=identmat, name="ident"), 
                    tensor.shared_memory_tensor(mem_tensor=outptrace, name="outp").storage_request) 
            unitaryrevres.set_program_output()
        """
        print_utils.infoc("\nAssembling model ...")
        iops = pgm_pkg.assemble(auto_agt_dim=3)
        return {"iop": iops[0], "chainsize": chainsize, "max_gates": max_gates, "unitary": unitaryinit.name, "gates": gatespack.name, "othergates": othergatespack.name,
            "qbits": qbitinfo.name,
            #"targetqbits": targetqbits.name, "controlqbits": controlqbits.name, "derivates": derivates.name,
            "unitaryres": unitaryres.name} #, **({"hightcqbits" : hightcqbits.name} if num_qbits >= 9 else {})} #"unitaryrevres": unitaryrevres.name, 
    def get_unitary_sim(num_qbits, max_gates, tensornames=None, output_unitary=False):
        pow2qb = 1 << num_qbits
        if tensornames is None: tensornames = UnitarySimulator.build_chain(num_qbits, max_gates, output_unitary)
        iop = runtime.IOProgram(tensornames["iop"])
        driver = runtime.Driver()
        device = driver.next_available_device() # driver.devices[1]
        result = [None]
        import contextlib
        with contextlib.ExitStack() as exitstack:
            device_ = exitstack.enter_context(device)
            def closedevice(): exitstack.close()
            runfunc = [None]
            def loaddata():
                #for i in range(1+1+(2+(2 if num_qbits >= 9 else 0)+(2 if num_qbits >= 10 else 0))*2+2):
                #    device.load(iop[i], unsafe_keep_entry_points=True)
                device.load_all(iop, unsafe_keep_entry_points=True)
                num_inner_splits = (pow2qb+320-1)//320
                def actual(u, num_qbits, parameters, target_qbits, control_qbits):
                    num_gates = len(parameters)
                    padgates = 0 if (num_gates % tensornames["chainsize"]) == 0 else tensornames["chainsize"] - (num_gates % tensornames["chainsize"])
                    gateparams = [make_u3(parameters[i,:]) if target_qbits[i] == control_qbits[i] else make_cry(parameters[i,:]) for i in range(num_gates)] + [np.eye(2, dtype=np.complex128)]*padgates
                    num_inner_qbits = (max_gates+320-1)//320*320
                    padqbits = 0 if (num_gates % num_inner_qbits) == 0 else num_inner_qbits - (num_gates % num_inner_qbits)
                    target_qbits = np.concatenate((target_qbits, np.zeros(padqbits, dtype=target_qbits.dtype)))
                    control_qbits = np.concatenate((control_qbits, np.zeros(padqbits, dtype=control_qbits.dtype)))
                    num_gates += padgates
                    inputs = {}
                    inputs[tensornames["unitary"]] = np.ascontiguousarray(u.astype(np.complex64)).view(np.float32).reshape(pow2qb, pow2qb, 2).transpose(0, 2, 1).reshape(pow2qb*2, pow2qb)
                    invoke([device], iop, 0, 0, [inputs])
                    inputs = {}
                    #inputs[tensornames["gates"]] = np.concatenate([np.repeat(gateparams[i].astype(np.complex64).view(np.float32).flatten(), min(256, pow2qb)) for i in range(0, num_gates, 2)] + [np.zeros((2*2*2*min(256, pow2qb)), dtype=np.float32)]*((max_gates+1)//2-(num_gates-num_gates//2)))
                    inputs[tensornames["gates"]] = np.concatenate([gateparams[i].astype(np.complex64).view(np.float32).flatten() for i in range(0, num_gates, 2)] + [np.zeros((2*2*2), dtype=np.float32)]*((max_gates+1)//2-(num_gates-num_gates//2)))
                    #inputs[tensornames["othergates"]] = np.concatenate([np.repeat(gateparams[i].astype(np.complex64).view(np.float32).flatten(), min(256, pow2qb)) for i in range(1, num_gates, 2)] + [np.zeros((2*2*2*min(256, pow2qb)), dtype=np.float32)]*((max_gates+1)//2-num_gates//2))
                    inputs[tensornames["othergates"]] = np.concatenate([gateparams[i].astype(np.complex64).view(np.float32).flatten() for i in range(1, num_gates, 2)] + [np.zeros((2*2*2), dtype=np.float32)]*((max_gates+1)//2-num_gates//2))
                    #inputs[tensornames["targetqbits"]] = np.concatenate((np.repeat(np.hstack((target_qbits.astype(np.uint8)[:,np.newaxis]%8*2, target_qbits.astype(np.uint8)[:,np.newaxis]%8*2+1, np.array([[16]*14]*num_gates, dtype=np.uint8))), 20, axis=0).reshape(-1, 320), np.zeros((max_gates-num_gates, 320), dtype=np.uint8)))
                    adjcontrolqbits = np.where(control_qbits==target_qbits, 0, (control_qbits - (control_qbits > target_qbits)).astype(np.uint8))
                    deriv = control_qbits!=target_qbits
                    inputs[tensornames["qbits"]] = (target_qbits & 7) | ((adjcontrolqbits & 7) << 3)
                    if num_qbits == 9: inputs[tensornames["qbits"]] |= ((target_qbits>>3)<<6)
                    elif num_qbits == 10: inputs[tensornames["qbits"]] |= ((adjcontrolqbits>>3)<<6) | ((target_qbits>>3)<<7)
                    inputs[tensornames["qbits"]] = inputs[tensornames["qbits"]].astype(np.uint16) | (deriv.astype(np.uint16) << 8)
                    #inputs[tensornames["controlqbits"]] = np.concatenate((np.repeat(np.hstack((adjcontrolqbits[:,np.newaxis]%8*2, adjcontrolqbits[:,np.newaxis]%8*2+1, np.array([[16]*14]*num_gates, dtype=np.uint8))), 20, axis=0).reshape(-1, 320), np.zeros((max_gates-num_gates, 320), dtype=np.uint8)))
                    #if num_qbits >= 9:
                    #    hightcq = (adjcontrolqbits//8 + (target_qbits//8)*2).astype(np.uint8) if num_qbits==10 else (target_qbits//8).astype(np.uint8)
                    #    inputs[tensornames["hightcqbits"]] = np.concatenate((np.repeat(np.hstack((hightcq[:,np.newaxis]*2, hightcq[:,np.newaxis]*2+1, np.array([[16]*14]*num_gates, dtype=np.uint8))), 20, axis=0).reshape(-1, 320), np.zeros((max_gates-num_gates, 320), dtype=np.uint8)))
                    #derivs = np.array([0 if target_qbits[i]==control_qbits[i] else (i//2*2) ^ ((max_gates+1)//2*2) for i in range(num_gates)], dtype=np.uint16)
                    #inputs[tensornames["derivates"]] = np.concatenate((np.repeat(np.hstack(((derivs & 255).astype(np.uint8)[:,np.newaxis], (derivs >> 8).astype(np.uint8)[:,np.newaxis], np.array([[0]*14]*num_gates, dtype=np.uint8))), 20, axis=0).reshape(-1, 320), np.zeros((max_gates-num_gates, 320), dtype=np.uint8)))                    
                    #inputs[tensornames["derivates"]] = np.zeros((max_gates, 320), dtype=np.uint8)
                    invoke([device], iop, 1, 0, [inputs])
                    for i in range(0, num_gates, tensornames["chainsize"]):
                        #progidx = int(1+1+(2+(2 if num_qbits >= 9 else 0)+(2 if num_qbits >= 10 else 0) if (i&1)!=0 else 0) + target_qbits[i]//8*2 + (0 if target_qbits[i] == control_qbits[i] else 1+(2+(target_qbits[i]//8==0))*(adjcontrolqbits[i]//8)))
                        #progidx = int(1+1+(1+(1 if num_qbits >= 9 else 0)+(2 if num_qbits >= 10 else 0) if (i&1)!=0 else 0) + target_qbits[i]//8 + (0 if target_qbits[i] == control_qbits[i] else 2*(adjcontrolqbits[i]//8)))
                        progidx = 1+1
                        np.set_printoptions(threshold=sys.maxsize, formatter={'int':hex})
                        print(invoke([device], iop, progidx, 0, None, None, None))
                    progidx = 1+1+1 #1+1+(1+(1 if num_qbits >= 9 else 0)+(2 if num_qbits >= 10 else 0))*2+(num_gates&1) #1+1+(2+(2 if num_qbits >= 9 else 0)+(2 if num_qbits >= 10 else 0))*2+(num_gates&1)
                    res, _ = invoke([device], iop, progidx, 0, None, None, None)                    
                    if output_unitary:
                        #print(np.ascontiguousarray(res[0][tensornames["unitaryres" if (num_gates&1)==0 else "unitaryrevres"]].reshape(num_inner_splits, pow2qb, 2, min(256, pow2qb)).transpose(1, 0, 3, 2)).view(np.int32))
                        result[0] = np.ascontiguousarray(res[0][tensornames["unitaryres" if (num_gates&1)==0 else "unitaryrevres"]].reshape(num_inner_splits, pow2qb, 2, min(256, pow2qb)).transpose(1, 0, 3, 2)).view(np.complex64).reshape(pow2qb, pow2qb).astype(np.complex128)
                    else:
                        result[0] = np.sum(res[0][tensornames["unitaryres" if (num_gates&1)==0 else "unitaryrevres"]], axis=1)
                runfunc[0] = actual
        loaddata()
        actual = runfunc[0]
        return actual, result, closedevice
        
    def chain_test(num_qbits, max_gates, output_unitary=False):
        pow2qb = 1 << num_qbits
        num_gates, use_identity = 1, False #max_gates, False

        u = np.eye(pow2qb) + 0j if use_identity else unitary_group.rvs(pow2qb)
        target_qbits = np.array([np.random.randint(num_qbits) for _ in range(num_gates)], dtype=np.uint8)
        control_qbits = np.array([np.random.randint(num_qbits) for _ in range(num_gates)], dtype=np.uint8)
        parameters = np.random.random((num_gates, 3))
        oracleres = [None]
        def oracle():
            oracleres[0] = process_gates32(u, num_qbits, parameters, target_qbits, control_qbits)
            #oracleres[0] = qiskit_oracle(u, num_qbits, parameters, target_qbits, control_qbits)
            if not output_unitary: oracleres[0] = oracle = trace_corrections(oracleres[0], num_qbits)
        actual, result, closefunc = UnitarySimulator.get_unitary_sim(num_qbits, max_gates, output_unitary=output_unitary)
        oracle()
        actual(u, num_qbits, parameters, target_qbits, control_qbits)
        closefunc()
        oracleres, result = oracleres[0], result[0]
        if np.allclose(result, oracleres):
            print_utils.success("\nQuantum Simulator Chain Test Success ...")
        else:
            print_utils.err("\nQuantum Simulator Chain Test Failure")
            print_utils.infoc(str(abs(oracleres[~np.isclose(result, oracleres)] - result[~np.isclose(result, oracleres)]) / abs(oracleres[~np.isclose(result, oracleres)])))
def get_max_gates(num_qbits, max_levels):
    max_gates = num_qbits+3*(num_qbits*(num_qbits-1)//2*max_levels)
    if (max_gates % 80) != 0: max_gates += (80 - max_gates % 80)
    return max_gates
DUMP_AT_SLICE = 0
def main():
    global DUMP_AT_SLICE
    max_levels = 6
    num_qbits = 2
    DUMP_AT_SLICE = 25    
    UnitarySimulator.chain_test(num_qbits, get_max_gates(num_qbits, max_levels), True)
    print("Good Data on Stream Group #31 coming from MEM WEST 35 -> MEM WEST 25")
    DUMP_AT_SLICE = 23
    UnitarySimulator.chain_test(num_qbits, get_max_gates(num_qbits, max_levels), True)
    print("First Data Corruption on Stream Group #31 coming from MEM WEST 35 -> MEM WEST 23 (3rd of 4 tensors)")
    DUMP_AT_SLICE = 21
    UnitarySimulator.chain_test(num_qbits, get_max_gates(num_qbits, max_levels), True)
    print("First Data Corruption on Stream Group #31 coming from MEM WEST 35 -> MEM WEST 21 (3rd of 4 tensors)")
    DUMP_AT_SLICE = 19
    UnitarySimulator.chain_test(num_qbits, get_max_gates(num_qbits, max_levels), True)
    print("Second Data Corruption on Stream Group #31 coming from MEM WEST 35 -> MEM WEST 19 (2nd and 3rd of 4 tensors) -> ALU")
if __name__ == "__main__":
    main()
