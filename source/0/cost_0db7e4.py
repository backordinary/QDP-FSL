# https://github.com/amarjahin/kitaev_models_vqe/blob/839a1231c8870ab3f4c2015ec3ba6210fc4d0895/cost.py
import os
os.environ['MPMATH_NOSAGE'] = 'true'

from numpy import conjugate, exp
from numpy.linalg import norm
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.opflow import CircuitStateFn, PauliExpectation, CircuitSampler, StateFn
from qiskit.utils import QuantumInstance


def phys_energy_ev(hamiltonian,simulator, qc_c, params, projector):
    qc_c = qc_c.bind_parameters(params)
    if isinstance(simulator, StatevectorSimulator):
        result = simulator.run(qc_c).result()
        state_vec = result.get_statevector()
        phys_state_vec = projector @ state_vec
        phys_prop = norm(phys_state_vec)
        phys_state_vec = phys_state_vec/phys_prop
        phys_energy_ev = (conjugate(phys_state_vec.T) @ hamiltonian @ phys_state_vec).real
        return phys_energy_ev # + exp(-1/(1.05-phys_prop))
    elif isinstance(simulator, QasmSimulator): 
        psi = CircuitStateFn(qc_c)
        QI = QuantumInstance(simulator, shots=10000)
        ms_ph = StateFn(projector @ hamiltonian, is_measurement=True) @ psi
        ms_p = StateFn(projector, is_measurement=True) @ psi
        ph_ev = PauliExpectation().convert(ms_ph)
        p_ev = PauliExpectation().convert(ms_p)
        ph_samp = CircuitSampler(QI).convert(ph_ev).eval().real
        p_samp = CircuitSampler(QI).convert(p_ev).eval().real
        return ph_samp/p_samp
    else: 
        raise ValueError('simulator can only be QasmSimulator or StatevectorSimulator')


def energy_ev(hamiltonian,simulator, qc_c, params):
    qc_c = qc_c.bind_parameters(params)
    if isinstance(simulator, StatevectorSimulator):
        result = simulator.run(qc_c).result()
        state_vec = result.get_statevector()
        energy_ev = (conjugate(state_vec.T) @ hamiltonian @ state_vec).real 
        return  energy_ev
    else: 
        raise ValueError('simulator can only be StatevectorSimulator')
