# https://github.com/msrn-yossy/violation_of_CHSH/blob/28f005c6d1d12d67191f1112cd79a0ad2afdc402/src/experiments.py
# coding: utf-8

import pandas as pd
import numpy as np
import os

from qiskit import QuantumCircuit, IBMQ, transpile

from qiskit.providers.ibmq.ibmqbackend import IBMQBackend
from qiskit.providers.ibmq import least_busy
from qiskit.tools import job_monitor


DIR_NAME = os.path.dirname(__file__)
PATH_CSV = DIR_NAME + "/../out/data/chshs.csv"


def _gen_bell_Phi_m() -> QuantumCircuit:
    r"""returns circuit a Bell state,
    .. math::
        $|\Psi^-\rangle = \dfrac{1}{\sqrt{2}} (|00\rangle - |11\rangle)$.
    """
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cz(0, 1)
    return circuit


def _gen_chsh_term(
    theta: float,
    phi: float,
) -> QuantumCircuit:
    r"""generate a term of CHSH correlation, by rotating a Bell state.
    """
    circuit = _gen_bell_Phi_m()
    circuit.rx(theta=-theta, qubit=0)
    circuit.rx(theta=-phi, qubit=1)

    return circuit


def _observe_ZZ(
    circuit: QuantumCircuit,
    backend: IBMQBackend,
) -> dict[str, int]:
    """returns counts of 00, 01, 10, and 11.
    """
    circuit.measure(0, 0)
    circuit.measure(1, 1)

    compiled_circuit = transpile(circuit, backend)
    job = backend.run(
        circuits=compiled_circuit,
    )
    job_monitor(job, interval=2)

    result = job.result()
    counts = result.get_counts(compiled_circuit)

    return counts


def _calc_corr_term(counts: dict[str, int]) -> float:
    """calculates a term of CHSH correlation.
    """
    counts.setdefault('00', 0)
    counts.setdefault('01', 0)
    counts.setdefault('10', 0)
    counts.setdefault('11', 0)

    shots = sum(counts.values())

    p00 = counts['00'] / shots
    p01 = counts['01'] / shots
    p10 = counts['10'] / shots
    p11 = counts['11'] / shots

    return  p00 + p11 - p01 - p10


def calc_chshs() -> None:
    """calculates quantum-version CHSH correlations in which an angle is changed.
    """
    IBMQ.load_account()

    for idx_t, t in enumerate(np.linspace(0, 2 * np.pi, 9)):
        di_TPs = dict( # TP = \Theta and \Phi 
            t_=- np.pi / 2.,
            p=- np.pi / 4.,
            p_=- 3. * np.pi / 4.,
        )
        di_TPs["t"] = t
        print(t)

        li_pairs_directions = [
            ("t", "p"),
            ("t_", "p"),
            ("t", "p_"),
            ("t_", "p_"),
        ]
        for TP in li_pairs_directions:
            print(TP)

            IBMQ.providers()
            provider = IBMQ.get_provider(hub='ibm-q')
            backend = least_busy(provider.backends(
                simulator=False,
                operational=True,
            ))

            circuit = _gen_chsh_term(
                theta=di_TPs[TP[0]],
                phi=di_TPs[TP[1]],
            )
            counts = _observe_ZZ(
                circuit=circuit,
                backend=backend,
            )
            corr = _calc_corr_term(counts=counts)

            df_chshs = pd.read_csv(PATH_CSV, index_col=0)
            df_chshs.loc[f"{idx_t}", TP[0] + TP[1]] = corr
            df_chshs.to_csv(PATH_CSV)


if __name__ == "__main__":
    calc_chshs()
