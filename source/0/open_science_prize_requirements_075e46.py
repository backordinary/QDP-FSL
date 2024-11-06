# https://github.com/Fkaneko/ibm-quantum-open-science-prize-2021/blob/9c8018d61972707df7a3120d1c6cb1699973743f/src/open_science_prize_requirements.py
import copy
import logging
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig, listconfig
from qiskit import QuantumCircuit, QuantumRegister, execute
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.ignis.mitigation.measurement.fitters import MeasurementFilter

# Import state tomography modules
from qiskit.ignis.verification.tomography import StateTomographyFitter, state_tomography_circuits
from qiskit.opflow import One, Zero
from qiskit.providers import Job
from qiskit.providers.backend import BackendV1
from qiskit.quantum_info import state_fidelity
from qiskit.result.result import Result
from qiskit.tools.monitor import job_monitor

from src.carib import CalibName

log = logging.getLogger(__name__)


# The final time of the state evolution
TARGET_TIME = np.pi
# observed state
OBSERVE_STATES_3_QUBITS = ("000", "001", "010", "011", "100", "101", "110", "111")
# the number of 3 qubits state tomography circuits
NUM_STATE_TOMOGRAPHY_CIRCUITS = 27


@dataclass(frozen=True)
class QCJob:
    """Storing quantum circuit job input

    Attributes
    ----------
    st_qcs: List[QuantumCircuit]
        quantum circuit with state tomography circuit
    backend: BackendV1
        backend instance
    num_job_repeats: int
        how many jobs are repeated
    shots: int
        the number of shots
    """

    st_qcs: QuantumCircuit
    backend: BackendV1
    num_job_repeats: int = 8
    shots: int = 8192


def prepare_initial_state_open_science_prize(
    trotter_steps: int,
) -> Tuple[QuantumCircuit, float, int]:
    """
    Preparing initial state of open science prize
    """
    # The final time of the state evolution
    target_time = TARGET_TIME

    # Number of trotter steps
    if trotter_steps < 4:
        raise ValueError(
            f"""trotter_steps should be >=4, provided {trotter_steps},
            this is Open Science Prize requirements"""
        )

    # Initialize quantum circuit for 3 qubits
    qr = QuantumRegister(7)
    qc = QuantumCircuit(qr)
    # Prepare initial state (remember we are only evolving 3 of the 7 qubits on jakarta
    # qubits (q_5, q_3, q_1) corresponding to the state |110>)
    qc.x([3, 5])  # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)

    num_job_repeats = 8

    return qc, target_time, num_job_repeats


def set_tomography(qc: QuantumCircuit) -> List[QuantumCircuit]:
    qr = qc.qregs[0]
    # Generate state tomography circuits to evaluate fidelity of simulation
    st_qcs = state_tomography_circuits(qc, [qr[1], qr[3], qr[5]])
    return st_qcs


def submit_job(
    qc_job: QCJob,
    optimization_level: int = 1,
    seed_transpiler: int = 42,
    seed_simulator: int = 42,
    job_name: str = "ibmq_job_name",
    job_tags: List[str] = ["ibmq_job_tag"],
    calib_job_inputs: Optional[OrderedDict[CalibName, QuantumCircuit]] = None,
) -> Tuple[List[Job], OrderedDict[CalibName, Job]]:

    # need exact list class for job_tags
    if isinstance(job_tags, listconfig.ListConfig):
        job_tags = list(job_tags)

    calib_jobs = OrderedDict()
    if calib_job_inputs:
        for calib_name, calib_qcs in calib_job_inputs.items():
            log.info(f"{calib_name} job submission")
            log.info(f"backend: {qc_job.backend}")
            calib_job = execute(
                calib_qcs,
                qc_job.backend,
                shots=qc_job.shots,
                seed_transpiler=seed_transpiler,
                seed_simulator=seed_simulator + random.getrandbits(20),
                job_name=calib_name.value + "_" + job_name,
                job_tags=[calib_name.value] + job_tags,
            )
            log.info(f"{calib_name} JOB-ID {calib_job.job_id()}")
            calib_jobs[calib_name] = calib_job

    log.info(f"num_job_repeats: {qc_job.num_job_repeats}")
    log.info(f"backend: {qc_job.backend}")
    jobs = []
    for i in range(qc_job.num_job_repeats):
        # execute
        job = execute(
            qc_job.st_qcs,
            qc_job.backend,
            shots=qc_job.shots,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            seed_simulator=seed_simulator + random.getrandbits(20),
            job_name=job_name,
            job_tags=job_tags,
        )
        log.info(f"JOB-ID {i},  {job.job_id()}")
        jobs.append(job)

    return jobs, calib_jobs


def monitor_submitted_job_state(jobs: List[Job]) -> None:
    for job in jobs:
        log.info(f"Job status check, JOB ID: {job.job_id()}")
        job_monitor(job)
        try:
            if job.error_message() is not None:
                log.error(job.error_message())
        except AttributeError as e:
            log.debug(e)
        except Exception as e:
            log.error(e)


def state_tomo(result: Result, st_qcs: List[QuantumCircuit]) -> float:
    # The expected final state; necessary to determine state tomography fidelity
    target_state = (One ^ One ^ Zero).to_matrix()  # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)
    # Fit state tomography results
    tomo_fitter = StateTomographyFitter(result, st_qcs)
    rho_fit = tomo_fitter.fit(method="lstsq")
    # Compute fidelity
    fid = state_fidelity(rho_fit, target_state)
    return fid


def output_state_tomo(
    jobs: List[Job],
    st_qcs: List[QuantumCircuit],
    meas_fitter: Optional[CompleteMeasFitter],
    meas_method: str = "least_squares",
    xgb_pred_filters: Optional[List[MeasurementFilter]] = None,
) -> Tuple[List[float], List[Result]]:
    # Compute tomography fidelities for each repetition
    fids = []
    results: List[Result] = []
    for job_index, job in enumerate(jobs):
        result: Result = job.result()

        # check is there any circuits for mitigation
        num_circuits = len(result.results)
        if num_circuits != NUM_STATE_TOMOGRAPHY_CIRCUITS:
            if num_circuits == NUM_STATE_TOMOGRAPHY_CIRCUITS + len(OBSERVE_STATES_3_QUBITS):
                meas_res = copy.deepcopy(job.result())
                meas_res.results = meas_res.results[: len(OBSERVE_STATES_3_QUBITS)]
                meas_fitter = CompleteMeasFitter(
                    meas_res,
                    OBSERVE_STATES_3_QUBITS,
                    circlabel="mcal",
                )
                result.results = result.results[len(OBSERVE_STATES_3_QUBITS) :]

        if meas_fitter is not None:
            log.debug(f"meas calib is applied, \n {meas_fitter.cal_matrix}")
            result = meas_fitter.filter.apply(result, method=meas_method)

        # filter by xgboost model prediction is expected after measumrent filter
        if xgb_pred_filters is not None:
            result = xgb_pred_filters[job_index].apply(result, method="least_squares")

        fid = state_tomo(result, st_qcs)
        fids.append(fid)
        results.append(result)

    log.info("state tomography fidelity = {:.4f} \u00B1 {:.4f}".format(np.mean(fids), np.std(fids)))
    return fids, results
