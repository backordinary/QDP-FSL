# https://github.com/rum-yasuhiro/experiments_crosstalk_multitasking/blob/5d92c40dcfc68fab7922e806ab75e898bf3bda81/compiler/execute.py
from typing import List, Union, Dict, Callable, Any, Optional, Tuple
import datetime
import logging
from time import time
from qiskit import IBMQ
from qiskit.compiler import assemble
from qiskit.qobj.utils import MeasLevel, MeasReturnType

from multitasking_transpiler.palloq.compiler import multitasking_transpile
from experiments.benchmark_circuits import make_benckmarks
from experiments.compiler.utils.experimental_backend_regulation import (
    xtalk_scoped_bprop,
)
from experiments.compiler.readout_error_mitigation import run_meas_mitigation
from experiments.utils import send_slack, pickle_dump, pickle_load

logger = logging.getLogger(__name__)


def _log_submission_time(start_time, end_time):
    log_msg = "Total Job Submission Time - %.5f (ms)" % ((end_time - start_time) * 1000)
    logger.info(log_msg)


def run_experiments(
    jobfile_dir,
    multi_circuit_components,
    backend,
    simulator,
    shots=1024,
    crosstalk_info_filepath=None,
    crosstalk_prop=None,
    fake_device=None,
):
    """FIXME!
    以下の実験パート
    量子ビット使用数が増加すると、measurement error mitigation ができなくなるので、
    一時的にコメントアウトしている。
    削除するかどうか、要検討。
    """

    # edit backend property
    if fake_device:
        _backend = fake_device
        backend = simulator
    else:
        _backend = backend

    selected_bprop = xtalk_scoped_bprop(_backend)
    onlyCircuit = False

    # qiskit
    # job, tranpiled_circuit = _run_experiments(
    #     multi_circuit_components,
    #     backend,
    #     shots=shots,
    #     onlyCircuit=onlyCircuit,
    #     returnCircuit=True,
    #     optimization_level=3,
    # )

    # multi-tasking
    job_multi, circuit_multi = _run_experiments(
        multi_circuit_components,
        backend,
        shots=shots,
        onlyCircuit=onlyCircuit,
        returnCircuit=True,
        backend_properties=selected_bprop,
        multi_opt=True,
    )
    # job_multi_cal, state_labels_multi = run_meas_mitigation(
    #     circuit_multi, backend)

    # multi-tasking with xtalk noise
    job_xtalk, circuit_xtalk = _run_experiments(
        multi_circuit_components,
        backend,
        shots=shots,
        onlyCircuit=onlyCircuit,
        returnCircuit=True,
        backend_properties=selected_bprop,
        multi_opt=True,
        crosstalk_info_filepath=crosstalk_info_filepath,
        crosstalk_prop=crosstalk_prop,
    )
    # job_xtalk_cal, state_labels_xtalk = run_meas_mitigation(
    #     circuit_xtalk, backend)

    # run on simulator
    job_sim, original_circuit = _run_experiments(
        multi_circuit_components,
        backend=simulator,
        onlyCircuit=onlyCircuit,
        optimization_level=3,
        shots=shots,
        returnCircuit=True,
    )

    # get the job id
    # job_id = job.job_id()
    # job_id_cal = job_cal.job_id()

    job_id_multi = job_multi.job_id()
    # job_id_multi_cal = job_multi_cal.job_id()

    job_id_xtalk = job_xtalk.job_id()
    # job_id_xtalk_cal = job_xtalk_cal.job_id()

    job_id_sim = job_sim.job_id()

    return_dict = {
        # "qiskit": {
        #     "job": job_id,
        #     "circuit": tranpiled_circuit,
        #     # 'job_cal': job_id_cal,
        #     # 'state_labels': state_labels,
        # },
        "multi_opt": {
            "job": job_id_multi,
            "circuit": circuit_multi,
            # 'job_cal': job_id_multi_cal,
            # 'state_labels': state_labels_multi,
        },
        "xtalk_aware": {
            "job": job_id_xtalk,
            "circuit": circuit_xtalk,
            # 'job_cal': job_id_xtalk_cal,
            # 'state_labels': state_labels_xtalk,
        },
        "simulator": {"job": job_id_sim, "circuit": original_circuit},
    }

    execution_datetime = datetime.datetime.now().isoformat(timespec="seconds")

    benchmarking_circuits = ""
    for circ, num in multi_circuit_components.items():
        benchmarking_circuits = benchmarking_circuits + "_" + str(circ) + "-" + str(num)
    jobfile_path = jobfile_dir + execution_datetime + benchmarking_circuits + ".pickle"
    pickle_dump(return_dict, jobfile_path)
    print("############### successfully saved! ###############")
    # url = "https://hooks.slack.com/services/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"
    # send_slack('Experiments was done.', url)

    return [
        # tranpiled_circuit,
        circuit_multi,
        circuit_xtalk,
        original_circuit,
    ]


def _run_experiments(
    multi_circuit_components,
    backend=None,
    crosstalk_prop=None,
    crosstalk_info_filepath=None,
    returnCircuit=False,
    onlyCircuit=False,
    multi_opt=False,
    basis_gates=None,
    coupling_map=None,  # circuit transpile options
    backend_properties=None,
    initial_layout=None,
    seed_transpiler=None,
    optimization_level=None,
    pass_manager=None,
    qobj_id=None,
    qobj_header=None,
    shots=1024,  # common run options
    memory=False,
    max_credits=10,
    seed_simulator=None,
    default_qubit_los=None,
    default_meas_los=None,  # schedule run options
    schedule_los=None,
    meas_level=MeasLevel.CLASSIFIED,
    meas_return=MeasReturnType.AVERAGE,
    memory_slots=None,
    memory_slot_size=100,
    rep_time=None,
    rep_delay=None,
    parameter_binds=None,
    schedule_circuit=False,
    inst_map=None,
    meas_map=None,
    scheduling_method=None,
    init_qubits=None,
    **run_config
):
    circuit_list = make_benckmarks(multi_circuit_components)

    if crosstalk_prop is None and isinstance(crosstalk_info_filepath, str):
        cprop = pickle_load(crosstalk_info_filepath)
    else:
        cprop = crosstalk_prop

    experiments = multitasking_transpile(
        multi_circuits=circuit_list,
        backend=backend,
        backend_properties=backend_properties,
        multi_opt=multi_opt,
        crosstalk_prop=cprop,
        optimization_level=optimization_level,
    )

    if not onlyCircuit:
        qobj = assemble(
            experiments,
            qobj_id=qobj_id,
            qobj_header=qobj_header,
            shots=shots,
            memory=memory,
            max_credits=max_credits,
            seed_simulator=seed_simulator,
            default_qubit_los=default_qubit_los,
            default_meas_los=default_meas_los,
            schedule_los=schedule_los,
            meas_level=meas_level,
            meas_return=meas_return,
            memory_slots=memory_slots,
            memory_slot_size=memory_slot_size,
            rep_time=rep_time,
            rep_delay=rep_delay,
            parameter_binds=parameter_binds,
            backend=backend,
            init_qubits=init_qubits,
            **run_config
        )

        # executing the circuits on the backend and returning the job
        start_time = time()
        job = backend.run(qobj, **run_config)
        end_time = time()
        _log_submission_time(start_time, end_time)

        if returnCircuit:
            return job, experiments
        return job
    return experiments


class BackendNotDefinedError(Exception):
    "backend deviceがユーザ側で定義されていない場合"
    pass
