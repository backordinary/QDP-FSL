# https://github.com/carstenblank/dc-quantum-scheduling/blob/52d28dcc071f1d75199317c621521f6fca7d0408/dc_quantum_scheduling/processor.py
import logging
from datetime import datetime
from typing import Optional, Union, Dict

import pytz
import qiskit
from qiskit.providers import BaseJob
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq import IBMQBackend, IBMQBackendJobLimitError, BackendJobLimit
from qiskit.providers.ibmq.ibmqbackend import IBMQSimulator
from qiskit.providers.models import BackendProperties
from qiskit.qobj import Qobj
from retry.api import retry_call

from . import PreparedExperiment
from .models import RepeatAgainError, RunningExperiment
from .qiskit import get_gate_times

LOG = logging.getLogger(__name__)


def _unfinished_qobjs(running_experiment: 'RunningExperiment'):
    running_qobjs = len(running_experiment.job_list)
    unfinished_qobjs = running_experiment.qobj_list[running_qobjs:]
    return unfinished_qobjs


def _get_latest_calibration_noise_model(time: datetime, noise_models: Dict[datetime, NoiseModel]) -> Optional[NoiseModel]:
    try:
        all_eligible_cd = [(i, cd) for i, cd in enumerate(noise_models.keys()) if cd <= time]
        _, max_cd = max(all_eligible_cd, key=lambda kv: kv[0])
        return noise_models[max_cd]
    except:
        pass
    return None


def _run_inner_loop(running_experiment: 'RunningExperiment', qobj: Qobj, qobj_index: int,
                    key: str) -> BaseJob:

    backend = running_experiment.execution_backend

    if hasattr(backend, 'remaining_jobs_count') and backend.remaining_jobs_count() <= 0:
        # If the method 'remaining_jobs_count' exists, so does job_limit (it is called)
        # noinspection PyUnresolvedReferences
        job_limit: BackendJobLimit = backend.job_limit()
        LOG.info(f'Waiting for backend to start job for Qobj #{qobj_index} on {backend.name()}: '
                 f'{job_limit.active_jobs} / {job_limit.maximum_jobs}')
        raise RepeatAgainError(suggested_delay=60)

    LOG.info(f'Starting job for Qobj #{qobj_index} on {backend}...')
    if isinstance(backend, AerBackend):
        noise_model = _get_latest_calibration_noise_model(datetime.now(tz=pytz.utc), running_experiment.noise_model)
        job: BaseJob = backend.run(qobj=qobj, noise_model=noise_model,
                                   backend_options=running_experiment.parameters.get('backend_options', None),
                                   validate=running_experiment.parameters.get('validate', False))
    elif isinstance(backend, IBMQSimulator):
        noise_model = _get_latest_calibration_noise_model(datetime.now(tz=pytz.utc), running_experiment.noise_model)
        job: BaseJob = backend.run(qobj=qobj, job_name=f'{key}--{qobj_index}',
                                   job_tags=running_experiment.tags, validate_qobj=True,
                                   noise_model=noise_model)
    elif isinstance(backend, IBMQBackend):
        try:
            job: BaseJob = backend.run(qobj=qobj, job_name=f'{key}--{qobj_index}',
                                       job_tags=running_experiment.tags, validate_qobj=True)
        except IBMQBackendJobLimitError:
            LOG.warning(f"We tried to add a job while the job limit was reached. "
                        f"This shouldn't happen at this spot.")
            raise RepeatAgainError(suggested_delay=60)
    else:
        raise AssertionError('Either an IBMQBackend or AerBackend must be given as backend.')

    return job


def create_noise_model(running_experiment: 'RunningExperiment') -> None:
    skip_noise_model_creation = running_experiment.parameters.get('no_noise', False) \
        or running_experiment.transpiler_backend is None \
        or isinstance(running_experiment.transpiler_backend, IBMQSimulator)

    if skip_noise_model_creation or not isinstance(running_experiment.transpiler_backend, IBMQBackend):
        return

    device_properties: Optional[BackendProperties] = running_experiment.transpiler_backend.properties()
    # noinspection PyTypeChecker
    calibration_date: datetime = device_properties.last_update_date
    # noinspection PyTypeChecker
    gate_times = get_gate_times(running_experiment.transpiler_backend)
    noise_model: Optional[NoiseModel] = NoiseModel.from_backend(device_properties, gate_lengths=gate_times,
                                                                temperature=0, gate_length_units='ns')

    if running_experiment.noise_model is None:
        running_experiment.noise_model = {}

    if calibration_date not in running_experiment.noise_model:
        LOG.debug(f'Noise Models do not have the calibration date on file. Adding.')
        running_experiment.noise_model[calibration_date] = noise_model

    if 'gate_times' not in running_experiment.parameters:
        running_experiment.parameters['gate_times'] = {}
    if 'device_properties' not in running_experiment.parameters:
        running_experiment.parameters['device_properties'] = {}

    if calibration_date not in running_experiment.parameters['gate_times']:
        LOG.debug(f'Gate Times do not have the calibration date on file. Adding.')
        running_experiment.parameters['gate_times'][calibration_date] = gate_times
    if calibration_date not in running_experiment.parameters['device_properties']:
        LOG.debug(f'Device Properties do not have the calibration date on file. Adding.')
        running_experiment.parameters['device_properties'][calibration_date] = device_properties


def continue_running_experiment(running_experiment: 'RunningExperiment', key: Optional[str] = None) -> 'RunningExperiment':
    """
    Continue the processing.

    :param key:
    :param running_experiment: the running experiment to be executed
    :return: The updated running experiment
    :raises  RepeatAgainError: Something during the execution cannot be done at the moment. Please repeat later.
    :raises  AssertionError: wrong backend given!
    """

    unfinished_qobjs = _unfinished_qobjs(running_experiment)
    LOG.info(f'Processing of running experiment. Still {len(unfinished_qobjs)} open jobs.')
    while len(unfinished_qobjs) > 0:
        qobj = unfinished_qobjs[0]
        i: int = len(running_experiment.qobj_list) - len(unfinished_qobjs)

        if isinstance(running_experiment.transpiler_backend, IBMQBackend):
            LOG.debug(f'Check to create noise model from device properties of '
                      f'{running_experiment.transpiler_backend_name}.')
            create_noise_model(running_experiment)

        job = _run_inner_loop(running_experiment, qobj=qobj, qobj_index=i, key=key or running_experiment.external_id)
        LOG.info(f'Appending Job {job.job_id()} for Qobj #{i} on {running_experiment.transpiler_backend_name}...')
        running_experiment.job_list.append(job)
        unfinished_qobjs = _unfinished_qobjs(running_experiment)

        LOG.info(f'Still {len(unfinished_qobjs)} open jobs...')

    return running_experiment


def run_prepared_experiment(prepared_experiment: 'PreparedExperiment',
                            backend: Optional[Union[IBMQBackend, AerBackend]] = None) -> 'RunningExperiment':
    backend = backend if backend is not None else prepared_experiment.transpiler_backend
    LOG.info(f'Starting run on backend {backend.name()} with noise model {prepared_experiment.noise_model is not None}.')

    running_experiment = RunningExperiment(
        prepared_experiment=prepared_experiment,
        job_list=[],
        execution_backend=backend
    )
    running_experiment.external_id = f'{prepared_experiment.external_id}-{backend.name()}'

    return running_experiment


def execute_simulation(prepared_experiment: 'PreparedExperiment') -> 'RunningExperiment':
    running = run_prepared_experiment(prepared_experiment, backend=qiskit.Aer.get_backend('qasm_simulator'))
    return retry_call(exceptions=(RepeatAgainError,), f=continue_running_experiment, fargs=[running], delay=3)


def execute_experiment(prepared_experiment: 'PreparedExperiment', backend: IBMQBackend) -> 'RunningExperiment':
    LOG.info(f'Starting experiment on {backend}: {prepared_experiment.external_id}.')
    running = run_prepared_experiment(prepared_experiment, backend=backend)
    return retry_call(exceptions=(RepeatAgainError,), f=continue_running_experiment, fargs=[running], delay=10)
