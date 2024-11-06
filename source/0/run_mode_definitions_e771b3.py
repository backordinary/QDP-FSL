# https://github.com/Fkaneko/ibm-quantum-open-science-prize-2021/blob/9c8018d61972707df7a3120d1c6cb1699973743f/src/run_mode_definitions.py
import logging
from enum import Enum, Flag, auto
from typing import Any, Dict, Tuple

from omegaconf import DictConfig, OmegaConf
from qiskit import IBMQ
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.backend import BackendV1

log = logging.getLogger(__name__)


class RunMode(Flag):
    """RunMode flag. Switching top level process on run_circuit.py

    Due to IBMQ job needs hours or days, that's why this flag is generated for manifestly separate
    job submission and job evaluation.

    Attributes
    ----------
    SUBMIT_JOB: job submission mode. when using noisy_sim just calculated in a local and no submission to IBMQ
    EVALUATE: job evaluation mode after job submission.
    ALL: Both includes SUBMIT_JOB and EVALUATE mode
    """

    SUBMIT_JOB = auto()
    EVALUATE = auto()
    ALL = SUBMIT_JOB | EVALUATE


class TargetProcess(Flag):
    """TargetProcess flag. Switching top level process on run_circuit.py working with RunMode.

    This is generated for Trotter Step Cumulative Error data collection.
    Because at that process quite different circuits generation are required and additional logging is also needed to
    ExpLog on src/logger/exp_logger.py.

    Attributes
    ----------
    DATASET_GENERATION: data generation mode for Trotter Step Cumulative Error data.
        When RunMode.SUBMIT_JOB -> ~300 circuits will be generated with config.yaml specified parameter and src/carib.py.\n
        When RunMode.EVALUATE -> specifying input IBQ job with exp_log pickle paths, and relative jobs from IBMQ
                                 and then generate CSV data for XGBoost input
    NORMAL: normal job submission/evaluation mode
    ALL: Both includes DATASET_GENERATION and NORMAL mode

    Notes
    ----------
    * DATASET_GENERATION mode only work with real device jakarta not worked with qiskit simulator now.

    Warnings
    ----------
    * TargetProcess.ALL has not been tested. It would not work well.

    """

    DATASET_GENERATION = auto()
    NORMAL = auto()
    ALL = DATASET_GENERATION | NORMAL


class BackendName(Enum):
    """Backend name for qiskit job execution

    This class just provide the flag and actual instance of qiskit backend is generated through
    src.run_mode_definitions.get_backend.

    Attributes
    ----------
    DEFAULT: "ibmq_armonk" backend
    SIM: "sim" aer backend
    NOISY_SIM: "noisy_sim" aer with noisy model
    JAKARTA: "ibmq_jakarta" this contest IBMQ device
    """

    DEFAULT = "ibmq_armonk"
    SIM = "sim"
    NOISY_SIM = "noisy_sim"
    JAKARTA = "ibmq_jakarta"


def get_backend(backend_name: BackendName) -> Tuple[BackendV1, BackendV1]:
    """return backend which is specified with BackendName.

    For contest usage, always return jakarta backend.

    Parameters
    ----------
    backend_name : BackendName
        backend_name

    Returns
    -------
    jakarta: BackendV1
        jakarta backend for experiment log. This instance will not be used for job execution
    backend: BackendV1
        actual backend will be used for job execution


    Raises
    ------
    * If there is no saved IBMQ accounts in a local host.

    """
    try:
        # load IBMQ Account data
        provider = IBMQ.load_account()
    except Exception as e:
        log.info(f"Could not find any saved IBMQ accounts in a local host:{e}")
        log.error('Please save your IBMQ before run this code, \n "IBMQ.save_account(TOKEN, overwrite=True)"')
        log.info("About token: https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq")
        raise ValueError()

    # Get backend for experiment
    provider = IBMQ.get_provider(hub="ibm-q-community", group="ibmquantumawards", project="open-science-22")
    jakarta = provider.get_backend(BackendName.JAKARTA.value)

    if backend_name == BackendName.DEFAULT:
        # free, open backend
        provider = IBMQ.get_provider(hub="ibm-q", group="open", project="main")
        backend = provider.get_backend(BackendName.DEFAULT.value)
    elif backend_name == BackendName.SIM:
        # Noiseless simulated backend
        backend = QasmSimulator()
    elif backend_name == BackendName.NOISY_SIM:
        # Simulated backend based on ibmq_jakarta's device noise profile
        backend = QasmSimulator.from_backend(jakarta)
    elif backend_name == BackendName.JAKARTA:
        # real quantum machine backend
        backend = jakarta

    log.info(f"backend: {backend_name}")
    return jakarta, backend


def update_config_and_set_evaluation_args(
    conf_new: DictConfig, conf_old: DictConfig
) -> Tuple[DictConfig, Dict[str, Any]]:
    evaluation_kwargs = {}
    kwarg2conf_key = {
        "job_retrieval_ids": "job_retrieval.job_ids",
        "meas_fitter_override_skip_filter": "meas_fitter_override.skip_meas_filter",
        "meas_fitter_override_path": "meas_fitter_override.path",
    }
    conf_update_only_keys = ["backend_name", "run_mode", "target_process", "exp_log_path"]

    def _update_conf(conf_key: str) -> Any:
        value_new = OmegaConf.select(conf_new, conf_key)
        value_old = OmegaConf.select(conf_old, conf_key)
        if value_new != value_old:
            log.info(f"update conf_key: {conf_key}, {value_old} -> {value_new}")
            OmegaConf.update(conf_old, conf_key, value_new, force_add=True)
        return value_new

    for kwarg, conf_key in kwarg2conf_key.items():
        value_new = _update_conf(conf_key)
        evaluation_kwargs[kwarg] = value_new

    for conf_key in conf_update_only_keys:
        value_new = _update_conf(conf_key)

    return conf_old, evaluation_kwargs
