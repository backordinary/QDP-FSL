# https://github.com/gaqqie/gaqqie-door/blob/5fc6141d726121a9182371138f88fd8b7a575510/gaqqie_door/adaptor/qiskit/gaqqie_backend.py
import json
from typing import List

from qiskit import QuantumCircuit
from qiskit.compiler.assembler import assemble
from qiskit.providers import BackendV1, Job, Options
from qiskit.providers.models import BackendStatus
from qiskit.providers.models import QasmBackendConfiguration

from .gaqqie_job import GaqqieJob
from ...rest import Jobbeforesubmission, Device


class GaqqieBackend(BackendV1):
    """This class is created based on the following.

    https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/providers/backend.py

    """

    def __init__(
        self,
        configuration: QasmBackendConfiguration,
        device: Device,
        provider: "GaqqieProvider",
    ) -> None:
        """Initializes the backend object.

        Parameters
        ----------
        configuration : qiskit.providers.models.QasmBackendConfiguration
            configuration of the backend.
        device : Device
            Device model for the backend.
        provider : GaqqieProvider
            GaqqieProvider for the backend.
        """
        super().__init__(configuration, provider)
        self._device: Device = device

    @classmethod
    def _default_options(cls) -> Options:
        """Returns the default options.

        This method will return a :class:`qiskit.providers.Options`
        subclass object that will be used for the default options. These
        should be the default parameters to use for the options of the
        backend.

        Returns
        -------
        qiskit.providers.Options
            A options object with default values set.
        """
        return Options(shots=1024)

    # see https://github.com/Qiskit/qiskit-ibmq-provider/blob/5bc1d7a61a057894679e00784faa24bfb6c8de19/qiskit/providers/ibmq/ibmqbackend.py#L146
    def run(self, circuit: QuantumCircuit, **options) -> Job:
        """Runs on the backend.

        This method that will return a :class:`~qiskit.providers.Job` object
        that run circuits. Depending on the backend this may be either an async
        or sync call. It is the discretion of the provider to decide whether
        running should  block until the execution is finished or not. The Job
        class can handle either situation.

        Parameters
        ----------
        circuits : qiskit.QuantumCircuit
            An individual or a
            list of :class:`~qiskit.circuits.QuantumCircuit` or
            :class:`~qiskit.pulse.Schedule` objects to run on the backend.
            For legacy providers migrating to the new versioned providers,
            provider interface a :class:`~qiskit.qobj.QasmQobj` or
            :class:`~qiskit.qobj.PulseQobj` objects should probably be
            supported too (but deprecated) for backwards compatibility. Be
            sure to update the docstrings of subclasses implementing this
            method to document that. New provider implementations should not
            do this though as :mod:`qiskit.qobj` will be deprecated and
            removed along with the legacy providers interface.
        options : Any
            Any kwarg options to pass to the backend for running the
            config. If a key is also present in the options
            attribute/object then the expectation is that the value
            specified will be used instead of what's set in the options
            object.

        Returns
        -------
        qiskit.providers.Job
            tThe job object for the run.
        """
        serialized_json = json.dumps(assemble(circuit, self).to_dict(), indent=2)
        request = Jobbeforesubmission(
            provider_name=self._device.provider_name,
            device_name=self.name(),
            instructions=serialized_json,
        )
        response = self.provider().job_api.submit_job(request)
        job = GaqqieJob(self, response)
        return job

    def status(self) -> BackendStatus:
        """Returns the backend status.

        Returns
        -------
        qiskit.providers.models.BackendStatus
            the status of the backend.
        """
        return BackendStatus(
            backend_name=self.name(),
            backend_version="1",
            operational=True,
            pending_jobs=self._device.queued_jobs,
            status_msg=self._device.status,
        )

    def jobs(self) -> List[GaqqieJob]:
        """Returns the backend status.

        Returns
        -------
        List[GaqqieJob]
            list of jobs.
        """
        # get job information from cloud
        jobs = self.provider().job_api.get_jobs()

        # create backend instances
        gaqqie_jobs = []
        for job in jobs:
            gaqqie_job = GaqqieJob(
                backend=self,
                job=job,
            )
            gaqqie_jobs.append(gaqqie_job)

        return gaqqie_jobs
