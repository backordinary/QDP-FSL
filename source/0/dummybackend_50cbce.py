# https://github.com/indisoluble/qasm-simulator-cpp-debian-pkg/blob/5fb81719defacf8731b96c74d934d7bf9191afa8/qiskit-sdk-py-0.5.1/test/python/_dummybackend.py
# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Dummy backend simulator.
The purpose of this class is to create a Simulator that we can trick for testing
purposes. Testing local timeouts, arbitrary responses or behavior, etc.
"""

import uuid
import logging
from concurrent import futures
import time

from qiskit import Result
from qiskit.backends import BaseBackend
from qiskit.backends import BaseJob
from qiskit.backends.basejob import JobStatus
from qiskit.backends.baseprovider import BaseProvider

logger = logging.getLogger(__name__)


class DummyProvider(BaseProvider):
    """Dummy provider just for testing purposes."""
    def get_backend(self, name):
        return DummySimulator()

    def available_backends(self, filters=None):
        # pylint: disable=arguments-differ
        backends = {DummySimulator.name: DummySimulator()}

        filters = filters or {}
        for key, value in filters.items():
            backends = {name: instance for name, instance in backends.items()
                        if instance.configuration.get(key) == value}
        return list(backends.values())


class DummySimulator(BaseBackend):
    """ This is Dummy backend simulator just for testing purposes """

    DEFAULT_CONFIGURATION = {
        'name': 'local_dummy_simulator',
        'url': 'https://github.com/IBM/qiskit-sdk-py',
        'simulator': True,
        'local': True,
        'description': 'A dummy simulator for testing purposes',
        'coupling_map': 'all-to-all',
        'basis_gates': 'u1,u2,u3,cx,id'
    }

    def __init__(self, configuration=None, time_alive=10):
        """
        Args:
            configuration (dict): backend configuration
            time_alive (int): time to wait before returning result
        """
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy())
        self.time_alive = time_alive

    def run(self, q_job):
        return DummyJob(self.run_job, q_job)

    def run_job(self, q_job):
        """ Main dummy simulator loop """
        job_id = str(uuid.uuid4())
        qobj = q_job.qobj

        time.sleep(self.time_alive)

        return Result({'job_id': job_id, 'result': [], 'status': 'COMPLETED'}, qobj)


class DummyJob(BaseJob):
    """dummy simulator job"""
    _executor = futures.ProcessPoolExecutor()

    def __init__(self, fn, qobj):
        super().__init__()
        self._qobj = qobj
        self._future = self._executor.submit(fn, qobj)

    def result(self, timeout=None):
        # pylint: disable=arguments-differ
        return self._future.result(timeout=timeout)

    def cancel(self):
        return self._future.cancel()

    def status(self):
        if self.running:
            _status = JobStatus.RUNNING
        elif not self.done:
            _status = JobStatus.QUEUED
        elif self.cancelled:
            _status = JobStatus.CANCELLED
        elif self.done:
            _status = JobStatus.DONE
        elif self.error:
            _status = JobStatus.ERROR
        else:
            raise Exception('Unexpected state of {0}'.format(
                self.__class__.__name__))
        _status_msg = None
        return {'status': _status,
                'status_msg': _status_msg}

    @property
    def cancelled(self):
        return self._future.cancelled()

    @property
    def done(self):
        return self._future.done()

    @property
    def running(self):
        return self._future.running()

    @property
    def error(self):
        """
        Return Exception object if exception occured else None.

        Returns:
            Exception: exception raised by attempting to run job.
        """
        return self._future.exception(timeout=0)
