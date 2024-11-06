# https://github.com/Alan-Robertson/QiskitPool/blob/f37f1b079f8e30c3498be4b89c57095bd6279709/src/qiskitpool/job.py
'''
    qiskitpool/job.py
    Contains the QJob class
'''
from functools import partial
from qiskit import execute

class QJob():
    '''
        QJob
        Job manager for asynch qiskit backends
    '''
    def __init__(self, *args, qjob_id=None, **kwargs):
        '''
            QJob.__init__
            Initialiser for a qiksit job
            :: *args    :: Args for qiskit execute
            :: **kwargs :: Kwargs for qiskit execute
        '''
        self.job_fn = partial(execute, *args, **kwargs)
        self.job = None
        self.done = False
        self.test_count = 10
        self.qjob_id = qjob_id

    def __call__(self):
        '''
            QJob.__call__
            Wrapper for QJob.run
        '''
        return self.run()

    def run(self):
        '''
            QJob.run
            Send async job to qiskit backend
        '''
        self.job = self.job_fn()
        return self

    def poll(self):
        '''
            QJob.poll
            Poll qiskit backend for job completion status
        '''
        if self.job is not None:
            return self.job.done()
        return False

    def cancel(self):
        '''
            QJob.cancel
            Cancel job on backend
        '''
        if self.job is None:
            return None
        return self.job.cancel()

    def position(self):
        pos = self.job.queue_position()
        if pos is None:
            return 0
        return pos

    def status(self):
        if self.job is None:
            return 'LOCAL QUEUE'
        else:
            status = self.job.status().value
            if 'running' in status:
                return 'RUNNING'
            if 'run' in status:
                return 'COMPLETE'
            if 'validated' in status:
                return 'VALIDATING'
            if 'queued' in status:
                pos = self.position()
                return f'QISKIT QUEUE: {self.position()}'

    def status_short(self):
        if self.job is None:
            return ' '
        else:
            status = self.job.status().value
            if 'running' in status:
                return 'R'
            if 'run' in status:
                return 'C'
            if 'validated' in status:
                return 'V'
            if 'queued' in status:
                return str(self.position())

    def result(self):
        '''
            QJob.result
            Get result from backend
            Non blocking - returns False if a job is not yet ready
        '''
        if self.poll():
            return self.job.result()
        return False
