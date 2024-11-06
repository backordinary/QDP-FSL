# https://github.com/mar-be/master-thesis-code/blob/79c102a2b685eb1c54ace12cb6df31f32d4a51ae/execution_handler/execution_handler.py
import copy
import math
import time
from collections import Counter
from concurrent.futures.process import BrokenProcessPool
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Dict, List, Union

import logger
import psutil
import qiskit.providers.ibmq.job.exceptions
import qiskit.tools.parallel
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.providers import Backend
from qiskit.providers.ibmq.accountprovider import AccountProvider
from qiskit.providers.job import Job
from qiskit.providers.provider import Provider
from qiskit.qobj import Qobj
from qiskit.result.models import ExperimentResultData
from qiskit.result.result import Result
from quantum_execution_job import QuantumExecutionJob


def new_parallel_map(task, values, task_args=tuple(), task_kwargs={}, num_processes=qiskit.tools.parallel.CPU_COUNT):
    if num_processes == psutil.cpu_count(logical=True) and num_processes > 1:
        # at least one free logical core
        num_processes -= 1
    print(f"num_processes={num_processes}")
    return qiskit.tools.parallel.parallel_map(task, values, task_args, task_kwargs, num_processes)

# overwrite the parallel_map function to enable the usage of more CPU cores
transpile.__globals__["parallel_map"] = new_parallel_map

class BackendLookUp():
    """Look up information about the remote backends
    """

    def __init__(self, provider:Provider) -> None:
        self._log = logger.get_logger(type(self).__name__)
        self._provider = provider
        self._backends = {}

    def get(self, backend_name:str, exclusiv=False) -> Backend:
        """Get the corresponding backend

        Args:
            backend_name (str)
            exclusiv (bool, optional): If true, return a new exlusive object. Defaults to False.

        Returns:
            Backend: representing the remote QPU
        """
        if exclusiv:
            return self._provider.get_backend(backend_name)
        try:
            backend = self._backends[backend_name]
        except KeyError:
            backend = self._provider.get_backend(backend_name)
            self._log.info(f"Retrieved backend {backend_name}")
            self._backends[backend_name] = backend
        return backend

    def max_shots(self, backend_name:str) -> int:
        backend = self.get(backend_name)
        return backend.configuration().max_shots

    def max_experiments(self, backend_name:str) -> int:
        backend = self.get(backend_name)
        return backend.configuration().max_experiments


class BackendControl():
    """Control the access to the backends to regulate the number of queued jobs"""

    def __init__(self,):
        self._log = logger.get_logger(type(self).__name__)
        self._locks = {}
        self._counters = {}
        
    def try_to_enter(self, backend_name:str, backend:Backend) -> bool:
        """Try to enter the lock of the backend

        Args:
            backend_name (str)
            backend (Backend)

        Returns:
            bool: True, if successfully entered the lock. False, otherwise
        """
        try:
            lock = self._locks[backend_name]
            counter = self._counters[backend_name]
        except KeyError:
            lock = Lock()
            counter = 0
            self._locks[backend_name] = lock
            self._counters[backend_name] = counter

        with lock:
            limit = backend.job_limit()
            self._log.debug(f"Backend: {backend_name} Counter:{counter} Active_Jobs:{limit.active_jobs} Maximum_jobs:{limit.maximum_jobs}")
            if limit.active_jobs < limit.maximum_jobs:
                if counter < limit.maximum_jobs:
                    self._counters[backend_name] += 1
                    return True
            return False

    def leave(self, backend_name:str):
        """Leave the lock of the backend

        Args:
            backend_name (str)
        """
        with self._locks[backend_name]:
            self._counters[backend_name] -= 1


class Transpiler():

    def __init__(self, input:Queue, output:Queue, backend_look_up:BackendLookUp, timeout:int, max_transpile_batch_size:Union[float, int]=float('inf')) -> None:
        self._log = logger.get_logger(type(self).__name__)
        self._input = input
        self._output = output
        self._backend_look_up = backend_look_up
        self._timeout = timeout
        self._max_transpile_batch_size = max_transpile_batch_size
        self._jobs_to_transpile = {}
        self._timers = {}
        self._pending_transpilation = {}
        self._pending = Queue()
        self._finished = Queue()
        self._log.info("Init")

    def start(self):
        Thread(target=self._route_job).start()
        Thread(target=self._transpile).start()
        self._log.info("Started")


    def _transpile(self):
        """Function that transpiles the pending transpilation Batches 

        Raises:
            e: BrokenProcessPool
        """
        while True:
            backend_name, jobs = self._pending.get()
            backend = self._backend_look_up.get(backend_name)
            circuits = list([job.circuit for job in jobs])
            self._log.debug(f"Start transpilation of {len(circuits)} circuits for backend {backend.name()}")
            trans_start_time = time.time()
            tries = 3
            for i in range(tries):
                try:
                    transpiled_circuits = transpile(circuits, backend=backend)
                    break
                except BrokenProcessPool as e:
                    if hasattr(e, 'message'):
                        self._log.info(f"In try {i} the following error occured {e.message}")
                    else:
                        self._log.exception(e)
                    if i == tries-1:
                        raise e
            time_diff = time.time() - trans_start_time
            self._log.info(f"Transpiled {len(transpiled_circuits)} circuits for backend {backend.name()} in {time_diff}s")
            self._finished.put((backend_name, zip(transpiled_circuits, jobs)))
            
            
    def _create_transpilation_batch(self, backend_name:str) -> bool:
        """Creates a batch for the transpilation for the given backend, if there is not already a pending transpilation for the backend

        Args:
            backend_name (str)

        Returns:
            bool: True, if a batch was created
        """
        try:
            if self._pending_transpilation[backend_name]:
                return False
        except KeyError:
            pass
        n_jobs = min([len(self._jobs_to_transpile[backend_name]), self._backend_look_up.max_experiments(backend_name), self._max_transpile_batch_size])
        self._log.debug(f"Prepared {n_jobs} circuits for the transpilation for backend {backend_name}")
        jobs = self._jobs_to_transpile[backend_name][:n_jobs]
        self._jobs_to_transpile[backend_name] = self._jobs_to_transpile[backend_name][n_jobs:]
        self._pending.put((backend_name, jobs))
        self._pending_transpilation[backend_name] = True
        if len(self._jobs_to_transpile[backend_name]) > 0:
                self._timers[backend_name] =  time.time()
        return True

    
    def _add_job(self, job:QuantumExecutionJob):
        """Add an incoming job to the internal data structure in order to transpile it for the given backend.
        Checks if there are enough jobs to create a full transpilation batch.

        Args:
            job (QuantumExecutionJob): job to transpile
        """
        backend_name = job.backend_data.name
        try:
            self._jobs_to_transpile[backend_name].append(job)
        except KeyError:
            self._jobs_to_transpile[backend_name] = [job]
        if not backend_name in self._timers.keys():
            self._timers[backend_name] = time.time()
        if len(self._jobs_to_transpile[backend_name]) == self._backend_look_up.max_experiments(backend_name):
            # Todo try to cancel
            if self._create_transpilation_batch(backend_name):
                self._timers.pop(backend_name)

    def _check_timers(self):
        """Check the timers. If an timeout occurs, try to create a transpilation batch.
        """
        timers_to_clear = []
        for backend_name in self._timers.keys():
            time_diff = time.time() - self._timers[backend_name]
            if time_diff > self._timeout:
                if self._create_transpilation_batch(backend_name):
                    self._log.debug(f"Transpilation timeout for backend {backend_name}: {time_diff}s")
                    timers_to_clear.append(backend_name)
        for backend_name in timers_to_clear:
            self._timers.pop(backend_name)
            if len(self._jobs_to_transpile[backend_name]) > 0:
                # start a new timer
                self._timers[backend_name] = time.time()


    def _any_pending_transpilation(self) -> bool:
        """
        Returns:
            bool: True if there are any pending transpilation batches
        """
        if len(self._pending_transpilation) == 0:
            return False
        else:
            return any(self._pending_transpilation.values())

    def _route_job(self):
        """Function that processes incoming jobs, periodically chechs the timers, and forwards transpiled jobs to the output
        """
        while True:
            for i in range(1000):
                try:
                    timeout = 0.1
                    if i == 0:
                        timeout = 1
                    # only block in the first iteration
                    job = self._input.get(timeout=timeout)
                    self._add_job(job)
                except Empty:
                    break
            if not self._any_pending_transpilation():
                self._check_timers()
            try:
                backend_name, transpiled_result = self._finished.get(block=False)
                self._pending_transpilation[backend_name] = False
                for transpiled_tuple in transpiled_result:
                    self._output.put(transpiled_tuple)
                self._check_timers()
            except Empty:
                pass


        

class Batch():
    '''A batch represents a job on a backend. It can contain multiple experiments.'''

    def __init__(self, backen_name:str, max_shots: int, max_experiments: int, batch_number:int):
        self._log = logger.get_logger(type(self).__name__)
        self.backend_name = backen_name
        self.max_shots = max_shots
        self.shots = 0
        self.max_experiments = max_experiments
        self.experiments = []
        self.remaining_experiments = max_experiments
        self.n_circuits = 0
        self.batch_number = batch_number

    
    def add_circuit(self, key, circuit:QuantumCircuit, shots:int) -> int:
        """Add a circuit to the Batch.

        Args:
            key (Any): Identifier for the circuit
            circuit (QuantumCircuit): The circuit, which should be executed
            shots (int): The number of shots
    
        Returns:
            int: remaining shots. If they are 0, all shots are executed
        """
        if self.remaining_experiments == 0:
            return shots

        self.n_circuits += 1
        reps = math.ceil(shots/self.max_shots)
        self.shots = max(self.shots, min(shots, self.max_shots))

        if reps <= self.remaining_experiments:
            remaining_shots = 0
        else:
            reps = self.remaining_experiments
            remaining_shots = shots - reps*self.max_shots

        self.remaining_experiments -= reps
        self.experiments.append({"key":key, "circuit":circuit, "reps":reps, "shots":shots-remaining_shots, "total_shots":shots})
        
        return remaining_shots       
                

class Batcher(Thread):

    def __init__(self, input:Queue, output: Queue, quantum_job_table:Dict, backend_look_up:BackendLookUp, batch_timeout:int=30) -> None:
        self._log = logger.get_logger(type(self).__name__)
        self._input = input
        self._output = output
        self._batch_timeout = batch_timeout
        self._quantum_job_table = quantum_job_table
        self._backend_look_up = backend_look_up
        self._batch_timers = {}
        self._batch_count = {}
        self._batches = {}
        Thread.__init__(self)
        self._log.info("Init")


    def _get_or_create_batch(self, backend_name:str):
        try:
            batch = self._batches[backend_name]
            if batch.remaining_experiments == 0:
                batch = self._create_new_batch(backend_name)
        except KeyError:
            batch = Batch(backend_name, self._backend_look_up.max_shots(backend_name), self._backend_look_up.max_experiments(backend_name), 0)
            self._batches[backend_name] = batch
            self._batch_count[backend_name] = 0
        return batch

    def _create_new_batch(self, backend_name:str):
        self._batch_count[backend_name] += 1
        batch = Batch(backend_name, self._backend_look_up.max_shots(backend_name), self._backend_look_up.max_experiments(backend_name), self._batch_count[backend_name])
        self._batches[backend_name] =  batch
        return batch
        

    def _add_to_batch(self, transpiled_circuit:QuantumCircuit, job:QuantumExecutionJob):
        """Add the circuit to the batch. Automatically forwards full batches and creates new batches.

        Args:
            transpiled_circuit (QuantumCircuit)
            job (QuantumExecutionJob): corresponding job to the transpiled circuit containing the shot number and the backend
        """
        backend_name = job.backend_data.name
        if not backend_name in self._batch_timers.keys():
            self._batch_timers[backend_name] = time.time()
        key = job.id
        remaining_shots = job.shots
        while remaining_shots > 0:
            batch = self._get_or_create_batch(backend_name)
            remaining_shots = batch.add_circuit(key, transpiled_circuit, remaining_shots)
            if batch.remaining_experiments == 0:
                self._log.info(f"Generated full batch {backend_name}/{self._batch_count[backend_name]}")
                self._output.put(batch)
                if remaining_shots > 0:
                    self._batch_timers[backend_name] = time.time()
                else:
                    self._batch_timers.pop(backend_name)
        
    
    def _check_timers(self):
        """Checks if an timeout occured for a batch and then forward it.
        """
        timers_to_clear = []
        for backend_name in self._batch_timers.keys():
            time_diff = time.time() - self._batch_timers[backend_name]
            if time_diff > self._batch_timeout:
                batch = self._batches[backend_name]
                self._log.debug(f"Timeout for batch {backend_name}/{self._batch_count[backend_name]}, Time passed: {time_diff}, batch_size:{batch.max_experiments - batch.remaining_experiments}, max batch size {batch.max_experiments}")
                self._output.put(batch)
                self._create_new_batch(backend_name)
                timers_to_clear.append(backend_name)
        for backend_name in timers_to_clear:
            self._batch_timers.pop(backend_name)


    def run(self) -> None:
        self._log.info("Started")
        while True:
            try:
                transpiled_circ, job = self._input.get(timeout=5)
                self._quantum_job_table[job.id] = job
                self._add_to_batch(transpiled_circ, job)
            except Empty:
                pass
            self._check_timers()


class Submitter(Thread):

    def __init__(self, input: Queue, output: Queue, backend_look_up:BackendLookUp, backend_control:BackendControl, defer_interval=60):
        self._log = logger.get_logger(type(self).__name__)
        self._input = input
        self._output = output
        self._backend_look_up = backend_look_up
        self._backend_control = backend_control
        self._defer_interval = defer_interval
        self._internal_jobs_queue = []
        Thread.__init__(self)
        self._log.info("Init")

    def _assemble(self, batch:Batch) -> Qobj:
        """Assemble a Qobj from a Batch.

        Args:
            batch (Batch)

        Returns:
            Qobj
        """
        backend_name = batch.backend_name
        backend = self._backend_look_up.get(backend_name)
        circuits = list([circuit_item["circuit"] for circuit_item in batch.experiments])
        multiplied_circuits = []
        for i, circuit_item in enumerate(batch.experiments):
            reps = circuit_item["reps"]
            circ = circuits[i]
            multiplied_circuits.extend([circ]*reps)
        #self._log.info(f"Transpiled batch {backend_name}/{batch.batch_number}")
        qobj =  assemble(multiplied_circuits, backend, shots=batch.shots, memory=True)
        self._log.info(f"Assembled Qobj for batch {backend_name}/{batch.batch_number}")
        return qobj

        
    def run(self) -> None:
        self._log.info("Started")
        batch: Batch
        qobj: Qobj
        submit_interval = 30
        last_time = 0
        while True:
            try:
                batch = self._input.get(timeout=self._defer_interval)
                qobj = self._assemble(batch)
                self._internal_jobs_queue.append((batch, qobj))
            except Empty:
                pass
            current = time.time()
            if current - last_time > submit_interval:
                deferred_jobs = []
                for batch, qobj in self._internal_jobs_queue:
                    backend_name = batch.backend_name
                    backend = self._backend_look_up.get(backend_name)
                    if self._backend_control.try_to_enter(backend_name, backend):
                        job = backend.run(qobj)
                        self._log.info(f"Submitted batch {batch.backend_name}/{batch.batch_number}")
                        self._output.put((batch, job))
                    else:
                        # self._log.debug(f"Reached limit of queued jobs for backend {backend_name} -> defer job for batch {batch.batch_number}")
                        deferred_jobs.append((batch, qobj))
                self._internal_jobs_queue = deferred_jobs
                last_time = time.time()

            
class Retriever(Thread):

    def __init__(self, input: Queue, output: Queue, wait_time:float, backend_control:BackendControl):
        self._log = logger.get_logger(type(self).__name__)
        self._input = input
        self._output = output
        self._wait_time = wait_time
        self._backend_control = backend_control
        self._jobs = []
        self._batch_counter = {}
        self._deferred_results = {}   
        Thread.__init__(self)
        self._log.info("Init")    


    def run(self):
        self._log.info("Started")
        batch: Batch
        job: Job
        while True:
            i = 0
            while not self._input.empty() and i < 5:
                job_tuple = self._input.get()
                self._jobs.append(job_tuple)
                i += 1
            final_state_jobs = []
            for batch, job in self._jobs:
                try:
                    if job.in_final_state():
                        final_state_jobs.append((batch, job))
                        self._backend_control.leave(batch.backend_name)
                except qiskit.providers.ibmq.job.exceptions.IBMQJobApiError as e:
                    self._log.info("Connection Problem") 
            for job_tuple in final_state_jobs:
                self._jobs.remove(job_tuple)
                batch, job = job_tuple
                self._log.info(f"Received result for batch {batch.backend_name}/{batch.batch_number}")
                try:
                    batch_counter = self._batch_counter[batch.backend_name]
                except KeyError:
                    batch_counter = 0
                    self._batch_counter[batch.backend_name] = 0
                    self._deferred_results[batch.backend_name] = []
                if batch_counter == batch.batch_number:
                    # the received batch is the next batch -> output it
                    batch_counter += 1
                    self._output.put(job_tuple)
                    # check if deferred results can be output
                    if len(self._deferred_results[batch.backend_name]) > 0:
                        # sort the deferred job tuples according to their batch number
                        self._deferred_results[batch.backend_name].sort(key=lambda tuple:tuple[0].batch_number)
                    while len(self._deferred_results[batch.backend_name]) > 0 and batch_counter == self._deferred_results[batch.backend_name][0][0].batch_number:
                        # ouput jobs as long as their are deferred jobs and the batch number is next following number
                        self._output.put(self._deferred_results[batch.backend_name].pop(0))
                        batch_counter += 1
                    
                    self._batch_counter[batch.backend_name] = batch_counter
                else:
                    # the received batch is not the next batch
                    self._deferred_results[batch.backend_name].append(job_tuple)
                    self._log.info(f"Deferred result for batch {batch.backend_name}/{batch.batch_number} to establish order")

            time.sleep(self._wait_time)




    

class ResultProcessor(Thread):

    def __init__(self, input: Queue, output: Queue, quantum_job_table:Dict, memory:bool=False):
        self._log = logger.get_logger(type(self).__name__)
        self._input = input
        self._output = output
        self._quantum_job_table = quantum_job_table
        self._memory = memory
        self._previous_key = {}
        self._previous_memory = {}
        self._previous_counts = {}
        Thread.__init__(self)
        self._log.info("Init") 

    def _add_dicts(self, d1, d2):
        c = Counter(d1)
        c.update(d2)
        return dict(c)

    def _process_job_result(self, job_result:Result, batch:Batch) -> Dict[str, Result]:
        """Post-process the job result corresponding to a batch. Recreate the single results by adding up the shots of multiple executions

        Args:
            job_result (Result)
            batch (Batch): corresponding batch to the job_result

        Returns:
            Dict[str, Result]: Maps the keys of the initial QuantumExecutionJobs to their Results
        """
        results = {}
        exp_number = 0
        # get the Result as dict and delete the results 
        result_dict = job_result.to_dict()

        index = batch.batch_number
        backend_name = batch.backend_name
        try:
            previous_key = self._previous_key[backend_name]
            previous_memory = self._previous_memory[backend_name]
            previous_counts = self._previous_counts[backend_name]
        except KeyError:
            previous_key = None
            previous_memory = None
            previous_counts = None
        
        self._log.info(f"Process result of job {index}")

        for exp in batch.experiments:
            key = exp["key"]
            circ = exp["circuit"]
            reps = exp["reps"]
            shots = exp["shots"]
            total_shots = exp["total_shots"]
            memory = []
            counts = {}
            result_data = None

            

            if previous_memory:
                # there is data from the previous job
                assert(previous_key==key)
                memory.extend(previous_memory)
                counts.update(previous_counts)
                shots += len(previous_memory)
                total_shots += len(previous_memory)
                previous_memory = None
                previous_counts = None
                previous_key = None
            
            # get ExperimentResult as dict
            job_exp_result_dict = job_result._get_experiment(exp_number).to_dict() 

            if not (shots == total_shots and reps == 1 and len(memory) == 0):
                # do not run this block if it is only one experiment (shots == total_shots) with one repetition and no previous data is available
                for exp_index in range(exp_number, exp_number+reps):
                    mem = job_result.data(exp_index)['memory']
                    memory.extend(mem)
                    cnts = job_result.data(exp_index)['counts']
                    
                    if exp_index == exp_number+reps-1 and shots == total_shots:
                        # last experiment for this circuit
                        if len(memory) > total_shots:
                            # trim memory and counts w.r.t. number of shots
                            too_much = len(memory) - total_shots
                            memory = memory[:total_shots]
                            mem = mem[:-too_much]
                            cnts = dict(Counter(mem))

                    counts = self._add_dicts(counts, cnts)
                
                if shots < total_shots:
                    previous_memory = copy.deepcopy(memory)
                    previous_counts = copy.deepcopy(counts)
                    previous_key = key
                    continue
                if self._memory:
                    result_data = ExperimentResultData(counts=counts, memory=memory).to_dict()
                else:
                    result_data = ExperimentResultData(counts=counts).to_dict()

                # overwrite the data and the shots
                job_exp_result_dict["data"] = result_data
                job_exp_result_dict["shots"] = total_shots

            else:
                if not self._memory:
                    counts = job_result.data(exp_number)['counts']
                    result_data = ExperimentResultData(counts=counts).to_dict()
                    job_exp_result_dict["data"] = result_data

            # overwrite the results with the computed result
            result_dict["results"] = [job_exp_result_dict]
            results[key] = Result.from_dict(result_dict)
            exp_number += reps
        self._previous_key[backend_name] = previous_key
        self._previous_memory[backend_name] = previous_memory
        self._previous_counts[backend_name] = previous_counts
        return results

    def run(self) -> None:
        self._log.info("Started")
        batch: Batch
        job: Job
        while True:
            batch, job = self._input.get()
            job_result = job.result()
            self._log.info(f"Got result for batch {batch.batch_number} from {batch.backend_name}")
            result_for_batch = self._process_job_result(job_result, batch)
            for key, result in result_for_batch.items():
                try:
                    qjob = self._quantum_job_table.pop(key)
                    qjob.result = result
                    self._output.put(qjob)
                except KeyError as ke:
                    # TODO Exception Handling
                    raise ke

    

class ExecutionHandler():
    
    def __init__(self, provider:AccountProvider, input:Queue, output:Queue, batch_timeout:int = 60, retrieve_interval:int = 30, transpile_timeout=20, max_transpile_batch_size=float('inf'),  submitter_defer_interval=30, provide_memory:bool=False) -> None:
        transpiler_batcher = Queue()
        batcher_submitter = Queue()
        submitter_retrieber = Queue()
        retriever_processor = Queue()
        quantum_job_table = {}
        backend_look_up = BackendLookUp(provider)
        backend_control = BackendControl()
        self._transpiler = Transpiler(input=input, output=transpiler_batcher, backend_look_up=backend_look_up, timeout = transpile_timeout, max_transpile_batch_size=max_transpile_batch_size)
        self._batcher = Batcher(input=transpiler_batcher, output=batcher_submitter, quantum_job_table=quantum_job_table, backend_look_up=backend_look_up, batch_timeout=batch_timeout)
        self._submitter = Submitter(input=batcher_submitter, output=submitter_retrieber, backend_look_up=backend_look_up, backend_control=backend_control, defer_interval=submitter_defer_interval)
        self._retriever = Retriever(input=submitter_retrieber, output=retriever_processor, wait_time=retrieve_interval, backend_control=backend_control)
        self._processor = ResultProcessor(input=retriever_processor, output=output, quantum_job_table=quantum_job_table, memory=provide_memory)
    
    def start(self):
        self._transpiler.start()
        self._batcher.start()
        self._submitter.start()
        self._retriever.start()
        self._processor.start()
