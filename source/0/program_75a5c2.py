# https://github.com/PlanQK/planqk-platform-samples/blob/324c4dcfd4467d032454d9dcb9ef2ef95b6e6d4a/notebooks/rng_stock_prediction/src/program.py

"""
Template for implementing services running on the PlanQK platform
"""

from typing import Dict, Any, Optional
from loguru import logger

# Import response wrappers:
# - use ResultResponse to return computation results
# - use ErrorResponse to return meaningful error messages to the caller
from .libs.return_objects import Response, ResultResponse, ErrorResponse

# Import your own libs
import qiskit as q
import time


def run(data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None):
    """
    Default entry point of your code. Start coding here!

    Parameters:
        data (Optional[Dict[str, Any]]): The input data sent by the client
        params (Optional[Dict[str, Any]]): Contains parameters, which can be set by the client for parametrizing the execution

    Returns:
        response: (ResultResponse | ErrorResponse): Response as arbitrary json-serializable dict or an error to be passed back to the client
    """
    response: Response

    try:
        n_numbers = data.get('n_numbers', 10)
        n_bits = params.get('n_bits', 1)
        backend_string = params.get('backend', None)

        circuit = q.QuantumCircuit(n_bits, n_bits)
        circuit.h(range(n_bits))

        # perform measurement:
        circuit.measure(range(n_bits), range(n_bits))

        # choose backend for execution
        if backend_string is None:
            backend = q.Aer.get_backend('qasm_simulator')
        elif backend_string in ['qasm_simulator', 'statevector_simulator']:
            backend = q.Aer.get_backend(backend_string)
        else:
            backend = None
            # should be adapted to include cloud backend-strings as soon as
            # some sort of interceptor framework is available

        max_shots = backend.configuration().max_shots

        if not backend.configuration().simulator:
            max_experiments = backend.configuration().max_experiments
            max_numbers = max_experiments*max_shots
        else:
            max_experiments = 0
            max_numbers = 10_000_000

        if max_experiments > 0 and n_numbers > max_numbers:
            logger.info(f"## Only {max_numbers} numbers possible for one run ##")
            logger.info(f"## Set n_numbers to {max_numbers} ##")
            n_numbers = max_numbers

        # create list of circuits if more random numbers are necessary than shots are allowed
        # even if only max_shots + 1 numbers are needed the circuit will be evaluated 2*max_shots
        # in order to send only one job (all circuits need the same amount of shots) 
        if n_numbers > max_shots:
            reps = n_numbers//max_shots + 1  
            circ_list = [circuit]*reps
            shots = max_shots
            logger.info(f"## Executing RNG-circuit {reps} times with {max_shots} shots each ##") 
            logger.info(f"## in order to generate {n_bits} numbers ##")
        else:
            reps = 1
            circ_list = [circuit]
            shots = n_numbers
            logger.info(f"## Executing RNG-circuit 1 time with {shots} shots ##")

        # execute the circuit on the chosen qiskit backend
        start_time = time.time()
        job = q.execute(circ_list, backend = backend, shots = shots, memory = True)

        # list of binary random variables
        if reps == 1:
            rng_bin_list = job.result().get_memory()
        else:
            # include results from first execution that are a remainder of multiple of max_shots
            rng_bin_list = job.result().get_memory(0)[:n_numbers%max_shots]
            for run_idx in range(1,reps):
                rng_bin_list += job.result().get_memory(run_idx)
        
        logger.info('## Finished Execution ##')
        
        exec_time = time.time()-start_time

        # transform into decimals
        rng_list = [int(binary, 2) for binary in rng_bin_list]

        result = {
            'random_number_list': rng_list
        }
        metadata = {
            'execution_time': round(exec_time, 3),
            'avg_circuit_execution_time': round(exec_time/reps, 3)
        }
        
        logger.info("## Calculation successfully executed ##")
        logger.info(result)
        
        return ResultResponse(result=result, metadata=metadata)
    except Exception as e:
        
        return ErrorResponse(code="500", detail=f"{type(e).__name__}: {e}")

