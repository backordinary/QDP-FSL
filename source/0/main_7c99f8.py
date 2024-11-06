# https://github.com/CS6620-S21/Asynchronous-quantum-job-submission-framework/blob/248f019e9bbc1971df9ffa5964a82037bd86a5f7/async_job/api/main.py
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.responses import JSONResponse
from qiskit.providers import backend
# from controller import Controller

from response_models import Result
from object_store import ObjectStore
from qiskit import IBMQ
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.qobj.qasm_qobj import QasmQobj as QasmQobj
import boto3
from datetime import datetime

import pickle
import json, os
import logging

import numpy

BACKEND = os.getenv("BACKEND")
COMPLETED_BUCKET = os.getenv("COMPLETED_BUCKET")
PENDING_BUCKET = os.getenv("PENDING_BUCKET")

# Initialize ObjectStore
try:
    ob = ObjectStore()
except Exception as ex:
    logging.error("Error is -", ex)

# Initialize Backend 
def init_backend():
    """Returns the backend to work with in the cron job"""
    token=os.getenv('BACKEND_TOKEN')
    IBMQ.save_account(token, overwrite=True)
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')
    IBMQ.get_provider(group='open')
    try:
        backend = provider.get_backend(BACKEND)
    except QiskitBackendNotFoundError as er:
        logging.error("Qiskit Backend not found, please check backend in .env.")
        raise er
    return backend

try:
    backend = init_backend()
except Exception as ex:
    logging.error("Error is -", ex)


def _generateResult(request_body):
    """Generate input json object."""
    result_to_write = {
        "input": request_body,
        "timestamp": str(datetime.now())
    }
    return result_to_write

app = FastAPI()

@app.get("/")
async def homepage(request: Request):
    return("Welcome to Async Job Framework")


getResponses = {
                404: {"result": Result},
                102: {"result": Result}
            }
@app.post("/getResult/", responses=getResponses)
async def getResult(request: Request):
    """Get the job id and try fetching the result."""
    body = await request.json()
    job_id = body['job_id']
    job_id_extension = str(job_id)
    try:
        result = ob.get_object(job_id_extension,COMPLETED_BUCKET)
        # check if result is available or not in completed bucket and if not available check for the job in pending bucket
        if result is None:
            result = ob.get_object(job_id_extension,PENDING_BUCKET)
            if result is None:
                return JSONResponse(status_code=404, content={"result": "No job found with given ID."})
            return JSONResponse(status_code=201, content={"result": "Job pending or being fetched."})
        else:
            return result
    except Exception as ex:
        logging.error("Error is -", ex)


@app.post("/submit/")
async def submit(request: Request):
    """Get the job id and submit the job to the selected backend."""
    body = await request.json()
    recieved_qobj = QasmQobj.from_dict(body)
    try:
        job = backend.run(recieved_qobj)
    except Exception as ex:
        return JSONResponse(status_code=500, content={"Error": "Job couldn't be submitted to the backend.", "Trace": ex})
    jobID = job.job_id()
    input_json = _generateResult(body)
    try:
        ob.put_object(job_body=input_json, file_name=jobID, bucket_name=PENDING_BUCKET)
    except Exception:
        logging.error("Failed to write into completed bucket.")
    return JSONResponse(status_code=200, 
                        content={"job_id": jobID, "backend": BACKEND}
                        )


