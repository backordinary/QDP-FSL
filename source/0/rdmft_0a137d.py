# https://github.com/pc2/rdmft_on_qc/blob/00a461db1fe12e771e144f854ab65316d3b015d1/rdmft.py
import os
import copy
import math
import sys

os.environ['QISKIT_PARALLEL'] = 'true'

from networkx.generators.directed import random_uniform_k_out_graph
import groundstate
import aca
import mueller
import ci
import time
import measurement_circuits
import numpy as np
import dotenv
import matplotlib.pyplot as plt
import pylatexenc
import configparser
import itertools
import qiskit
from scipy import sparse
from scipy.optimize import minimize, BFGS
from math import pi
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression
from qiskit import IBMQ, assemble, transpile, Aer, BasicAer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import L_BFGS_B, SPSA, COBYLA, QNSPSA
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.opflow.state_fns import CircuitStateFn
from qiskit.quantum_info import Pauli,mutual_information
from qiskit.opflow.gradients import Gradient, NaturalGradient, QFI, Hessian
from qiskit.opflow import Z, X, I, StateFn, CircuitStateFn, SummedOp
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_histogram
import qiskit_nature
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit_nature.operators.second_quantization import FermionicOp
import pickle
import logging
import h5py
import shutil
from qiskit.test.mock import FakeBelem,FakeBogota,FakeSantiago


def opt_callback(nfunc,par,f,stepsize,accepted):
    #print("Opt step:",nfunc,par,f,stepsize,accepted)
    print("Opt step:",nfunc,f,stepsize,accepted)

def qubit_convert_and_map(m,qubit_converter,num_particles,qubit_map):
    #qubit_converter
    m_op = qubit_converter.convert(m,num_particles=num_particles)
    #qubit mapping
    p=m_op.to_pauli_op()
    newsummedop=SummedOp([])
    for op in p:
        #map op with qubit_map
        s=str(op.primitive)
        n=list(str(op.primitive))
        for i in range(len(s)):
            n[len(s)-qubit_map[i]-1]=s[len(s)-i-1]
        print("map",s,n,qubit_map)
        op=PauliOp(Pauli("".join(n)),op.coeff)
        newsummedop+=op
    return newsummedop

def paulis_for_op(m,qubit_converter,qc,qubit_map,num_particles=0):
    m_op = qubit_convert_and_map(m,qubit_converter,num_particles,qubit_map)
    pauliops=[]
    coeff=[]
    const=0

    #build measuring programs here
    p=m_op.to_pauli_op()
    for op in p:
        if str(op).count('I')<len(str(op.primitive)):
            coeff.append(op.coeff)
            pauliops.append(op)
        else:
            const+=op.coeff

    return {"pauliops":pauliops,"coeff":coeff,"const":const}


def measure_all_programs(nq,ansatz,mqcs,x,backend,shots,seed,tsim,optimization_level=3):
    global measurement_counter
    t0=time.time()
    measurement_counter+=1
    #measure all quantum programs
    qcs_par=[]
    q = QuantumRegister(nq)
    c = ClassicalRegister(nq)
    qc=QuantumCircuit(q,c)
    qc=qc.compose(ansatz.bind_parameters(x))
    t1=time.time()

    for mqc in mqcs:
        tcirc = transpile(qc.compose(mqc['mqc']),backend=ibmq_backend, seed_transpiler=seed,optimization_level=optimization_level)
        qcs_par.append(tcirc)
    t2=time.time()
    if not tnoise:
        jobs = execute(
                qcs_par,
                backend=backend,
                backend_properties=backend.properties(),
                shots=shots,
                seed_simulator=seed,
                seed_transpiler=seed,
                optimization_level=optimization_level
            )
    else:
        if tsim:
            #FIXME: eventuell python threading rundrum?
            backend.set_options(method="density_matrix",max_parallel_threads=max_parallel_threads,max_parallel_shots=max_parallel_shots,max_parallel_experiments=max_parallel_experiments)
#            jobs = execute(
#                qcs_par,
#                backend=backend,
#                backend_properties=ibmq_backend_props,
#                shots=shots,
#                seed_simulator=seed,
#                seed_transpiler=seed,
#                optimization_level=optimization_level,
#                coupling_map=ibmq_coupling_map,
#                basis_gates=ibmq_basis_gates,
#                noise_model=ibmq_noise_model
#            )
#        else:
        jobs = execute(
                qcs_par,
                backend=backend,
                #backend_properties=ibmq_backend_props,
                shots=shots,
                seed_simulator=seed,
                seed_transpiler=seed,
                optimization_level=optimization_level
        )
    t3=time.time()

    if not tsim:
        print("waiting for job to finish: https://quantum-computing.ibm.com/jobs/"+str(jobs.job_id()))
        jobs.wait_for_final_state(wait=1)
    else:
        jobs.wait_for_final_state(wait=0.01)

    Vs=[]
    if jobs.done():
        t4=time.time()
        if not tsim:
            #write measurement result to file
            with open('experiment_'+jobs.job_id()+"_qcs", 'wb') as outfile:
                pickle.dump(qcs_par, outfile)
            with open('experiment_'+jobs.job_id()+"_job", 'wb') as outfile:
                pickle.dump(jobs, outfile)
#        else:
#            print(mutual_information())

        res=jobs.result().results
        #loop over programs
        for im in range(len(mqcs)):
            if mqcs[im]['mode']=="single_qubit":
                Va=[]
                #loop over measurements in program
                for ic in range(len(mqcs[im]['ops'])):
                    #print("op=",mqcs[im]['ops'][ic],"is measured at",mqcs[im]['mqubits'][ic],mqcs[im]['signs'][ic])
                    V=0
                    for r in res[im].data.counts:
                        b=bin(int(r, base=16))[2:].zfill(nq)
                        v=res[im].data.counts[r]/shots
                        #if b[nq-1-mqcs[im]['mqubits'][ic]]=='1':
                        if b[nq-1-ic]=='1':
                            V=V-v
                        else:
                            V=V+v
                    V=V*mqcs[im]['signs'][ic]
                    Va.append(V)
                Vs.append(Va)
            elif mqcs[im]['mode']=="bitstring":
                #loop over measurements in program
                Va=[]
                for ic in range(len(mqcs[im]['ops'])):
                    V=0
                    for r in res[im].data.counts:
                        b=bin(int(r, base=16))[2:].zfill(nq)
                        #print("counts",r,b,res[im].data.counts[r]/shots)
                        #find sign of bitstring b
                        sign=1
                        for i in range(nq):
                            if b[i]=="1" and mqcs[im]['ops'][ic][i]!="I":
                                sign=sign*(-1)
                        V=V+sign*res[im].data.counts[r]/shots
                    Va.append(V)
                Vs.append(Va)
            #print(mqcs[im]['ops'],Va)
#        if thdf5_out:
#            try:
#                h5f = h5py.File('measurements.h5', 'a')
#                h5f.create_dataset('opt_x_'+str(measurement_counter), data=x)
#                for i in range(len(Vs)):
#                    h5f.create_dataset('opt_V_'+str(measurement_counter)+"_"+str(i), data=Vs[i])
#                h5f.close()
#            except BlockingIOError:
#                print("hdf5 output is blocking")
        #print("t_measure",t1-t0,t2-t1,t3-t2,t4-t3)
#    quit()
    else:
        raise RuntimeError("some error running the circuit has occoured")
    return Vs

def measurements_to_interact(mqcs,v,pauli_interact):
    W=pauli_interact['const']

    for ic in range(len(pauli_interact['pauliops'])):
        op=str(pauli_interact['pauliops'][ic].primitive)
        #find measurement in mqcs
        found=False
        for im in range(len(mqcs)):
            if found:
                break
            for io in range(len(mqcs[im]['ops'])):
                if op==mqcs[im]['ops'][io]:
                    W=W+v[im][io]*pauli_interact['coeff'][ic]
                    found=True
                    break
    return W

Urot_D=[]
Urot_constraints=[]

def measurements_to_constraints(mqcs,v,constraints,tprint=False,returnUrot=False):
    c=np.zeros(len(constraints))
    for ic in range(len(constraints)):
        cv=constraints[ic]['pauli']['const']-constraints[ic]['cval']
        for ip in range(len(constraints[ic]['pauli']['pauliops'])):
            found=False
            op=str(constraints[ic]['pauli']['pauliops'][ip].primitive)
            #find measurement in mqcs
            for im in range(len(mqcs)):
                if found:
                    break
                for io in range(len(mqcs[im]['ops'])):
                    if op==mqcs[im]['ops'][io]:
                        cv=cv+v[im][io]*constraints[ic]['pauli']['coeff'][ip]
                        found=True
                        break
        c[ic]=cv
    if auto_opt_bath_U:
        return Urot_run(constraints,c,tprint,returnUrot)
    else:
        return c

def Urot_run(constraints,c,tprint=False,returnUrot=False):
      #FIXME orbital count hardcoded
      norb=4
      ninteract=2

      before=np.sum(c**2)
#      if tprint:
#          print("before Urot",before)
      D=c_to_rdm(constraints,c)
      s=""
      N=0
      [f,v]=np.linalg.eig(D)
      for i in range(norb):
          s=s+" "+str(f.real[i])
          N=N+f.real[i]

      #find unitary transformation so that constraint violation is minimal
      neff=norb-ninteract
      np.random.seed(832476)
      x0=0.01*np.random.random(neff**2)

      global Urot_D,Urot_constraints
      Urot_D=copy.deepcopy(D)
      Urot_constraints=copy.deepcopy(constraints)
    
      method="BFGS"
      res=minimize(Urot_obj, x0, method=method,tol=1e-9,options={'maxiter':10000})
      point=res.x
      nfev=res.nfev

      D2=aca.mitigate_trans(point,D,norb,ninteract)
      #[f,v]=np.linalg.eig(D)
      #print("after f=",f)
      #build constraint values from transformed 1rdm
      c2=rdm_to_c(constraints,D2)
      after=np.sum(c2**2)
#      if tprint:
#        print("after Urot",after)
      if returnUrot:
        d={}
        d['c']=copy.deepcopy(c2)
        d['cin']=copy.deepcopy(c)
        d['before']=before
        d['after']=after
        d['x']=res.x
        d['N']=N
        d['f']=f.real
        return d
      else:
        return copy.deepcopy(c2)

def Urot_obj(x):
      #FIXME orbital count hardcoded
      D2=aca.mitigate_trans(x,Urot_D,4,2)
      c=rdm_to_c(Urot_constraints,D2)
      L=0
      for ic in range(len(Urot_constraints)):
        L=L+lagrange[ic]*c[ic]+0.5*penalty[ic]*abs(c[ic])**penalty_exponent
      return L


def c_to_rdm(constraints,c):
    #FIXME orbital count hardcoded
    D=np.zeros((4,4),dtype=np.complex_)
    for ic in range(len(constraints)):
      if constraints[ic]['observable']=="1RDM":
        if constraints[ic]['type']=="real":
          i=constraints[ic]['i']
          j=constraints[ic]['j']
          D[i,j]=D[i,j]+c[ic]+constraints[ic]['cval']
          if i!=j:
            D[j,i]=D[j,i]+c[ic]+constraints[ic]['cval']
        elif constraints[ic]['type']=="imag":
          i=constraints[ic]['i']
          j=constraints[ic]['j']
          D[i,j]=D[i,j]+1j*(c[ic]+constraints[ic]['cval'])
          D[j,i]=D[j,i]-1j*(c[ic]+constraints[ic]['cval'])
    return D

def rdm_to_c(constraints,D):
    c=np.zeros(len(constraints))
    for ic in range(len(constraints)):
      if constraints[ic]['observable']=="1RDM":
        if constraints[ic]['type']=="real":
          i=constraints[ic]['i']
          j=constraints[ic]['j']
          c[ic]=D[i,j].real-constraints[ic]['cval']
        elif constraints[ic]['type']=="imag":
          i=constraints[ic]['i']
          j=constraints[ic]['j']
          c[ic]=D[i,j].imag-constraints[ic]['cval']
    return c



def rdmf_obj(x):
    global rdmf_obj_eval,c_qc,W_qc
    rdmf_obj_eval+=1
    t0 = time.time()
    c_qc=np.zeros(len(constraints))
    W_qc=0
    L=0

    circ=qc.bind_parameters(x)
    job=execute(circ,backend_check)
    result=job.result()
    psi=result.get_statevector()
    #build constraint values from results
    for ic in range(len(constraints)):
        if not build_sparse:
            raise RuntimeError("build_sparse not enabled")
        a=np.dot(np.conj(psi),constraints[ic]['opsparse'].dot(psi)).real
        c_qc[ic]=a-constraints[ic]['cval']
    if auto_opt_bath_U:
        c_qc=Urot_run(constraints,c_qc)
    for ic in range(len(constraints)):
      L=L+lagrange[ic]*c_qc[ic]+0.5*penalty[ic]*abs(c_qc[ic])**penalty_exponent
    #build interaction expectation value from result
    if not build_sparse:
        raise RuntimeError("build_sparse not enabled")
    a=np.dot(np.conj(psi),interact_sparse.dot(psi)).real
    W_qc=a
    L=L+W_qc

    L2=L
    W2_qc=W_qc
    c2_qc=copy.deepcopy(c_qc)

    if not texact_expect:
        L=0
        v=measure_all_programs(nq,ansatz,mqcs,x,backend,shots,seed+rdmf_obj_eval,tsim,optimization_level)
        W_qc=measurements_to_interact(mqcs,v,pauli_interact)
        Urot=measurements_to_constraints(mqcs,v,constraints,tprint=True,returnUrot=True)
        c_qc=Urot['c']
        for ic in range(len(constraints)):
            L=L+lagrange[ic]*c_qc[ic]+0.5*penalty[ic]*abs(c_qc[ic])**penalty_exponent
        L=L+W_qc
        if thdf5_out:
            try:
                h5f = h5py.File('opt_iter.h5', 'a')
                grp = h5f.create_group(str(rdmf_obj_eval))
                grp.create_dataset('x', data=x)
                grp.create_dataset('seed', data=seed+rdmf_obj_eval)
                grp.create_dataset('cqc', data=c_qc)
                grp.create_dataset('wqc', data=W_qc)
                for i in range(len(v)):
                    grp.create_dataset('opt_v_'+str(i), data=v[i])
                grp.create_dataset('L', data=L)
                grp.create_dataset('penalty', data=penalty)
                grp.create_dataset('lagrange', data=lagrange)
                grp.create_dataset('measurement', data=measurement_counter)
                grp.create_dataset('Urot_before', data=Urot['before'])
                grp.create_dataset('Urot_after', data=Urot['after'])
                grp.create_dataset('Urot_x', data=Urot['x'])
                grp.create_dataset('Urot_c', data=Urot['c'])
                grp.create_dataset('Urot_cin', data=Urot['cin'])
                grp.create_dataset('N', data=Urot['N'])
                grp.create_dataset('f', data=Urot['f'])
                h5f.close()
            except BlockingIOError:
                print("hdf5 output is blocking")

    t1 = time.time()
    if tprintevals and rdmf_obj_eval%printevalsevery==0:
        print("rdmf_obj_eval=",rdmf_obj_eval,"L=",L,"W=",W_qc,"sum(c^2)=",np.sum(c_qc**2),"t=",t1-t0,"exact=","L=",L2,"W=",W2_qc,"sum(c^2)=",np.sum(c2_qc**2))
    return L

def rdmf_cons(x):
    global rdmf_cons_eval
    c_qc=np.zeros(len(constraints))
    rdmf_cons_eval+=1

    L=0
    if texact_expect:
        circ=qc.bind_parameters(x)
        job=execute(circ,backend_check)
        result=job.result()
        psi=result.get_statevector()
        #build constraint values from results
        for ic in range(len(constraints)):
            if not build_sparse:
                raise RuntimeError("build_sparse not enabled")
            a=np.dot(np.conj(psi),constraints[ic]['opsparse'].dot(psi)).real
            c_qc[ic]=a-constraints[ic]['cval']
    else:
        raise RuntimeError("rdmft_cons must not be used with tsampling=True or tsim=False")
    return c_qc

#main code

dotenv.load_dotenv()
apikey=os.getenv("QISKIT_APIKEY")

measurement_counter=0

config = configparser.ConfigParser()
config.sections()
config.read(sys.argv[1])

seed=int(config['rnd']['seed'])

for f in ["measurements.h5","opt_iter.h5"]:
    t=time.time()
    if os.path.isfile(f):
        f2=f+"_"+str(t)
        shutil.move(f,f2)

build_sparse=config.getboolean('QC','build_sparse')
commutation_mode_1rdm=config['QC']['commutation_1rdm']
commutation_mode_W=config['QC']['commutation_W']
if config.getboolean('QC','two_qubit_reduction'):
    raise RuntimeError("two_qubit_reduction no longer implemented")
two_qubit_reduction=False

tdoqc=config.getboolean('QC','tdoqc')
tsim=config.getboolean('QC','tsim')
tsimcons=config.getboolean('QC','tsimcons')
tsampling=config.getboolean('QC','tsampling')
tnoise=config.getboolean('QC','tnoise')
if tnoise:
    tsampling=True

no_multipliers_for_up_down=config.getboolean('QC','no_multipliers_for_up_down')

thdf5_out=config.getboolean('QC','hdf5_out')
tcheck=config.getboolean('QC','tcheck')
shots=int(config['QC']['shots'])
systemtype=config['system']['type']
L=int(config['system']['L'])
Lint=int(config['system']['Lint'])
if Lint==-1:
    Lint=L
Llocal=int(config['LocalApprox']['Llocal'])
ilocal=int(config['LocalApprox']['ilocal'])
mitigate_extreme_offdiagonal_1rdm_values=config.getboolean('QC','mitigate_extreme_offdiagonal_1rdm_values')
auto_opt_bath_U=config.getboolean('QC','auto_opt_bath_U')
np.random.seed(seed)

max_parallel_threads=int(config['QC']['max_parallel_threads'])
max_parallel_experiments=int(config['QC']['max_parallel_experiments'])
max_parallel_shots=int(config['QC']['max_parallel_shots'])

rdmf_obj_eval=0


#GS of system to get a physical density matrix
if systemtype=='hubbard_chain':
    U=float(config['system']['U'])
    t=float(config['system']['t'])

    if config['system']['mu']=="mU2":
        mu=-U/2
    else:
        mu=float(config['system']['mu'])
    [E,D,W]=groundstate.hubbard_chain(L,t,U,mu,Lint,mode="DMRG")
elif systemtype=='hubbard_2d':
    U=float(config['system']['U'])
    t=float(config['system']['t'])
    if config['system']['mu']=="mU2":
        mu=-U/2
    else:
        mu=float(config['system']['mu'])
    [E,D,W]=groundstate.hubbard_2d(L,t,U,mu,Lint,mode="DMRG")
else:
    print("system type not implemented")
    exit();


print("E=",E)
print("D=",D)
print("W=",W)

norb=2*L

if False:
    for i in range(norb):
        for j in range(i,norb):
            if i==j:
                D[i,i]=D[i,i]+0.01*(np.random.random(1)-0.5)
            else:
                D[i,j]=D[i,j]+0.01*(np.random.random(1)-0.5)
                D[i,j]=D[i,j]+0.01j*(np.random.random(1)-0.5)
                D[j,i]=np.conj(D[i,j])
    

#apply local approximation and ACA
print("local approximation with ",Llocal,"sites")
Wlocal=[]
W=[]
I=0
for i in range(L):
    if len(W)==Llocal:
        Wlocal.append(W)
        W=[]
    W.append(I)
    I=I+1
Wlocal.append(W)

print("local interactions on sites",Wlocal)

print("Functional for interaction on sites",Wlocal[ilocal])
orbinteract=[]

#set up functional to be solved
#reorder orbitals

orbinteract=[]
ninteract=0
for j in Wlocal[ilocal]:
    ninteract+=2
    orbinteract.append(2*j)
    orbinteract.append(2*j+1)
aca.printmat(norb,norb,"D_in_"+str(ilocal),D)

print("interacting orbitals:",orbinteract)

[Dreordered,orbinteractaca]=aca.aca_reorder(norb,orbinteract,D,orbinteract)
aca.printmat(norb,norb,"D_reordered_"+str(ilocal),Dreordered)
print("orbinteractaca=",orbinteractaca)

Daca=Dreordered.copy()
Daca_levels=[]
#Daca_levels.append(np.copy(Daca[0:ninteract,0:ninteract]))

for l in range(0,int(norb/ninteract-1)):
    Dacain=np.zeros((norb-l*ninteract,norb-l*ninteract),dtype=np.complex_)
    Dacain[::,::]=np.copy(Daca[l*ninteract::,l*ninteract::])
    Dacaout=aca.aca(norb-l*ninteract,ninteract,Dacain)
    Daca[l*ninteract::,l*ninteract::]=np.copy(Dacaout)
    aca.printmat(norb,norb,"D_aca_"+str(l),Daca)
    aca.printmat((l+1)*ninteract,(l+1)*ninteract,"D_aca_truncated_"+str(l),Daca[0:(l+1)*ninteract,0:(l+1)*ninteract])
    Daca_levels.append(np.copy(Daca[0:(l+1)*ninteract,0:(l+1)*ninteract]))
Daca_levels.append(np.copy(Daca))

l=int(config['ACA']['level'])
if l==-1:
    l=len(Daca_levels)-1
Daca=Daca_levels[l]

#limit U/t->\infty
#Daca=0*Daca_levels[l]
#Daca[0,0]=0.5
#Daca[1,1]=0.5
#Daca[2,2]=0.5
#Daca[3,3]=0.5

norb_aca=np.shape(Daca)[0]
aca.printmat(norb_aca,norb_aca,"D_aca_used_",Daca)
print("aca: level",l," norb=",norb_aca)

print("Daca=",Daca)
[f,v]=np.linalg.eig(Daca)
print("Occ(Daca)=",f.real)

if mitigate_extreme_offdiagonal_1rdm_values:
    print("trying to mitigate extreme off-diagonal values in the 1rdm between the impurity and bath and in the bath")
    Din=copy.deepcopy(Daca)
    Daca=aca.mitigate_extreme(norb_aca,ninteract,Din)
    print("Daca after mitigation=",Daca)


#count spin-up and spin-down electrons
Naca1=0
Naca2=0
for i in range(int(norb_aca/2)):
    Naca1+=abs(Daca[2*i,2*i])
    Naca2+=abs(Daca[2*i+1,2*i+1])

if abs(Naca1-round(Naca1))<1e-4:
    print("rounding Naca1 to integer")
    Naca1=round(Naca1)
if abs(Naca2-round(Naca2))<1e-4:
    print("rounding Naca2 to integer")
    Naca2=round(Naca2)
Naca=(Naca1,Naca2)
print("Naca=",Naca)



tdoci=config.getboolean('CI','tdoci')
tder=config.getboolean('CI','tder')
if tdoci:
    tcplx=config.getboolean('CI','tcplx')
    options={'tol': float(config['CI']['tol']),'maxiter': int(config['CI']['maxiter']),"tder":tder}
    Faca_local=ci.F_hubbard(norb_aca,U,orbinteractaca,Daca,options,tcplx)
    print("F_aca (full CI space)=",Faca_local)

print("FM=",mueller.mueller_hubbard(norb_aca,ninteract,U,Daca))
mueller.mueller_hubbard_der(norb_aca,ninteract,U,Daca)

if tdoci:
    tcplx=config.getboolean('CI','tcplx')
    options={'tol': float(config['CI']['tol']),'maxiter': int(config['CI']['maxiter']),"tder":False}
    Faca_local=ci.F_hubbard(norb_aca,U,orbinteractaca,Dreordered,options,tcplx)
    print("F_reordered (full CI space)=",Faca_local)

qubit_converter = QubitConverter(mapper=ParityMapper())
mapping=config['QC']['mapping']
if mapping=="Parity":
    qubit_converter = QubitConverter(mapper=ParityMapper(),two_qubit_reduction=two_qubit_reduction)
elif mapping=="JordanWigner":
    qubit_converter = QubitConverter(mapper=JordanWignerMapper(),two_qubit_reduction=two_qubit_reduction)
elif mapping=="BravyiKitaev":
    qubit_converter = QubitConverter(mapper=BravyiKitaevMapper(),two_qubit_reduction=two_qubit_reduction)
else:
    print("fermionic mapping unknown")
    exit()




# set the backend for the quantum computation
if tnoise:
    # Build noise model from backend properties
    IBMQ.save_account(apikey,overwrite=True)
    # ['ibmq_qasm_simulator', 'ibmq_armonk', 'ibmq_santiago', 'ibmq_bogota', 'ibmq_lima', 'ibmq_belem', 'ibmq_quito', 'simulator_statevector', 'simulator_mps', 'simulator_extended_stabilizer', 'simulator_stabilizer', 'ibmq_manila']
    
    IBMQ.load_account()
    print("providers=",IBMQ.providers())
    provider = IBMQ.get_provider(hub=config["QC"]['hub'],group=config["QC"]['group'])
    print("backends=")
    for b in provider.backends():
        print(b.name())
    ibmq_backend = provider.get_backend(config['QC']['qc'])
    backend = provider.get_backend(config['QC']['qc'])

    if tsim:
        backend=AerSimulator.from_backend(ibmq_backend)

#    if config['QC']['qc'].find("fake")==0:
#        if config['QC']['qc']=="fake_belem":
#            ibmq_backend = FakeBelem()
#        elif config['QC']['qc']=="fake_bogota":
#            ibmq_backend = FakeBogota()
#        elif config['QC']['qc']=="fake_santiago":
#            ibmq_backend = FakeSantiago()
#        else:
#            raise RuntimeError("fake qc not known please add in rdmft.py")
#    else:


    ibmq_backend_config = ibmq_backend.configuration()

    if ibmq_backend_config.n_qubits < norb_aca:
        raise RuntimeError("chosen qc doesn't have enough qubits")

    # >>> backend_config.basis_gates
    # ['id', 'rz', 'sx', 'x', 'cx', 'reset']
    ibmq_coupling_map = ibmq_backend_config.coupling_map

    #set coupling map for transpiler that is used in the construction of measurement circuits
    couplingstring=""
    for ic in range(len(ibmq_coupling_map)-1):
        couplingstring+=str(ibmq_coupling_map[ic][0])+"_"+str(ibmq_coupling_map[ic][1])+","
    ic=len(ibmq_coupling_map)-1
    couplingstring+=str(ibmq_coupling_map[ic][0])+"_"+str(ibmq_coupling_map[ic][1])
    config['QC']['transpiler_couplings']=couplingstring
    print("setting transpiler_couplings to",couplingstring)

    #set gates for transpiler that is used in the construction of measurement circuits
    config['QC']['transpiler_gates']=",".join(ibmq_backend_config.basis_gates)
    print("setting transpiler_gates to",",".join(ibmq_backend_config.basis_gates))

    # [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]
    # Below: directed graph specifying fixed coupling. Nodes correspond to physical qubits (integers) and directed edges correspond to permitted CNOT gates.
    # cm = qiskit.transpiler.CouplingMap(coupling_map)
    # >>> cm.distance_matrix
    # array([[0., 1., 2., 2., 3.],
    #        [1., 0., 1., 1., 2.],
    #        [2., 1., 0., 2., 3.],
    #        [2., 1., 2., 0., 1.],
    #        [3., 2., 3., 1., 0.]])
    ibmq_backend_props = ibmq_backend.properties()
    # nqubits = len(backend_props.qubits)

    # Noise....
    ibmq_noise_model = NoiseModel.from_backend(ibmq_backend)
    ibmq_basis_gates = ibmq_noise_model.basis_gates
    # ibmq_quito: ['cx', 'id', 'reset', 'rz', 'sx', 'x'] ('cx' ist the only two-qubit gate)

#    #backend = BasicAer.get_backend('qasm_simulator')
#    backend = Aer.get_backend('qasm_simulator')
#    backend.set_options(
#            method="density_matrix",
#            max_parallel_threads=0,
#            max_parallel_experiments=0,
#            max_parallel_shots=1
#    )
    #backend = QasmSimulator.from_backend(ibmq_backend)
    optimization_level=3
else:
    #backend = BasicAer.get_backend('statevector_simulator')
    backend = Aer.get_backend('aer_simulator_statevector')
    optimization_level=3

#get number of qubits
nq=config["QC"]["nq"]
if nq=="qc":
    if not tnoise:
        raise RuntimeError("nq=qc only possible for tnoise=True")
    nq=ibmq_backend_config.n_qubits
else:
    nq=int(nq)

print("nq=",nq)



#get qubit mapping
qmap=config['QC']['qubit_map']
qubit_map=[]
if qmap=="none":
    for i in range(nq):
        qubit_map.append(i)
else:
    for i in range(nq):
        qubit_map.append(int(qmap.split(",")[i]))
print("qubit_map",qubit_map)

print("only physical qubits",qubit_map[0:norb_aca],"are used")

#set entangler map in hardware-effcieint trial state
entanglement=config['QC']['entanglement']
reps=int(config['QC']['reps'])
entanglement_blocks=config['QC']['entanglement_blocks']
rotation_blocks=config['QC']['rotation_blocks'].split(",")

if entanglement=="map":
    entanglement_map=config['QC']['entanglement_map']
    entanglement=[]
    for e in entanglement_map.split(","):
        a=int(e.split("_")[0])
        b=int(e.split("_")[1])
        if a in qubit_map[0:norb_aca] and b in qubit_map[0:norb_aca]:
            entanglement.append((a,b))

elif entanglement=="qc":
    #get entangler from hardware coupling map
    if not tnoise:
        raise RuntimeError("entanglement=qc requires tnoise=True")
    entanglement=[]
    for c in ibmq_coupling_map:
        if [c[1],c[0]] in entanglement:
            continue
        if c[0] in qubit_map[0:norb_aca] and c[1] in qubit_map[0:norb_aca]:
            entanglement.append([c[0],c[1]])
    #entanglement=[[0,1],[2,3],[1,2],[1,0],[3,2]]
    entanglement=[[0,1],[2,3],[1,2]]
    print("using entangler (physical qubits) =",entanglement)

backend_check = BasicAer.get_backend('statevector_simulator')
#backend_check = Aer.get_backend('aer_simulator_statevector')

if not tsim:
    provider = IBMQ.load_account()
    backend = provider.get_backend(config['QC']['qc'])
    backend_config = ibmq_backend.configuration()
    status = backend.status()

    if backend_config.n_qubits < norb_aca:
        raise RuntimeError("chosen qc doesn't have enough qubits")
    if not status.operational:
        raise RuntimeError("chosen qc is not operational")

num_particles=Naca

up=0
dn=1

c_ops=[]
for i in range(norb_aca):
    c_ops.append(FermionicOp("-_"+str(i),register_length=nq))

#build interaction operator
interact=0
Waca=[]
for i in range(int(len(orbinteract)/2)):
    Waca.append(i)

for i in Waca:
    interact+=U*((~c_ops[2*i]) @ c_ops[2*i]@(~c_ops[2*i+1]) @ c_ops[2*i+1])



ansatztype=config['QC']['ansatztype']
q = QuantumRegister(nq)
c = ClassicalRegister(nq)
ansatz=QuantumCircuit(q,c)
ipar=0
npar=0
# setup the initial state for the ansatz
if ansatztype=="twolocal":
    ansatz = TwoLocal(nq,rotation_blocks = rotation_blocks, entanglement_blocks = entanglement_blocks,entanglement=entanglement, reps=reps, parameter_prefix = 'y',insert_barriers=False,skip_unentangled_qubits=True)
    npar=ansatz.num_parameters
elif ansatztype=="hardwarefficient0":
    for irep in range(reps):
        #rotation layer
        for iq in range(norb_aca):
            theta = Parameter('y['+str(ipar)+"]")
            ansatz.rz(theta,iq)
            ipar=ipar+1
        #sx layer
        for iq in range(norb_aca):
          ansatz.sx(iq)
        for iq in range(norb_aca):
            theta = Parameter('y['+str(ipar)+"]")
            ansatz.rz(theta,iq)
            ipar=ipar+1
        
        #entangler layer
        for e in entanglement:
            ansatz.cx(e[0],e[1])

    for iq in range(norb_aca):
        theta = Parameter('y['+str(ipar)+"]")
        ansatz.rz(theta,iq)
        ipar=ipar+1
    for iq in range(norb_aca):
      ansatz.sx(iq)
    for iq in range(norb_aca):
        theta = Parameter('y['+str(ipar)+"]")
        ansatz.rz(theta,iq)
        ipar=ipar+1
    npar=ipar
elif ansatztype=="hardwarefficient_u3":
    for irep in range(reps):
        #rotation layer
        for iq in range(norb_aca):
            t1 = Parameter('y['+str(ipar)+"]")
            t2 = Parameter('y['+str(ipar+1)+"]")
            t3 = Parameter('y['+str(ipar+2)+"]")
            ansatz.u3(t1,t2,t3,iq)
            ipar=ipar+3
        #sx layer
        for iq in range(norb_aca):
          ansatz.sx(iq)
        for iq in range(norb_aca):
            t1 = Parameter('y['+str(ipar)+"]")
            t2 = Parameter('y['+str(ipar+1)+"]")
            t3 = Parameter('y['+str(ipar+2)+"]")
            ansatz.u3(t1,t2,t3,iq)
            ipar=ipar+3
        
        #entangler layer
        for e in entanglement:
            ansatz.cx(e[0],e[1])

    for iq in range(norb_aca):
        t1 = Parameter('y['+str(ipar)+"]")
        t2 = Parameter('y['+str(ipar+1)+"]")
        t3 = Parameter('y['+str(ipar+2)+"]")
        ansatz.u3(t1,t2,t3,iq)
        ipar=ipar+3
    for iq in range(norb_aca):
      ansatz.sx(iq)
    for iq in range(norb_aca):
        t1 = Parameter('y['+str(ipar)+"]")
        t2 = Parameter('y['+str(ipar+1)+"]")
        t3 = Parameter('y['+str(ipar+2)+"]")
        ansatz.u3(t1,t2,t3,iq)
        ipar=ipar+3
    npar=ipar
elif ansatztype=="2103.16161_spin_model":
    theta = Parameter('y['+str(ipar)+"]")
    ansatz.ry(theta,0)
    ipar=ipar+1

    theta = Parameter('y['+str(ipar)+"]")
    ansatz.ry(theta,1)
    ipar=ipar+1
    
    theta = Parameter('y['+str(ipar)+"]")
    ansatz.ry(theta,3)
    ipar=ipar+1
    
    ansatz.cnot(0,1)
    
    theta = Parameter('y['+str(ipar)+"]")
    ansatz.ry(theta,1)
    ipar=ipar+1
    
    ansatz.cnot(1,2)
    
    theta = Parameter('y['+str(ipar)+"]")
    ansatz.ry(theta,2)
    ipar=ipar+1
    
    ansatz.cnot(2,3)
    
    theta = Parameter('y['+str(ipar)+"]")
    ansatz.ry(theta,2)
    ipar=ipar+1
    
    theta = Parameter('y['+str(ipar)+"]")
    ansatz.ry(theta,3)
    ipar=ipar+1
    
    ansatz.cnot(1,2)
    
    theta = Parameter('y['+str(ipar)+"]")
    ansatz.ry(theta,1)
    ipar=ipar+1
    
    theta = Parameter('y['+str(ipar)+"]")
    ansatz.ry(theta,2)
    ipar=ipar+1
    
    ansatz.cnot(0,1)
    
    theta = Parameter('y['+str(ipar)+"]")
    ansatz.ry(theta,1)
    ipar=ipar+1

    npar=ipar
print(ansatz)



ansatz.draw(output='latex',filename="ansatz.pdf")

#define registers
q = QuantumRegister(nq)
c = ClassicalRegister(nq)
qc=QuantumCircuit(q,c)
qc=qc.compose(ansatz)
#qc=qc.decompose()

print("variational state:")
print(qc)

#transpiler variational state
initial_point = np.random.random(npar)

circ=qc.bind_parameters(initial_point)
for i in range(nq):
    circ.measure(i,i)

transpiled_qc = transpile(circ, backend=ibmq_backend, optimization_level=3,seed_transpiler=seed)
print("transpiled variational state:")
print(transpiled_qc)



print("ansatz-state is written to ansatz.txt")
qc.draw(output='text',filename="ansatz.txt")
#qc.draw(output='mpl',filename="ansatz.png")

if False:
    #gradient test
    q = QuantumRegister(nq)
    qc2=QuantumCircuit(q)
    qc2=qc2.compose(ansatz).decompose()
    #print(ansatz)
    H = qubit_converter.convert(interact,num_particles=num_particles) #qc_interact2['qcs'][i]
    params = ParameterVector('y', length=npar)
    print("params=",params)
    qc2=qc2.assign_parameters(params)

    op = ~StateFn(H) @ CircuitStateFn(primitive=qc2, coeff=1.)
    print("op=",op)
    grad = Gradient(grad_method='param_shift').convert(operator = op, params = params)
    for i in range(len(grad)):
        print("grad=",params[i],grad[i])
    exit()

pauli_interact=paulis_for_op(interact,qubit_converter,qc,qubit_map,num_particles)
print("Interaction")
print(pauli_interact)

auto_opt_bath_U=config.getboolean('QC','auto_opt_bath_U')

if config.getboolean('QC','ignore_W'):
    pauli_interact['pauliops']=[]
interact_op=qubit_convert_and_map(interact,qubit_converter,num_particles,qubit_map)
#interact_op=qubit_converter.convert(interact,num_particles=num_particles)
if build_sparse:
    interact_sparse=sparse.csr_matrix(interact_op.to_pauli_op().to_matrix())

#build list of constraints
print("building constraints...")
constraints=[]
for i in range(norb_aca):
    for j in range(norb_aca):
        if i>j:
            continue
        #real part
        c={}
        c['observable']='1RDM'
        c['type']='real'
        if (i%2==0 and j%2!=0) or (i%2!=0 and j%2==0):
            c['updown']=True
        else:
            c['updown']=False
        c['i']=i
        c['j']=j
        m=~c_ops[i]@ c_ops[j]
        c['op']=0.5*(m+~m)
        op=qubit_convert_and_map(c['op'],qubit_converter,num_particles,qubit_map)
        #op=qubit_converter.convert(c['op'],num_particles=num_particles)
        if build_sparse:
            c['opsparse']=sparse.csr_matrix(op.to_matrix())
        c['pauli']=paulis_for_op(c['op'],qubit_converter,qc,qubit_map,num_particles)
        c['cval']=0.5*(Daca[i,j]+np.conjugate(Daca[i,j])).real
        constraints.append(c)
        if i!=j:
            c={}
            c['observable']='1RDM'
            c['type']='imag'
            if (i%2==0 and j%2!=0) or (i%2!=0 and j%2==0):
                c['updown']=True
            else:
                c['updown']=False
            c['i']=i
            c['j']=j
            m=~c_ops[i]@ c_ops[j]
            c['op']=0.5/1j*(m-~m)
            #op=qubit_converter.convert(c['op'],num_particles=num_particles)
            op=qubit_convert_and_map(c['op'],qubit_converter,num_particles,qubit_map)
            if build_sparse:
                c['opsparse']=sparse.csr_matrix(op.to_matrix())
            c['pauli']=paulis_for_op(c['op'],qubit_converter,qc,qubit_map,num_particles)
            c['cval']=(0.5/1j*(Daca[i,j]-np.conjugate(Daca[i,j]))).real
            constraints.append(c)
        #print("constraint",i,j,c['pauli'])

print("number of constraints=",len(constraints))


############################################################################
#build measurement programs for 1rdm
pauliops=[]
for c in constraints:
    for cc in c['pauli']['pauliops']:
        pauliops.append(str(cc.primitive))
unique_pauliops=measurement_circuits.get_unique_pauliops(pauliops)['unique pauliops']
print("unique pauliops",len(unique_pauliops))
#compute commuting cliques
cliques=measurement_circuits.build_cliques(commutation_mode_1rdm,unique_pauliops)
if config.getboolean('QC','end_after_cliques'):
    exit()
#build measurement circuits for ciques
print("constructing measurement programs for 1rdm with commutativity mode",commutation_mode_1rdm)
mqcs_1rdm=measurement_circuits.build_measurement_circuits(cliques,commutation_mode_1rdm,nq,config)

############################################################################
#build measurement programs for interaction
pauliops=[]
for cc in pauli_interact['pauliops']:
    pauliops.append(str(cc.primitive))
unique_pauliops=measurement_circuits.get_unique_pauliops(pauliops)['unique pauliops']
#compute commuting cliques
cliques=measurement_circuits.build_cliques(commutation_mode_W,unique_pauliops)
#build measurement circuits for ciques
print("constructing measurement programs for interaction with commutativity mode",commutation_mode_W)
mqcs_interact=measurement_circuits.build_measurement_circuits(cliques,commutation_mode_W,nq,config)

#############################################################################
#combine measurement programs for 1rdm and interaction
mqcs=copy.deepcopy(mqcs_1rdm)
mqcs.extend(mqcs_interact)

print("Number of quantum programs:",len(mqcs_1rdm),"for 1rdm and ",len(mqcs_interact),"for interaction")
print(mqcs)

for i in range(len(constraints)):
    print("constraint",i)
    print("constraint observable",constraints[i]['observable'])
    print("constraint i         ",constraints[i]['i'])
    print("constraint j         ",constraints[i]['j'])
    print("constraint type      ",constraints[i]['type'])
    print("constraint cval      ",constraints[i]['cval'])
    print("constraint op        ",constraints[i]['op'])
    print("constraint pauli     ")
    print(constraints[i]['pauli'])


#set intial parameters
np.random.seed(seed)
initial_point = (np.random.random(npar)-0.5)*2*3.141592
#initial_point = np.zeros(npar)

#initial_point=[ 3.36617, 3.18817, 0.509538, 2.61726, 0.866021, -1.65892, -1.25411,                                                                -0.014429, -0.718797, 0.543284, -1.49364, -3.14805, 1.3689,                                                                                                                                                                                                                   2.37435, 1.40662, 0.863992  ]
#initial_point=[ 3.36345563,  3.18878481 , 0.50867812,  2.63822077,  0.74579453, -1.65532061,         -1.25159557, -0.01180757, -0.72536 ,    0.53567368, -1.48668484, -3.14471833,          1.36569587,  2.37699318 , 1.40626512 , 0.86731002]

#result from noisy simulation
#initial_point=[2.647972,  3.265368,  1.348365,   4.758774,   2.775753,   -1.444508,  -0.6095755,  -1.324078,  -1.025389,   0.7769008,     -0.3379025,  -3.089074,   1.578568,   1.581726,   0.3062963 ,   1.767301 ] 

#result from noiseless simulation
#initial_point= [ 2.56444228, 3.40698709,  1.57003307,  5.90891086,  2.57432067, -1.57066289,   -0.62658912, -1.3913296,  -0.90555285,  0.85861476, -0.29549811, -3.14158424,1.57078748,  1.57010815,  0.29546373,  1.57160398]



print("initial_point=")
print(initial_point)

print("number of parameters:",npar)




if not tdoqc:
    exit()

penalty_exponent=float(config["QC"]["penalty_exponent"])
penalty=np.zeros(len(constraints))
penalty[:]=0*penalty[:]+float(config["QC"]["initial_penalty"])
lagrange=np.zeros(len(constraints))

if config.getboolean('QC','initial_lagrange_from_mueller'):
    der=mueller.mueller_hubbard_der(norb_aca,ninteract,U,Daca)
    for i in range(len(constraints)):
        lagrange[i]=-der[i]
print("initial lagrange=",lagrange)

v=measure_all_programs(nq,ansatz,mqcs,initial_point,backend,shots,seed,tsim,optimization_level)
W_qc=measurements_to_interact(mqcs,v,pauli_interact)
c_qc=measurements_to_constraints(mqcs,v,constraints)

if tcheck:
    circ=ansatz.bind_parameters(initial_point)
    job=execute(circ,backend_check)
    result=job.result()
    psi=result.get_statevector()

    W_exact=np.dot(np.conj(psi),interact_sparse.dot(psi)).real
    print("W_exact=",W_exact,"W_qc=",W_qc)

    c_exact=np.zeros(len(constraints))
    for ic in range(len(constraints)):
        c_exact[ic]=np.dot(np.conj(psi),constraints[ic]['opsparse'].dot(psi)).real-constraints[ic]['cval']
        print("constraint",ic," exact=",c_exact[ic],"qc=",c_qc[ic],constraints[ic]['i'],constraints[ic]['j'],constraints[ic]['type'],constraints[ic]['cval'])


tstddev=config.getboolean('QC','tstddev')
if tstddev:
    #compute stddev
    stddev_count=int(config['QC']['stddev_count'])
    res=[]
    for i in range(stddev_count):
        v=measure_all_programs(nq,ansatz,mqcs,initial_point,backend,shots,seed+i,tsim,optimization_level)
        W_qc=measurements_to_interact(mqcs,v,pauli_interact)
        c_qc=measurements_to_constraints(mqcs,v,constraints)
        res.append([W_qc,c_qc])
    W_qc_sum=0
    c_qc_sum=np.zeros(len(constraints))
    W2_qc_sum=0
    c2_qc_sum=np.zeros(len(constraints))
    for i in range(stddev_count):
        W_qc_sum+=res[i][0]
        c_qc_sum[:]+=res[i][1][:]
        W2_qc_sum+=res[i][0]**2
        c2_qc_sum[:]+=res[i][1][:]**2

    W_qc_sum=W_qc_sum/stddev_count
    c_qc_sum[:]=c_qc_sum[:]/stddev_count
    W2_qc_sum=W2_qc_sum/stddev_count
    c2_qc_sum[:]=c2_qc_sum[:]/stddev_count

    print("stddev W=",math.sqrt(W2_qc_sum-W_qc_sum**2))
    for i in range(len(constraints)):
        print("stddev c[",i,"]=",math.sqrt(c2_qc_sum[i]-c_qc_sum[i]**2))

rdmf_obj_eval=0
rdmf_cons_eval=0
tprintevals=True
printevalsevery=1

qiskit.utils.algorithm_globals.random_seed=seed
maxiter=int(config['QC']['algo_maxiter'])
tol=float(config['QC']['algo_tol'])


#if not config.getboolean('QC','min_without_noise'):
#    initial_point=[3.10154661,2.33121779,1.56994994,2.87591172,1.56797337,-1.2759596,-2.5057679,-1.81839961,-1.57081844,0.05640153,-1.60075676,-2.45535814,0.32334325,3.21051636,0.6736446,0.81070058,3.14159784,-1.57079121,1.57151379,-1.37104868]


x0=initial_point
tprintevals=False

if tsim and tsimcons:
    texact_expect=True
    penalty=np.zeros(len(constraints))
    lagrange=np.zeros(len(constraints))
    method="trust-constr"
#    method="SLSQP"
    print("minimizing over parametrized qc-programs with constrained minimization with "+method+" (no derivatives, no sampling, no noise, i.e. exact expectation values)")
    eq_cons={'type': 'eq','fun' : lambda x: rdmf_cons(x)}
    if method=="trust-constr":
        res=minimize(rdmf_obj, x0, method=method, constraints=[eq_cons],tol=1e-4,options={'maxiter':10000,'verbose': 3,'disp': True,'initial_tr_radius':4*pi,'gtol':1e-4,'xtol':1e-4})
    else:
        res=minimize(rdmf_obj, x0, method=method, constraints=[eq_cons],tol=1e-4,options={'maxiter':10000,'verbose': 3,'iprint':2,'disp': True})
    print(res)
    texact_expect=False

print("Augmented Lagrangian")
penalty[:]=0*penalty[:]+float(config["QC"]["initial_penalty"])
print("initial penalty=",penalty)

tprintevals=True
rdmf_obj_eval=0



#augmented Lagrangian
algo=config['QC']['algo']

for oiter in range(int(config['QC']['auglag_iter_max'])):
    value=0
    point=[]
    nfev=0
    #unconstrainted steps
    print("Augmented Lagrangian: unconstrained subproblem")
    if tsim and not tsampling and not tnoise:
        texact_expect=True
        print("minimizing over parametrized qc-programs with augmented Lagrangian and LBFGS_B (numerical derivatives, no sampling, no noise, i.e., exact expectation values)")
        method="BFGS"
        res=minimize(rdmf_obj, x0, method=method,tol=1e-2,options={'maxiter':maxiter,'verbose': 2,'disp': True})
        point=res.x
        value=rdmf_obj(point)
        nfev=res.nfev
    else:
        logging.getLogger('qiskit.algorithms.optimizers.spsa').setLevel('INFO')
        spsa_blocking=config.getboolean('QC','spsa_blocking')
        if config.getboolean('QC','min_without_noise'):
            texact_expect=True
        else:
            texact_expect=False
        if algo=="LBFGS":
            print("minimizing over parametrized qc-programs with augmented Lagrangian and "+algo)
            optimizer = L_BFGS_B(maxiter=maxiter,tol=tol)
            [point, value, nfev]=optimizer.optimize(num_vars=npar,objective_function=rdmf_obj,initial_point=x0)
        if algo=="COBYLA":
            print("minimizing over parametrized qc-programs with augmented Lagrangian and "+algo)
            optimizer = COBYLA(maxiter=maxiter,disp=True,tol=tol,callback=opt_callback)
            [point, value, nfev]=optimizer.optimize(num_vars=npar,objective_function=rdmf_obj,initial_point=x0)
            if config.getboolean('QC','min_without_noise'):
                algo="LBFGS"
        elif algo=="SPSA":
            print("minimizing over parametrized qc-programs with augmented Lagrangian and "+algo)
            optimizer = SPSA(maxiter=maxiter,callback=opt_callback,blocking=spsa_blocking)
            [point, value, nfev]=optimizer.optimize(num_vars=npar,objective_function=rdmf_obj,initial_point=x0)
        elif algo=="cal-SPSA":
            print("minimizing over parametrized qc-programs with augmented Lagrangian and "+algo)
            print("calibrating")
            optimizer = SPSA(maxiter=maxiter,second_order=False,callback=opt_callback,blocking=spsa_blocking)
            [learning_rate,perturbation]=optimizer.calibrate(rdmf_obj,initial_point=x0,stability_constant=0, target_magnitude=None, alpha=0.602, gamma=0.101, modelspace=False)
            print("minimizing")
            optimizer = SPSA(maxiter=maxiter,second_order=False,callback=opt_callback,perturbation=perturbation,learning_rate=learning_rate,blocking=spsa_blocking)
            [point, value, nfev]=optimizer.optimize(num_vars=npar,objective_function=rdmf_obj,initial_point=x0)
        elif algo=="QNSPSA":
            print("minimizing over parametrized qc-programs with augmented Lagrangian and "+algo)
            fidelity = QNSPSA.get_fidelity(ansatz)
            qnspsa = QNSPSA(fidelity, maxiter=maxiter,callback=opt_callback,blocking=spsa_blocking)
            [point,value,nfev] = qnspsa.optimize(npar, rdmf_obj, initial_point=x0)

    print("point=",point)
    print("value=",value)
    print("nfev=",nfev)

    #measure with noise
    stddev_count=int(config['QC']['stddev_count'])
    res=[]
    for i in range(stddev_count):
        v=measure_all_programs(nq,ansatz,mqcs,point,backend,shots,seed+i,tsim,optimization_level)
        W_qc_noise=measurements_to_interact(mqcs,v,pauli_interact)
        c_qc_noise=measurements_to_constraints(mqcs,v,constraints)
        res.append([W_qc_noise,c_qc_noise])
    W_qc_sum=0
    c_qc_sum=np.zeros(len(constraints))
    W2_qc_sum=0
    c2_qc_sum=np.zeros(len(constraints))
    for i in range(stddev_count):
        W_qc_sum+=res[i][0]
        c_qc_sum[:]+=res[i][1][:]
        W2_qc_sum+=res[i][0]**2
        c2_qc_sum[:]+=res[i][1][:]**2
    
    W_qc_sum=W_qc_sum/stddev_count
    c_qc_sum[:]=c_qc_sum[:]/stddev_count
    W2_qc_sum=W2_qc_sum/stddev_count
    c2_qc_sum[:]=c2_qc_sum[:]/stddev_count

    print("stddev W=",math.sqrt(W2_qc_sum-W_qc_sum**2))
    for i in range(len(constraints)):
        print("stddev c[",i,"]=",math.sqrt(c2_qc_sum[i]-c_qc_sum[i]**2))

    if not texact_expect: 
        c_qc=c_qc_sum[:]
        W_qc=W_qc_sum
    else:
      rdmf_obj(point)

    print("W=",W_qc,"W_noise=",W_qc_sum)
    print("constraint violation sum(c^2)=",np.sum(c_qc**2))

    print("Augmented Lagrangian: penalty and multiplier update")
    x0=point[:]
    #multiplier and penalty update
    for i in range(len(constraints)):
        lagrange[i]=lagrange[i]+penalty[i]*c_qc[i]
        penalty[i]=penalty[i]*float(config["QC"]["penalty_factor"])
    for i in range(len(constraints)):
        print("constraint",i,constraints[i]['i'],constraints[i]['j'],constraints[i]['type'],"viol=",c_qc[i],"viol_noise=",c_qc_sum[i],"lagrange=",lagrange[i],"penalty=",penalty[i])




