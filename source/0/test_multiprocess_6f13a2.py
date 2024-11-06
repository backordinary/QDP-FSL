# https://github.com/fatginger1024/PQC_Descriptor/blob/ebc57c542eb5448e8e4c0e1ff07637a4a9e4bc5f/tests/test_multiprocess.py
import numpy as np
import ctypes
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from functools import partial
from multiprocessing import Array, Pool
#from pqc import Multiprocess


class Multiprocess:
    
    def __init__(self,circ=None,samples:int=10,num_proc:int=1,p_in=[0],t_in=[0]):
        self._samples = samples
        self._circ = circ
        self._num_params = circ.num_parameters
        self._num_proc = num_proc
        self.p = p_in
        self.t = t_in
        

    def init(self,arr1_base,arr2_base):
        global arr1, arr2
        arr1 = np.ctypeslib.as_array(arr1_base.get_obj())
        arr1 = arr1.view(np.complex128).reshape(self._samples, self._num_params*2)
        arr2 = np.ctypeslib.as_array(arr2_base.get_obj())
        arr2 = arr2.view(np.complex128).reshape(self._samples, self._num_params*2)

    # Parallel processing
    def target(self,func,i,p1,p2):
        """
        p1: theta
        p2: phi
        """
        out1 = func(p1)
        out2 = func(p2)
        arr1[i] = out1
        arr2[i] = out2
        assert arr1[i][:self._num_params].all() == self.p[i].all()
        assert arr2[i][:self._num_params].all() == self.t[i].all()
        #print("i=: ",i)
        #print("arr1",arr1[i])
        #print("arr2",arr2[i])
        
       
    def job(self,func,iterable):
        arr1_base = Array(ctypes.c_double, self._samples*self._num_params*4)
        arr2_base = Array(ctypes.c_double, self._samples*self._num_params*4)
        pool = Pool(processes=self._num_proc, initializer=self.init, initargs=(arr1_base,arr2_base,))
        pool.starmap(partial(self.target,func), iterable)
        pool.close() 
        pool.join()
        arr1 = np.ctypeslib.as_array(arr1_base.get_obj())
        arr1 = arr1.view(np.complex128).reshape(self._samples, self._num_params*2)
        arr2 = np.ctypeslib.as_array(arr2_base.get_obj())
        arr2 = arr2.view(np.complex128).reshape(self._samples, self._num_params*2)
        #print("check: ",arr1)
        return arr1, arr2
    
    
    
class test_multiprocess(Multiprocess):
    
    def __init__(self,circ=None,samples:int=10,num_proc:int=2,p_in=[0],t_in=[0]):
        
        Multiprocess.__init__(self,circ,samples,num_proc,p_in,t_in)
        
    @staticmethod
    def func(p):
        
        return np.concatenate((p,p),axis=None)

    

def test():
    Num = 4
    num_params = 8
    qc = QuantumCircuit(Num)
    x = ParameterVector(r'$\theta$', length=num_params)
    [qc.h(i) for i in range(Num)]   
    [qc.ry(x[int(2*i)], i) for i in range(Num)]
    [qc.rz(x[int(2*i+1)], i) for i in range(Num)]
    qc.cx(0, range(1, Num))
    circ = qc
    num_sample = 3
    p = np.random.random(num_sample*num_params)+np.random.random(num_sample*num_params)*1j
    t = np.random.random(num_sample*num_params)+np.random.random(num_sample*num_params)*1j
    p = p.reshape(num_sample,num_params)
    t = t.reshape(num_sample,num_params)
    run = test_multiprocess(circ=qc,samples=num_sample,p_in=p,t_in=t)
    num_params = run._num_params
    
    #print("p,t",p,t)
    iterable = [(i,p[i],t[i]) for i in range(num_sample)] 
    #print("i=0",iterable[0])
    #print(run.__dict__.keys())
    p_out,t_out = run.job(run.func,iterable)
    for i in range(num_sample):
        assert p[0].all() == p_out[0][:num_params].all()
        assert t[0].all() == t_out[0][:num_params].all()
    
    
    
if __name__=="__main__":
    test()


