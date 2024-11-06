# https://github.com/HGmin1159/binary_Dimension_Reduction/blob/5cae526b3aafe7bdc0295c2d3d3f839545b6bbc2/classes.py
import pandas as pd
import numpy as np
from typing import List, Tuple

from qiskit_optimization import QuadraticProgram
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from qiskit_optimization import QuadraticProgram
from qiskit import Aer
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer)

def f_obj(Q,beta) : 
    return -1*np.matmul(np.matmul(beta.T,Q),beta)
def f_nabla(Q,beta) :
    nabla_beta = -2*np.matmul(Q,beta)
    return nabla_beta.values
def l1_subgradient(beta) : return((beta>0)*1 - (beta<0)*1)


class OrdinaryEig(BaseEstimator, TransformerMixin):  
    def __init__(self, r=2):
        """
        Called when initializing the classifier
        """
        self.r = r

        # THIS IS WRONG! Parameters should have same name as attributes
    
    def fit(self, M):    
        n = M.shape[0];p=M.shape[1]
        num_iter= 5000
        lr = 0.005
        M_temp = M
        coef_frame = pd.DataFrame(np.zeros((p,self.r)))
        for j in range(self.r):
            theta_temp = np.random.normal(0,1,p)
            for i in range(num_iter) :
                theta_temp += -1*lr*f_nabla(M_temp,theta_temp)
                theta_temp = theta_temp/np.linalg.norm(theta_temp)
            theta_temp = theta_temp.reshape(-1,1)
            lamda = np.matmul(np.matmul(theta_temp.T,M),theta_temp)
            M_temp += -lamda[0][0]*np.matmul(theta_temp,theta_temp.T)
            coef_frame.iloc[:,j] = theta_temp
        self.coef_frame = coef_frame
        return self


    def transform(self, X):
        return self.coef_frame


class SparseEig(BaseEstimator, TransformerMixin):  
    def __init__(self, r=2, k = 0.5):
        """
        Called when initializing the classifier
        """
        self.r = r
        self.k = k
        # THIS IS WRONG! Parameters should have same name as attributes
    
    def fit(self, M):    
        n = M.shape[0];p=M.shape[1]
        num_iter= 5000
        lr = 0.005
        M_temp = M
        coef_frame = pd.DataFrame(np.zeros((p,self.r)))
        for j in range(self.r):
            theta_temp = np.random.normal(0,1,p)
            for i in range(num_iter) :
                theta_temp += -1*lr*(f_nabla(M_temp,theta_temp)+ (self.k)*l1_subgradient(theta_temp))
                theta_temp = theta_temp/np.linalg.norm(theta_temp)
            theta_temp = theta_temp.reshape(-1,1)
            lamda = np.matmul(np.matmul(theta_temp.T,M),theta_temp)
            M_temp += -lamda[0][0]*np.matmul(theta_temp,theta_temp.T)
            coef_frame.iloc[:,j] = theta_temp
        self.coef_frame = coef_frame
        return self


    def transform(self, X):
        return self.coef_frame


# class BinaryEig(BaseEstimator, TransformerMixin):  
#     def __init__(self, r=2, k=-1,reps=1,backend = Aer.get_backend("qasm_simulator")):
#         """
#         Called when initializing the classifier
#         """
#         self.r = r
#         if type(k) == int or type(k) == float : self.k = [k for i in range(self.r)]
#         else : self.k = k
#         self.reps = reps
#         self.backend = backend

#     def fit(self, M):
#         """kernel matix, axis, hyperparmeter"""
#         Q = -M
#         Q_temp = -M
#         p = M.shape[0]
#         coef_frame = pd.DataFrame(np.zeros((p,self.r)))

#         for j in range(self.r):
#             q = Q_temp.shape[0]
#             Q_temp = Q_temp-self.k[j]*np.identity(q)
#             beta = np.zeros((q,1))
#             result = qubo_qaoa(Q_temp,beta,self.reps,self.backend)
#             coef_temp = np.array([i for i in result[0]])
            
#             rest = 1-coef_frame.apply(sum,1)
#             coef_series = pd.DataFrame(np.zeros(p))
#             coef_series.loc[[bool(i) for i in rest]] = coef_temp.reshape(-1,1)
#             coef_frame.iloc[:,j] = coef_series
            
#             rest = 1-coef_frame.apply(sum,1)
#             Q_temp = Q.loc[[bool(i) for i in rest],[bool(i) for i in rest]]
#         self.coef_frame = coef_frame
#         return self

#     def transform(self, X):
#         return self.coef_frame
        
class BinaryEig(BaseEstimator, TransformerMixin):  
    def __init__(self, r=2, k=-1,reps=1,backend = Aer.get_backend("qasm_simulator")):
        """
        Called when initializing the classifier
        """
        self.r = r
        if type(k) == int or type(k) == float : self.k = [k for i in range(self.r)]
        else : self.k = k
        self.reps = reps
        self.backend = backend
        self.coef_frame = np.array([])

        # THIS IS WRONG! Parameters should have same name as attributes
    
    def fit(self, M, y=None):
        """kernel matix, axis, hyperparmeter"""
        #initialization
        Q = pd.DataFrame(-M)
        Q_temp = pd.DataFrame(-M)
        p = M.shape[0]
        coef_frame = pd.DataFrame(np.zeros((p,self.r)))

        # axis-wise hyperparameter
        if type(self.k) == int or type(self.k) == float : k_list = [self.k for i in range(self.r)] # 1 comes in ->1, 1,1,1, 1.... 
        else : k_list = self.k

        for j in range(self.r):
            q = Q_temp.shape[0] #redundant. p = M.shape[0] = q

            #fill the diagonal with the hyperparameter k.
            for i in range(q):
                Q_temp.iloc[i,i] += k_list[j] #Q_temp is symmetric.
            
            # solve a linear equation
            beta = np.zeros((q,1))
            result = qubo_qaoa(Q_temp,beta,self.reps,self.backend)
            coef_temp = np.array([i for i in result[0]])
            
            #update coef_frame
            rest = 1-coef_frame.apply(sum,1) # only update non-zero slots, for exclusiveness
            coef_series = pd.DataFrame(np.zeros(p))
            coef_series.loc[[bool(i) for i in rest]] = coef_temp.reshape(-1,1) # binary coeffs
            coef_frame.iloc[:,j] = coef_series #save the binary coeffs
            
            rest = 1-coef_frame.apply(sum,1) # after update, how many slots(dimensions) left?
            Q_temp = Q.loc[[bool(i) for i in rest],[bool(i) for i in rest]] # if rest = (0,0,0..,0), the iteration stops 
            try:
                if (i+1 < self.r) & (np.sum(rest) ==0):
                    raise Exception(f'{j+1}th PCA coefficients are all 1, iteration stops')    # 예외를 발생시킴
            except Exception as e:                             # 예외가 발생했을 때 실행됨
                    print('An error occurred.', e)
                    return self
        self.coef_frame = coef_frame.applymap(int)
        return self

    def transform(self, X, y=None):
        return self.coef_frame
        
class BinaryEigSA(BaseEstimator, TransformerMixin):

    def __init__(self, r=2, k=-1,alpha = 0.83,tau=1,k_flip=2,schedule_list = [10,10,20,20]):
        """
        Called when initializing the classifier
        """
        self.r = r
        if type(k) == int or type(k) == float : self.k = [k for i in range(self.r)]
        else : self.k = k
        self.alpha = alpha
        self.tau = tau
        self.k_flip = k_flip
        self.schedule_list = schedule_list

    # self, x, schedule_list, k_flip, alpha,tau,objective,y)
    def fit(self, M):
        """kernel matix, axis, hyperparmeter"""
        M = pd.DataFrame(M)
        M_temp = pd.DataFrame(M) 
        p = M.shape[0]

        schedule = self.schedule_list
        initial_t = self.tau
        coef_frame = pd.DataFrame(np.zeros(shape=[p,self.r]))
        for r in range(self.r):
            q = M_temp.shape[0]
            M_temp = M_temp-self.k[r]*np.identity(q)
            theta_zero = np.random.randint(2, size=q)
            for j in schedule:
                tau = initial_t / (1 + self.alpha * j)
                for m in range(j):
                    theta_star = flip(self.k_flip, np.where(theta_zero)[0], q)
                    if np.random.rand(1) <= min(1, np.exp((obj(M_temp, theta_zero) - obj(M_temp, theta_star)) / tau)):
                        theta_zero = theta_star
            theta_zero = theta_zero.reshape(-1, 1)
            
            coef_temp = np.array([i for i in theta_zero])
            
            rest = 1 - coef_frame.apply(sum, 1)
            coef_series = pd.DataFrame(np.zeros(p))
            coef_series.loc[[bool(i) for i in rest]] = coef_temp.reshape(-1, 1)
            # coef_frame.iloc[:,r] = coef_series
            coef_frame[r] = coef_series
            rest = 1 - coef_frame.apply(sum, 1)  # 왜두번
            M_temp = M.loc[[bool(i) for i in rest], [bool(i) for i in rest]]

            # result = theta_zero
        self.coef_frame = coef_frame.applymap(int)
        return self

    def transform(self, X):
        return self.coef_frame

    
## Solver
def qubo_qaoa(Q,beta,reps=1,backend = Aer.get_backend("qasm_simulator")):
    algorithm_globals.massive = True
    p = Q.shape[0]
    mod = QuadraticProgram("my problem")
    linear = {"x"+str(i): beta[i] for i in range(p)}
    quadratic = {("x"+str(i),"x"+str(j)): Q.values[i,j] for i in range(p) for j in range(p)}

    for i in range(p) :
        mod.binary_var(name="x"+str(i))

    mod.minimize(linear=linear,quadratic=quadratic)
    quantum_instance = QuantumInstance(backend)
    mes = QAOA(quantum_instance=quantum_instance,reps=reps)
    optimizer = MinimumEigenOptimizer(mes)
    result = optimizer.solve(mod)
    return([result,mod])

def qubo_exact(Q,beta):
    algorithm_globals.massive = True
    p = Q.shape[0]
    mod = QuadraticProgram("my problem")
    linear = {"x"+str(i): beta[i] for i in range(p)}
    quadratic = {("x"+str(i),"x"+str(j)): Q.values[i,j] for i in range(p) for j in range(p)}

    for i in range(p) :
        mod.binary_var(name="x"+str(i))

    mod.minimize(linear=linear,quadratic=quadratic)
    mes = NumPyMinimumEigensolver()
    optimizer = MinimumEigenOptimizer(mes)
    result = optimizer.solve(mod)
    return([result,mod])


# Sample_Generator

def generate_independent_sample(n_samples=500, n_features=10, beta_coef =[4,3,2,2],epsilon=4, random_state=None):

        rng = check_random_state(random_state)
        if n_features < 4:
            raise ValueError("`n_features` must be >= 4. "
                             "Got n_features={0}".format(n_features))
        # normally distributed features
        X = rng.randn(n_samples, n_features)
        # beta is a linear combination of informative features
        n_informative = len(beta_coef)
        beta = np.hstack((
            beta_coef, np.zeros(n_features - n_informative)))
        # cubic in subspace
        y = np.dot(X, beta)
        y += epsilon * rng.randn(n_samples)
        return X, y

def generate_dependent_sample(n_samples=500, n_features=10, beta_coef =[4,3,2,2],epsilon=4,covariance_parameter=1, random_state=None):

        rng = check_random_state(random_state)
        if n_features < 4:
            raise ValueError("`n_features` must be >= 4. "
                             "Got n_features={0}".format(n_features))

        # normally distributed features

        v = rng.normal(0, 0.4, (n_features, n_features))
        mean = np.zeros(n_features)
        cov = v @ v.T*covariance_parameter + 0.1 * np.identity(n_features)
        X = rng.multivariate_normal(mean, cov, n_samples)

        # beta is a linear combination of informative features
        n_informative = len(beta_coef)
        beta = np.hstack((
            beta_coef, np.zeros(n_features - n_informative)))

        # cubic in subspace
        y = np.dot(X, beta)
        y += epsilon * rng.randn(n_samples)

        return X, y


def read_otu(task_dir, otu_dir, positive_value):
    task = pd.read_csv(task_dir, '\t')
    otutable = pd.read_csv(otu_dir, '\t')
    samples = list(task['#SampleID'])
    data = dict()
    for sample in samples:
        otu = list(otutable[sample])
        otu.append(task.set_index('#SampleID').transpose()[sample][0])
        data[sample]=otu
    df = pd.DataFrame(data).transpose()
    df.columns = df.columns.map(lambda x: 'OTU_'+str(x+1))
    y = df["OTU_" + str(df.shape[1])].map(lambda x : 1 if x==positive_value else 0)
    X = df.drop("OTU_" + str(df.shape[1]), axis=1)
    return(X, y)

def flip(k, x, p):
    '''
        기존 선택된 변수들에서 k개만큼 flip해주는 함수
        input: flip할 횟수 k, 정수 index array, 총 변수 개수 p
        output: 새롭게 선택된 변수 결과
    '''
    zeros = np.zeros(p, dtype=int)
    idx = np.random.choice(p, size = k, replace = False)
    zeros[idx] = 1

    old = get_bin(x, p)
    new = abs(old - zeros)

    return new


def get_bin(x, p):
    '''
        선택된 변수의 정수 index를 [01100110..] 방식으로 변환해주는 함수
        input: 정수 index array, 총 변수 개수 p
        output: binary 방식 변수 선택 결과
    '''
    zeros = np.zeros(p, dtype=int)
    zeros[x] = 1

    return zeros


def obj(Q, beta):
    return -1 * np.matmul(np.matmul(beta.T, Q), beta)


