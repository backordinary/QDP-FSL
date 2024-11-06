# https://github.com/IndigoChild88/Data-Science-Machine-Learning-Projects/blob/291b2a79d89f68ba826b941e29c0e7c1c753fdda/Finance/Quantum/Qasm_Demo.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from qiskit.ml.datasets import wine
from qiskit import BasicAer
from qiskit.aqua.input import ClassificationInput
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.multiclass_extensions import AllPairs
from qiskit.aqua.utils.dataset_helper import get_feature_dimension
from qiskit.aqua.algorithms.many_sample.qsvm._qsvm_estimator import _QSVM_Estimator


# In[154]:


"""
from qiskit import IBMQ
token = '52e09c75d86e3953ebcf3a693465853ee1bfafd0795a68b2e99ac2e8103acf74d6034caa3df78b95bc9ca867cd3721f932d48aa2197f0ba82e9ef8d11f62edf1'
#my_provider = IBMQ.get_provider()
#backend = my_provider.get_backend('ibmq_qasm_simulator')
"""


# In[155]:


"""
provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_qasm_simulator')
"""


# In[2]:


data_1_Bid = pd.read_csv("USDJPY_Long_Bid.csv")
data_1_Ask = pd.read_csv("USDJPY_Long_Ask.csv")
data = pd.DataFrame({"AskClose":data_1_Ask['Close'], 
                         "BidClose":data_1_Bid['Close']})


# In[3]:


#Used to prep data for classifiers, returns dataframe then array that contains lags
def Prep(data, lag_num):
    data = data.dropna()
    df = pd.DataFrame(data[['AskClose', 'BidClose']].mean(axis=1), columns=['midclose'])
    #print(data[:5]/data[:5].shift(1))
    df['returns'] = np.log(df / df.shift(1))
    lags = lag_num
    cols = []
    for lag in range(1, lags + 1):
        col = 'lag_%s' % lag
        df[col] = df['returns'].shift(lag)
        cols.append(col)
    
    df = df.dropna()
    return df, cols;


# In[4]:


df_1, cols = Prep(data,4)
df_1 = np.sign(df_1)
df_1 = df_1.replace([-1], 2)
print(df_1[cols])
np.isin(2, df_1[cols])


# In[56]:


train_x, test_x, train_y, test_y = train_test_split(df_1[cols][80000:].to_numpy(), df_1['returns'][80000:].to_numpy(), test_size=0.2)


"""
train_x = np.sign(train_x)
test_x = np.sign(test_x)
train_y = np.sign(train_y)
test_y = np.sign(test_y)"""


num_features = 4
training_size = 1660
test_size = 66640
feature_map = SecondOrderExpansion(feature_dimension=num_features, depth=1)


# In[57]:


params = {
            'problem': {'name': 'classification'},
            'algorithm': {
                'name': 'QSVM',
            },
            'multiclass_extension': {'name': 'OneAgainstRest'},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 1 }
}
 

training_dataset={'Sell':train_x[train_y==2],
                'Nothing':train_x[train_y==0],
                'Buy':train_x[train_y==1]}
test_dataset={'Sell':test_x[test_y==2],
                        'Nothing':test_x[test_y==0],
                        'Buy':test_x[test_y==1]}
total_arr = np.concatenate((test_dataset['Sell'],test_dataset['Nothing'],test_dataset['Buy']))
alg_input = ClassificationInput(training_dataset, test_dataset, total_arr)


# In[58]:


aqua_globals.random_seed = 1024

backend = BasicAer.get_backend('qasm_simulator')
feature_map = SecondOrderExpansion(feature_dimension=get_feature_dimension(training_dataset),
                                   depth=1, entangler_map=[[0, 1]])
svm = QSVM(feature_map, training_dataset,test_dataset, total_arr,
          multiclass_extension=AllPairs(_QSVM_Estimator, [feature_map],
                                       ))
quantum_instance = QuantumInstance(backend, shots=1000,
                                    #seed_simulator=aqua_globals.random_seed,
                                    #seed_transpiler=aqua_globals.random_seed,
                                  skip_qobj_validation=False,
                                  wait = None)

#"""
result = svm.run(quantum_instance)
for k,v in result.items():
    print("'{}' : {}".format(k, v))
#"""


# In[59]:


#svm.train(train_x, train_y, quantum_instance)


# In[60]:


y_pred = svm.predict(test_x)
print(accuracy_score(test_y, y_pred ) * 100, '%')


# In[61]:


import pickle
pickle.dump(svm, open('Quantum_model', 'wb'))


# In[63]:


#loaded_model = pickle.load(open('Quantum_model', 'rb'))
#loaded_model.predict(test_x)


# In[47]:


#pred = svm.predict(test_x[len(test_x)-5:])


# In[48]:


#print(test_y[len(test_y)-5:])
#print(pred)


# In[44]:


#test_x == test_y


# In[18]:


#import os
#os.path.abspath(os.getcwd())


# In[49]:


#svm.save_model("C:\\Users\\Albert\\Documents\\NSBE\\Quantum_Model_85%")


# In[13]:


#np.isin(2, train_x)


# In[150]:


#test_y


# In[ ]:




