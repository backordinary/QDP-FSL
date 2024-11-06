# https://github.com/heineaabo/QuantumAlgorithms/blob/9ef038b5b2b721efada8360e76382b055a1fd38b/quantum_algorithms/attributes/update_all.py
import pickle
from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
devices = len(provider.backends())
# Prettyprint prepare
str_len = 0
for dev in provider.backends():
    if len(dev.name()) > str_len:
        str_len = len(dev.name())
for i,device in enumerate(provider.backends()):
    name = device.name()
    if name == 'ibmq_qasm_simulator':
        print('{} / {}'.format(i+1,devices),name,'\u0020'*(str_len-len(name)),'Skipped!')
        continue
    # Get coupling map
    coupling_map = device.configuration().coupling_map
    f = open('coupling_maps/'+name+'.pkl','wb')
    pickle.dump(coupling_map,f)
    f.close()
    # Get noise model
    noise_model = NoiseModel.from_backend(device)
    f = open('noise_models/'+name+'.pkl','wb')
    pickle.dump(noise_model.to_dict(),f)
    f.close()
    # Get Basis gates
    basis_gates = noise_model.basis_gates
    f = open('basis_gates/'+name+'.pkl','wb')
    pickle.dump(basis_gates,f)
    f.close()
    print('{} / {}'.format(i+1,devices),name,'\u0020'*(str_len-len(name)),'Updated!')

