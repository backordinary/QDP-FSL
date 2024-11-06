# https://github.com/zillerium/shoro/blob/d1d4d7e0959921b800aa48107503cddbf91348b2/game.py
import qiskit as qk
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit, execute


from  qiskit  import  register

#import Qconfig and set APIToken and API url
import sys, getpass
try:
    #sys.path.append("../../") # go to parent dir
    import Qconfig
    qx_config = {
        "APItoken": Qconfig.APItoken,
        "url": Qconfig.config['url']}
    print('Qconfig loaded from %s.' % Qconfig.__file__)
except:
    APItoken = getpass.getpass('Please input your token and hit enter: ')
    qx_config = {
        "APItoken": APItoken,
        "url":"https://quantumexperience.ng.bluemix.net/api"}
    print('Qconfig.py not found in qiskit-tutorial directory; Qconfig loaded using user input.')

register(qx_config['APItoken'], qx_config['url'])
