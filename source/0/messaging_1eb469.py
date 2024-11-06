# https://github.com/GEEGABYTE1/Quantmail/blob/a4a82d135d698fae0776bb74cfebe12c25d74c42/messaging.py
from sre_parse import State
import certifi
from datetime import datetime
from sympy import Quaternion
from termcolor import colored
from pymongo import MongoClient
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit import Aer
from accounts import qcs
import time




cluster = MongoClient("mongodb+srv://GEEGABYTE1:12345@socialmedia.few6z.mongodb.net/myFirstDatabase?retryWrites=true&w=majority", tls=True, tlsAllowInvalidCertificates=True)
db = cluster['socialmedia']['messaging']
qc_db = cluster['socialmedia']['qc']
account_db = cluster['socialmedia']['accounts']
email_db = cluster['socialmedia']['email']

chats = {}
restart = False



class Messages:
    root_user = None

    def sending_message(self, user, personal_qc, circuits):
        self.root_user = user
        while True:
            #names = []
            all_messages = db.find({})
            for message in all_messages:
                name = message['Id']
                name = name.strip('Name: ')
                circuit_names = list(circuits.keys())
                if name not in circuit_names:
                    circuit_names[name] = QuantumCircuit(2, 2)
                    
            
            date = datetime.now().strftime("%x")
            all_messages = db.find({})
            all_qcs = qc_db.find({})
            for message in all_messages:    
                #try:
                    if date != message['Date']:
                        print(colored('Today: {}'.format(message['Time']), 'red'))
                    else:
                        print(colored("{} ~ {}".format(message['Date'], message['Time']), "red"))
                    print(colored("From: ", 'green'), message['Id'])
                    name_split = message['Id'].strip('Name: ')
                                   
                    # Quantum Decryption
                    message_dict = self.untangle(personal_qc)
                    decrypted_message = list(message_dict.keys())[0]

                    print(colored("Message: {}".format(decrypted_message), 'green'))
                    print('-'*25)
                #except:
                    #pass
            
            person = "Name: {}".format(user)
            message = input("Message: ")
            message = message.strip(' ')
            if message == '/quit':
                break
            elif len(message) > 2:
                print('Message is too long')
            else:
                time = datetime.now().strftime("%X")
                new_message = self.decrypt(personal_qc, message)
                msg = {"Id": person, "Message": new_message, "Date": date, "Time":time}
                
                db.insert_one(msg)
                print(colored('Message sent successfully', 'green'))
                print('-'*25)

    def fetch_users(self):
        names_in_order = []
        count = 0
        for param in account_db:
            name = param['Id']
            names_in_order.append({name: count})
            count += 1

        return names_in_order

    
    def receive_email(self, personal_qc, root_user):
        print('Checking your Inbox...')
        time.sleep(0.2)
        all_emails = email_db.find({})
        for email_dict in all_emails:
            email_receiver = email_dict['Receiver']
            if email_dict['Sender'] == self.root_user:
                email_message = email_dict['Message']
                message_dict = self.untangle(personal_qc)
                decrypted_message = list(message_dict.keys())[0]
                print('-'*24)
                print(colored('Name: {}'.format(email_dict['Sender']), 'white'))
                print(colored('Message: {}'.format(decrypted_message), 'white'))
                print(colored('Time: {}'.format(email_dict['Date']), 'white'))
                time.sleep(0.1)


    def send_email(self, personal_qc):
        desired_user = str(input("Please type in the Desired User: "))
        desired_user = desired_user.strip(' ')
        all_accounts = list(account_db.find({}))
        user_found = False
        for account_dict in range(len(all_accounts)):
            name = all_accounts[account_dict]['User']
            inbox = all_accounts[account_dict]['Inbox']
            if desired_user == name:
                user_found = True
                print('You may now type your message: ')
                time.sleep(0.3)
                while True:
                    user_message = str(input(colored(': ')))
                    if user_message == '/quit':
                        print('You have left the Emailing process')
                        break 
                    elif len(user_message) > 2:
                        print(colored('Message is too long', 'red'))
                    else:
                        break
                #try:
                user_message = user_message.strip(' ')
                new_user_message = self.decrypt(personal_qc, user_message)
                email = {'Sender': self.root_user, 'Receiver': desired_user, 'Message':new_user_message, 'Date':datetime.now().strftime("%x")} 
                email_db.insert_one(email)
                print(all_accounts[account_dict]['Inbox'])
                print(colored('Email Sent Successfully!' , 'green'))
                #except:
                    #return
            else:
                continue
        if user_found == False:
            print(colored('User not found', 'red'))
        else:
            pass
    
    def decrypt(self, personal_qc, message):
        if message[-2] == '1':
            personal_qc.x(1)
            #message[-2] = '0'
        if message[-1] == '1':
            personal_qc.z(1)
        
        print(colored('Decryption in Process', 'white'))
        ket = Statevector(personal_qc)
        ket.draw()
        return message



    def untangle(self, personal_qc):
        resulting_qc = QuantumCircuit(2, 2)
        resulting_qc.cx(0, 1)
        resulting_qc.h(0)
        print(colored('Entanglement in Process', 'white'))
        resulting_qc.measure([0, 1], [0, 1])
        backend = Aer.get_backend('aer_simulator')
        resulting_dictionary = backend.run(personal_qc.compose(resulting_qc)).result().get_counts()
        return resulting_dictionary
        