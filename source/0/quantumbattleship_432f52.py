# https://github.com/Nancuharsha/QuantumBattleShip/blob/313c164d623d3213b3581c7b2258aa4b202d051e/QuantumBattleShip.py
#Libraries included here are qiskit
from qiskit import BasicAer, IBMQ
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import execute
from qiskit.providers.ibmq import least_busy
#importing python libraries for the further functionalities needed in the game 
import getpass, random, numpy, math
#in this game there are 2 players
players = [0,1]
#there are 3 battleships for each player 
ships = [0,1,2]
#there are 5 positions for placing the ships
positions = [0,1,2,3,4]
#each player can deploy their 3 ships at any of the 5 spots that their opponent is unaware of.

#function for the showing game title and play rules before the start of the game
def startDisplayScreen ():
    print()
    print()
    print("Took reference from a code")
    print("-----------------------------------")
    print("Battle ship game on Quantum")
    print("-----------------------------------")
    print()
    print("**********Game rules:************")
    print("No.of players:2")
    print()
    print("No.of ships you owe:3")
    print()
    print("No.of spots for ship to place:5")
    print()
    print("The first ship requires 1 bomb for destroying.")
    print()
    print("The second ship requires 2 bombs to destroy.")
    print()
    print("The third ship requires 3 bombs to destroy.")
    print()
    print("So why late. Let's start the game")
    print()
    print("Fasten your seat belts and be on your positions for the attack")
    print()

#function for connecting to the IBM quantum computer using API token    
def choosingQuantumDeviceForComputation ():
    #taking the user input for checking if the gamer wants to proceed with the game.
    choice = input("Are you excited to play a game on a real quantum device? (y/n)\n The game may occasionally produce mistakes owing to a shortage of 5 qubits.)").upper()
    #if the choice is yes then asking for the quantum device for the action
    if (choice=="Y"):
      #way to link or connect with the help of unique token id
        youApiToken = '6a8a975339d5506a8acf8e7baa06c5e009f2a7bc6e24ee49b71ec592ab29b83df115a01ef8747c9ae04ad20fa5c415afceef167b25fbf2f9a43c4f2f5390477e'
        IBMQ.save_account(youApiToken,overwrite=True)
        #IBMQ's load account function
        temp = IBMQ.load_account()
        #IBMQ's backends function
        ibmq_backends = temp.backends()
        #choosing the least busy device
        choosenDevice = least_busy(temp.backends(simulator=False))
        #output the backend quantum device
        print('quantum_backend: ', device)
    else:
      #if the choice is no then calling the method get backend for the quantum simulator
        choosenDevice = BasicAer.get_backend('qasm_simulator') 
    #returning device for the ask for device function    
    return choosenDevice

#function for placing the ships in the required positions.  
def choosingBattleShipLocations ():
    #ship positions
    battleShipsLocations = [ [-1]*3 for _ in range(2)] 
    #looping over all the players
    for player in players:
      #looping over all the ships
        for ship in ships:
            choosing = True
            #It runs until a correct location is choosen for placing the ships
            while (choosing):
                print()
                #printing the players choosen position of the ship
                print("@ Player " + str(player+1) + ", choose a spot for the ship " + str(ship+1) )
                print()
                #taking position input from the user
                position = input()
                flag = True
                #validating whether the position is digit
                if position.isdigit(): 
                  #converting the position variable to int type
                    position = int(position)
                    #Validating the choosen position with the already placed ship spots 
                    if (position in positions) and (not position in battleShipsLocations[player]): 
                      #if its a new position then assigning the spot to the new ship
                        battleShipsLocations[player][ship] = position
                        choosing = False
                        flag = False
                #checking if it is not a valid input
                if flag == True:
                    print()
                    print("Please try again it's an invalid input")
                    print()
    #returning the ship positions ie. array of ShipPos                
    return battleShipsLocations
                        
#function for aiming bomb positions
def choosingBombingLocations ( bombingTorpedo ):
    
    #looping over all the players in the game
    for player in players:
        #Letting know the user regarding the turns for better understanding
        print("\n\nNow it's Player " + str(player+1) + "'s turn.\n")
        #Variable used for bombing purpose
        choosing = True
        #It runs until a correct location is choosen for bombing
        while (choosing):
            #taking input of bomb position from the user 
            position = input("Choose a position to aim the bomb onto any spot\n")
            flag = True
            if position.isdigit(): 
                position = int(position)
                if position in positions: 
                    bombingTorpedo[player][position]+= 1
                    choosing = False
                    flag = False
            #checking if it is not a valid input        
            if flag == True:
                print("")
                print("Please check it's an invalid input.")
                print("\n")
                
    return bombingTorpedo

#function to show the overview of the battle area with the destroy percentage of the ships after bombing.
def statusOftheGame ( overviewOfGame, battleShipsLocations, numberOfHits ):
    #boolean varable for game play start
    statusOfgame = True
    #damage variable stores the percentage of the damage happed to the ship
    damageCalculations = [ [0]*5 for _ in range(2)] 
    #looping over all the players    
    for player in players:
        #looping over all the player's grid positions
        for bitString in overviewOfGame[player].keys():
            #looping over all the positions 
            for position in positions:
                #suming up the damage caused for the torpedo shots and checking with the thresholds
                if (bitString[4-position]=="1" and damageCalculations[player][position] + overviewOfGame[player][bitString]/numberOfHits <= 1.0):
                    damageCalculations[player][position] += overviewOfGame[player][bitString]/numberOfHits 
    print(damageCalculations)
    #looping over all the players in the game   
    for player in players:
        #Taking input from the user for viewing the ships condition
        input("\nPress anything to view the results for Player " + str(player+1) + "'s ships\n")
        #Display grid
        displayDamagePercentage = [" ?  "]*5
        #looping over the ship positions of a particular player
        for position in battleShipsLocations[player]:
            #Checking for the damage percentage and displaying the value if it is less than threshold 
            #for getting better understanding for the user to check the damage caused to the ship
            if (damageCalculations[player][position] > 0.1 and damageCalculations[player][position] <= 0.9):
                displayDamagePercentage[position] = str(int( 100*damageCalculations[player][position] )) + "% "
            #checking if the damage percentage is greater than the threshold then considering it to be a fully damaged ship      
            elif (damageCalculations[player][position]>0.9):
                displayDamagePercentage[position] = "100%"
        print(displayDamagePercentage)
        #printing the display grid for the users understanding of the current situations
        print("Present scenario of the ships \n")
        print(displayDamagePercentage[ 4 ] + "    " + displayDamagePercentage[ 0 ])
        #grid overview
        print(" |\     /|")
        print(" | \   / |")
        print(" |  \ /  |")
        print(" |  " + displayDamagePercentage[ 2 ] + " |")
        print(" |  / \  |")
        print(" | /   \ |")
        print(" |/     \|")
        print(displayDamagePercentage[ 3 ] + "    " + displayDamagePercentage[ 1 ])
        print()
        print()

        print()
        #Checking if all the ships got damaged and then announcing the winner of the game
        if (damageCalculations[player][ battleShipsLocations[player][0] ]>.9) and (damageCalculations[player][ battleShipsLocations[player][1] ]>.9) and (damageCalculations[player][ battleShipsLocations[player][2] ]>.9):
            if ((player+2)%2 ) !=0:
                print( "Player:  " + str((player+2)%2) + "'  is the winner\n\n")
            else:
                print ("Game is Draw between players\n\n")
            #pausing the gaming after the winner's announcement
            statusOfgame = False
        #The game is over here
        if (statusOfgame is False):
            print("")
            print("Done with the Game")
            print("")

    return statusOfgame
def main():

  startDisplayScreen() #function call for the showing game title and play rules before the start of the game

  QuantumDevice = choosingQuantumDeviceForComputation() #function call for connecting to the IBM quantum computer using API token 

  battleShipsLocations = choosingBattleShipLocations() #function call for placing the ships in the required positions.
  print("Player 1 choosen battleship locations: ")
  print(battleShipsLocations[0])
  print("Player 2 choosen battleship locations: ")
  print(battleShipsLocations[1])
  statusOfgame = True #starting the game

  bombingTorpedo = [ [0]*5 for _ in range(2)] #bomb positions

  numberOfHits = 1024 #Hits

  overviewOfGame = [{},{}] #overview of the battle space area 

  while (statusOfgame): 
    #Variable for the storing the bombing
    bombingTorpedo = choosingBombingLocations( bombingTorpedo )
    print(bombingTorpedo)
    #quantum circuit Qubit
    quantumCircuit = []
    #This game is built over 5 qiskit register
    #register for storing quantum bits for quantum computing ie. it stores rotational values of bomb aiming
    quantumBits = QuantumRegister(5) 
    #register for storing classical bits for classical computing ie. it stores the computational value of destroy percentage of the ship
    classicalBits = ClassicalRegister(5)
    #appending the registers in quantum circuit
    quantumCircuit.append( QuantumCircuit(quantumBits, classicalBits) )
    #looping over all the positions
    for position in positions:
          #looping over the bombing
        for n in range( bombingTorpedo[(0+1)%2][position] ):
            #This is where we decrease the value of the ship stability using rotation function using “U3” method over one axis
            for ship in ships:
                  #if the bombing position of player macthes the opponent ships spot then it gets destroyed
                if ( position == battleShipsLocations[0][ship] ):
                    frac = 1/(ship+1)
                    quantumCircuit[0].u3(frac * math.pi, 0.0, 0.0, quantumBits[position])
        #measuring the qbits and appending results in the classical registers                               
    for position in positions:
        quantumCircuit[0].measure(quantumBits[position], classicalBits[position])
    #This game is built over 5 qiskit register
    #register for storing quantum bits for quantum computing ie. it stores rotational values of bomb aiming
    quantumBits = QuantumRegister(5) 
    #register for storing classical bits for classical computing ie. it stores the computational value of destroy percentage of the ship
    classicalBits = ClassicalRegister(5)
    #appending the registers in quantum circuit
    quantumCircuit.append( QuantumCircuit(quantumBits, classicalBits) )
    #looping over all the positions
    for position in positions:
          #looping over the bombing
        for n in range( bombingTorpedo[(1+1)%2][position] ):
            #This is where we decrease the value of the ship stability using rotation function using “U3” method over one axis
            for ship in ships:
                  #if the bombing position of player macthes the opponent ships spot then it gets destroyed
                if ( position == battleShipsLocations[1][ship] ):
                    frac = 1/(ship+1)
                    quantumCircuit[1].u3(frac * math.pi, 0.0, 0.0, quantumBits[position])
        #measuring the qbits and appending results in the classical registers                               
    for position in positions:
        quantumCircuit[1].measure(quantumBits[position], classicalBits[position])
    #input values are executing over here
    job = execute(quantumCircuit, backend=QuantumDevice, shots=numberOfHits)
    print("We submitted the game to Quantum backend systems")
    for player in players:
        overviewOfGame[player] = job.result().get_counts(quantumCircuit[player])
    #calling display grid function for the overview of the output results
    statusOfgame = statusOftheGame ( overviewOfGame, battleShipsLocations, numberOfHits )

#main function to start with 
if __name__ == "__main__":
    main()