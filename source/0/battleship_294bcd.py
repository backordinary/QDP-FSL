# https://github.com/Apress/practical-quantum-computing/blob/227861a105f641e733fc568468e417aba7f098c1/Ch06/battleship/BattleShip.py
# Checking the version of PYTHON; we only support > 3.5
import sys
import qiskit

# path to QConfig
sys.path.append('../config/')

if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')
    
from qiskit import QuantumProgram
import Qconfig

import getpass, random, numpy, math

print("############### Quantum BattleSip ##################")
print("                 ___         ___                    _       _         ")
print("                | _ ) _  _  |   \  ___  __  ___  __| | ___ | |__ _  _ ")
print("                | _ \| || | | |) |/ -_)/ _|/ _ \/ _` |/ _ \| / /| || |")
print("                |___/ \_, | |___/ \___|\__|\___/\__,_|\___/|_\_\ \_,_|")
print("                      |__/                                            ")
print("                           James Wootton, University of Basel")
print("")
print("                        A game played on a real quantum computer!")
print("")
print("         Learn how to make your own game for a quantum computer at decodoku.com")
print("")
print("")
randPlace = input("> Press Enter to start placing ships...\n").upper()

d = input("Do you want to play on the real device? (y/n)\n").upper()
if (d=="Y"):
    device = 'ibmqx2'
else:
    device = 'ibmqx_qasm_simulator'
	
# note that device should be 'ibmqx_qasm_simulator', 'ibmqx2' or 'local_qasm_simulator'
# while we are at it, let's set the number of shots
shots = 1024

randPlace = input("> Device (" + device + ") Press Enter to start placing ships...\n").upper()

#  get the players to set up their boards.
# The variable ship[X][Y] will hold the position of the Yth ship of player X+1
shipPos = [ [-1]*3 for _ in range(2)] # all values are initialized to the impossible position -1|

# loop over both players and all three ships for each
for player in [0,1]:
    
    # if we chose to bypass player choice and do random, we do that
    if ((randPlace=="r")|(randPlace=="R")):
        randPos = random.sample(range(5), 3)
        for ship in [0,1,2]:
            shipPos[player][ship] = randPos[ship]
        #print(randPos) #uncomment if you want a sneaky peak at where the ships are
    else:
        for ship in [0,1,2]:

            # ask for a position for each ship, and keep asking until a valid answer is given
            choosing = True
            while (choosing):

                # get player input
                position = getpass.getpass("Player " + str(player+1) + ", choose a position for ship " + str(ship+1) + " (0, 1, 2, 3 or 4)\n" )

                # see if the valid input and ask for another if not
                if position.isdigit(): # valid answers  have to be integers
                    position = int(position)
                    if (position in [0,1,2,3,4]) and (not position in shipPos[player]): # they need to be between 0 and 5, and not used for another ship of the same player
                        shipPos[player][ship] = position
                        choosing = False
                        print ("\n")
                    elif position in shipPos[player]:
                        print("\nYou already have a ship there. Try again.\n")
                    else:
                        print("\nThat's not a valid position. Try again.\n")
                else:
                    print("\nThat's not a valid position. Try again.\n")

# main loop. For this game, each interation starts by asking players where on the opposing grid they want a bomb
# The quantum computer then calculates the effects of the bombing, and the results are presented to the players
# game continues until all the ships of one player are destroyed.
# the game variable will be set to False once the game is over
game = True

# the variable bombs[X][Y] will hold the number of times position Y has been bombed by player X+1
bomb = [ [0]*5 for _ in range(2)] # all values are initialized to zero

# the variable grid[player] will hold the results for the grid of each player
grid = [{},{}]

while (game):
    
    input("> Press Enter to place some bombs...\n")
    
    print("\n\nIt's now Player " + str(0+1) + "'s turn.\n")

    # keep asking until a valid answer is given
    choosing = True
    while (choosing):

        # get player input
        position = input("Choose a position to bomb (0, 1, 2, 3 or 4)\n")

        # see if this is a valid input. ask for another if not
        if position.isdigit(): # valid answers  have to be integers
            position = int(position)
            if position in range(5): # they need to be between 0 and 5, and not used for another ship of the same player
                bomb[0][position] = bomb[0][position] + 1
                choosing = False
                print ("\n")
            else:
                print("\nThat's not a valid position. Try again.\n")
        else:
            print("\nThat's not a valid position. Try again.\n")
    
    print("\n\nIt's now Player " + str(1+1) + "'s turn.\n")

    # keep asking until a valid answer is given
    choosing = True
    while (choosing):

        # get player input
        position = input("Choose a position to bomb (0, 1, 2, 3 or 4)\n")

        # see if this is a valid input. ask for another if not
        if position.isdigit(): # valid answers  have to be integers
            position = int(position)
            if position in range(5): # they need to be between 0 and 5, and not used for another ship of the same player
                bomb[1][position] = bomb[1][position] + 1
                choosing = False
                print ("\n")
            else:
                print("\nThat's not a valid position. Try again.\n")
        else:
            print("\nThat's not a valid position. Try again.\n")
    

    if device=='ibmqx2':
        print("\nWe'll now get the quantum computer to see what happens to Player " + str(0+1) + "'s ships.\n")
    else:
        print("\nWe'll now get the simulator (" + device + ") to see what happens to Player " + str(0+1) + "'s ships.\n")
    
    # now to set up the quantum program (QASM) to simulate the grid for this player
    
    Q_program = QuantumProgram()
    Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"]) # set the APIToken and API url
     
    #print ("Backends :" + str(qiskit.backends.discover_local_backends()))
	 
    # declare register of 5 qubits
    q = Q_program.create_quantum_register("q", 5)
    # declare register of 5 classical bits to hold measurement results
    c = Q_program.create_classical_register("c", 5)
    # create circuit
    gridScript = Q_program.create_circuit("gridScript", [q], [c])    
    # add as many bombs as have been placed at this position
    for n in range( bomb[(0+1)%2][0] ):
        # the effectiveness of the bomb
        # (which means the quantum operation we apply)
        # depends on which ship it is
        for ship in [0,1,2]:
            if ( 0 == shipPos[0][ship] ):
                frac = 1/(ship+1)
                # add this fraction of a NOT to the QASM
                gridScript.u3(frac * math.pi, 0.0, 0.0, q[0])
                                
    # add as many bombs as have been placed at this position
    for n in range( bomb[(0+1)%2][1] ):
        # the effectiveness of the bomb
        # (which means the quantum operation we apply)
        # depends on which ship it is
        for ship in [0,1,2]:
            if ( 1 == shipPos[0][ship] ):
                frac = 1/(ship+1)
                # add this fraction of a NOT to the QASM
                gridScript.u3(frac * math.pi, 0.0, 0.0, q[1])
                                
    # add as many bombs as have been placed at this position
    for n in range( bomb[(0+1)%2][2] ):
        # the effectiveness of the bomb
        # (which means the quantum operation we apply)
        # depends on which ship it is
        for ship in [0,1,2]:
            if ( 2 == shipPos[0][ship] ):
                frac = 1/(ship+1)
                # add this fraction of a NOT to the QASM
                gridScript.u3(frac * math.pi, 0.0, 0.0, q[2])
                                
    # add as many bombs as have been placed at this position
    for n in range( bomb[(0+1)%2][3] ):
        # the effectiveness of the bomb
        # (which means the quantum operation we apply)
        # depends on which ship it is
        for ship in [0,1,2]:
            if ( 3 == shipPos[0][ship] ):
                frac = 1/(ship+1)
                # add this fraction of a NOT to the QASM
                gridScript.u3(frac * math.pi, 0.0, 0.0, q[3])
                                
    # add as many bombs as have been placed at this position
    for n in range( bomb[(0+1)%2][4] ):
        # the effectiveness of the bomb
        # (which means the quantum operation we apply)
        # depends on which ship it is
        for ship in [0,1,2]:
            if ( 4 == shipPos[0][ship] ):
                frac = 1/(ship+1)
                # add this fraction of a NOT to the QASM
                gridScript.u3(frac * math.pi, 0.0, 0.0, q[4])
                                
    gridScript.measure(q[0], c[0])
    gridScript.measure(q[1], c[1])
    gridScript.measure(q[2], c[2])
    gridScript.measure(q[3], c[3])
    gridScript.measure(q[4], c[4])
    # to see what the quantum computer is asked to do, we can print the QASM file
    # this lines is typically commented out
    #print( Q_program.get_qasm("gridScript") )
    
    # compile and run the QASM
    results = Q_program.execute(["gridScript"], backend=device, shots=shots)

    # extract data
    grid[0] = results.get_counts("gridScript")
    

    if device=='ibmqx2':
        print("\nWe'll now get the quantum computer to see what happens to Player " + str(1+1) + "'s ships.\n")
    else:
        print("\nWe'll now get the simulator (" + device + ") to see what happens to Player " + str(1+1) + "'s ships.\n")
    
    # now to set up the quantum program (QASM) to simulate the grid for this player
    
    Q_program = QuantumProgram()
    Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"]) # set the APIToken and API url
     
    #print ("Backends :" + str(qiskit.backends.discover_local_backends()))
	 
    # declare register of 5 qubits
    q = Q_program.create_quantum_register("q", 5)
    # declare register of 5 classical bits to hold measurement results
    c = Q_program.create_classical_register("c", 5)
    # create circuit
    gridScript = Q_program.create_circuit("gridScript", [q], [c])    
    # add as many bombs as have been placed at this position
    for n in range( bomb[(1+1)%2][0] ):
        # the effectiveness of the bomb
        # (which means the quantum operation we apply)
        # depends on which ship it is
        for ship in [0,1,2]:
            if ( 0 == shipPos[1][ship] ):
                frac = 1/(ship+1)
                # add this fraction of a NOT to the QASM
                gridScript.u3(frac * math.pi, 0.0, 0.0, q[0])
                                
    # add as many bombs as have been placed at this position
    for n in range( bomb[(1+1)%2][1] ):
        # the effectiveness of the bomb
        # (which means the quantum operation we apply)
        # depends on which ship it is
        for ship in [0,1,2]:
            if ( 1 == shipPos[1][ship] ):
                frac = 1/(ship+1)
                # add this fraction of a NOT to the QASM
                gridScript.u3(frac * math.pi, 0.0, 0.0, q[1])
                                
    # add as many bombs as have been placed at this position
    for n in range( bomb[(1+1)%2][2] ):
        # the effectiveness of the bomb
        # (which means the quantum operation we apply)
        # depends on which ship it is
        for ship in [0,1,2]:
            if ( 2 == shipPos[1][ship] ):
                frac = 1/(ship+1)
                # add this fraction of a NOT to the QASM
                gridScript.u3(frac * math.pi, 0.0, 0.0, q[2])
                                
    # add as many bombs as have been placed at this position
    for n in range( bomb[(1+1)%2][3] ):
        # the effectiveness of the bomb
        # (which means the quantum operation we apply)
        # depends on which ship it is
        for ship in [0,1,2]:
            if ( 3 == shipPos[1][ship] ):
                frac = 1/(ship+1)
                # add this fraction of a NOT to the QASM
                gridScript.u3(frac * math.pi, 0.0, 0.0, q[3])
                                
    # add as many bombs as have been placed at this position
    for n in range( bomb[(1+1)%2][4] ):
        # the effectiveness of the bomb
        # (which means the quantum operation we apply)
        # depends on which ship it is
        for ship in [0,1,2]:
            if ( 4 == shipPos[1][ship] ):
                frac = 1/(ship+1)
                # add this fraction of a NOT to the QASM
                gridScript.u3(frac * math.pi, 0.0, 0.0, q[4])
                                
    gridScript.measure(q[0], c[0])
    gridScript.measure(q[1], c[1])
    gridScript.measure(q[2], c[2])
    gridScript.measure(q[3], c[3])
    gridScript.measure(q[4], c[4])
    # to see what the quantum computer is asked to do, we can print the QASM file
    # this lines is typically commented out
    #print( Q_program.get_qasm("gridScript") )
    
    # compile and run the QASM
    results = Q_program.execute(["gridScript"], backend=device, shots=shots)

    # extract data
    grid[1] = results.get_counts("gridScript")
    
    # we can check up on the data if we want
    # these lines are typically commented out
    #print( grid[0] )
    #print( grid[1] )
    
    # if one of the runs failed, tell the players and start the round again
    if ( ( 'Error' in grid[0].values() ) or ( 'Error' in grid[1].values() ) ):

        print("\nThe process timed out. Try this round again.\n")
        
    else:
        
        # look at the damage on all qubits (we'll even do ones with no ships)
        damage = [ [0]*5 for _ in range(2)] # this will hold the prob of a 1 for each qubit for each player
        for bitString in grid[0].keys():
            # and then over all positions
            for position in range(5):
                # if the string has a 1 at that position, we add a contribution to the damage
                # remember that the bit for position 0 is the rightmost one, and so at bitString[4]
                if (bitString[4-position]=="1"):
                    damage[0][position] += grid[0][bitString]/shots          
        for bitString in grid[1].keys():
            # and then over all positions
            for position in range(5):
                # if the string has a 1 at that position, we add a contribution to the damage
                # remember that the bit for position 0 is the rightmost one, and so at bitString[4]
                if (bitString[4-position]=="1"):
                    damage[1][position] += grid[1][bitString]/shots          
        
        # give results to players
        for player in [0,1]:

            input("\nPress Enter to see the results for Player " + str(player+1) + "'s ships...\n")

            # report damage for qubits that are ships, and which have significant damange
            # ideally this would be non-zero damage, but noise means that can happen for ships that haven't been hit
            # so we choose 5% as the threshold
            display = [" ?  "]*5
            # loop over all qubits that are ships
            for position in shipPos[player]:
                # if the damage is high enough, display the damage
                if ( damage[player][position] > 0.1 ):
                    if (damage[player][position]>0.9):
                         display[position] = "100%"
                    else:
                        display[position] = str(int( 100*damage[player][position] )) + "% "

            print("Here is the percentage damage for ships that have been bombed.\n")
            print(display[ 4 ] + "    " + display[ 0 ])
            print(" |\     /|")
            print(" | \   / |")
            print(" |  \ /  |")
            print(" |  " + display[ 2 ] + " |")
            print(" |  / \  |")
            print(" | /   \ |")
            print(" |/     \|")
            print(display[ 3 ] + "    " + display[ 1 ])
            print("\n")
            print("Ships with 95% damage or more have been destroyed\n")

            print("\n")

            # if a player has all their ships destroyed, the game is over
            # ideally this would mean 100% damage, but we go for 95% because of noise again
            if (damage[player][ shipPos[player][0] ]>.9) and (damage[player][ shipPos[player][1] ]>.9) and (damage[player][ shipPos[player][2] ]>.9):
                print ("***All Player " + str(player+1) + "'s ships have been destroyed!***\n\n")
                game = False

        if (game is False):
            print("")
            print("=====================================GAME OVER=====================================")
            print("")
