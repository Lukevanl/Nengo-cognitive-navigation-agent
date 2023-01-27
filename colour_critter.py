import grid
import nengo
import nengo.spa as spa
import numpy as np 
import detectors as det

#we can change the map here using # for walls and RGBMY for various colours
mymap="""
#######
#  M R#
# # # #
# #B# #
#G Y R#
#######
"""

#### Preliminaries - this sets up the agent and the environment ################ 

class Cell(grid.Cell):
    def color(self):
        if self.wall:
            return 'black'
        elif self.cellcolor == 1:
            return 'green'
        elif self.cellcolor == 2:
            return 'red'
        elif self.cellcolor == 3:
            return 'blue'
        elif self.cellcolor == 4:
            return 'magenta'
        elif self.cellcolor == 5:
            return 'yellow'
        return None

    def load(self, char):
        self.cellcolor = 0
        if char == '#':
            self.wall = True
        if char == 'G':
            self.cellcolor = 1
        elif char == 'R':
            self.cellcolor = 2
        elif char == 'B':
            self.cellcolor = 3
        elif char == 'M':
            self.cellcolor = 4
        elif char == 'Y':
            self.cellcolor = 5

world = grid.World(Cell, map=mymap, directions=int(4))
body = grid.ContinuousAgent()
world.add(body, x=1, y=2, dir=2)
#this defines the RGB values of the colours. We use this to translate the "letter" in 
#the map to an actual colour. Note that we could make some or all channels noisy if we
#wanted to
col_values = {
    0: [0.9, 0.9, 0.9], # White
    1: [0.2, 0.8, 0.2], # Green
    2: [0.8, 0.2, 0.2], # Red
    3: [0.2, 0.2, 0.8], # Blue
    4: [0.8, 0.2, 0.8], # Magenta
    5: [0.8, 0.8, 0.2], # Yellow
}

KEYS = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'MAGENTA', 'WHITE']
VALUES = [[0.8, 0.2, 0.2], [0.2, 0.8, 0.2],  [0.2, 0.2, 0.8], [0.8, 0.8, 0.2], [0.8, 0.2, 0.8], [0.9, 0.9, 0.9]]

D = 32

vocab_binary = spa.Vocabulary(D)
vocab_color = spa.Vocabulary(D)
noise_val = 0 # how much noise there will be in the colour info

#You do not have to use spa.SPA; you can also do this entirely with nengo.Network()
model = spa.SPA()
with model:
    # create a node to connect to the world we have created (so we can see it)
    env = grid.GridNode(world, dt=0.005)
    
    ### Input and output nodes - how the agent sees and acts in the world ######

    #--------------------------------------------------------------------------#
    # This is the output node of the model and its corresponding function.     #
    # It has two values that define the speed and the rotation of the agent    #
    #--------------------------------------------------------------------------#
    def move(t, x):

        speed, rotation = x
        dt = 0.001
        max_speed = 20.0
        max_rotate = 10.0
        body.turn(rotation * dt * max_rotate)
        body.go_forward(speed * dt * max_speed)

    movement = nengo.Node(move, size_in=2)

    #--------------------------------------------------------------------------#
    # First input node and its function: 3 proximity sensors to detect walls   #
    # up to some maximum distance ahead                                        #
    #--------------------------------------------------------------------------#
    
    def detect(t):
        angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]
        
    proximity_sensors = nengo.Node(detect)

    #--------------------------------------------------------------------------#
    # Second input node and its function: the colour of the current cell of    #
    # agent                                                                    #
    #--------------------------------------------------------------------------#
    
    def cell2rgb(t):
        c = col_values.get(body.cell.cellcolor)
        noise = np.random.normal(0, noise_val,3)
        c = np.clip(c + noise, 0, 1)
        return c

    current_color = nengo.Node(cell2rgb)

    #--------------------------------------------------------------------------#
    # Final input node and its function: the colour of the next non-whilte     #
    # cell (if any) ahead of the agent. We cannot see through walls.           #
    #--------------------------------------------------------------------------#
    
    def look_ahead(t):
        done = False
        cell = body.cell.neighbour[int(body.dir)]
        if cell.cellcolor > 0:
            done = True 
        while cell.neighbour[int(body.dir)].wall == False and not done:
            cell = cell.neighbour[int(body.dir)]
            if cell.cellcolor > 0:
                done = True
        c = col_values.get(cell.cellcolor)
        noise = np.random.normal(0, noise_val,3)
        c = np.clip(c + noise, 0, 1)
        return c
    #Create one-hot encoding where the index of the color x is set to 1
    def color_detector(x):
        one_hot = [0,0,0,0,0,0]
        for i, x_i in enumerate(VALUES):
            if (x_i == x).all():
                one_hot[i] = 1
        return one_hot

    ahead_color = nengo.Node(look_ahead)    
    #Detect current color, and keep in memory:
    
    #Switches to pick up on changes in color
    model.red_switch = spa.State(D, vocab=vocab_binary)
    model.green_switch = spa.State(D, vocab=vocab_binary)
    model.blue_switch = spa.State(D, vocab=vocab_binary)
    model.yellow_switch = spa.State(D, vocab=vocab_binary)
    model.magenta_switch = spa.State(D, vocab=vocab_binary)
    current_color_detector_node = nengo.Node(size_in=6)

    nengo.Connection(current_color, current_color_detector_node, function=color_detector)

    ### Agent functionality - your code adds to this section ###################

    #Binary vocab
    vocab_binary.parse("TRUE+FALSE")
    #Color vocab
    vocab_color.parse("RED+GREEN+BLUE+YELLOW+MAGENTA+WHITE")
    
    #Capture changes in color
    deriv_color = nengo.Ensemble(n_neurons= 1000, dimensions = 6, radius=3)
    deriv_color2 = nengo.Ensemble(n_neurons= 1000, dimensions = 6, radius=3)
    nengo.Connection(current_color_detector_node, deriv_color)
    nengo.Connection(current_color_detector_node, deriv_color, transform=-1, synapse=0.05)
    nengo.Connection(deriv_color, deriv_color2, function= lambda x: x*5)
    
    #Send changes to the switches
    nengo.Connection(deriv_color2[0], model.red_switch.input, transform=5*vocab_binary["TRUE"].v.reshape(D, 1))
    nengo.Connection(deriv_color2[1], model.green_switch.input, transform=5*vocab_binary["TRUE"].v.reshape(D, 1))
    nengo.Connection(deriv_color2[2], model.blue_switch.input, transform=5*vocab_binary["TRUE"].v.reshape(D, 1))
    nengo.Connection(deriv_color2[3], model.yellow_switch.input, transform=5*vocab_binary["TRUE"].v.reshape(D, 1))
    nengo.Connection(deriv_color2[4], model.magenta_switch.input, transform=5*vocab_binary["TRUE"].v.reshape(D, 1))

    #Memorize the switch activities
    model.memory_red = spa.State(D, vocab=vocab_binary)
    model.cleanupred = spa.AssociativeMemory(input_vocab=vocab_binary, wta_output=True)
    nengo.Connection(model.cleanupred.output, model.memory_red.input, synapse=0.01)
    nengo.Connection(model.memory_red.output, model.cleanupred.input, synapse=0.01)

    model.memory_green = spa.State(D, vocab=vocab_binary)
    model.cleanupgreen = spa.AssociativeMemory(input_vocab=vocab_binary, wta_output=True)
    nengo.Connection(model.cleanupgreen.output, model.memory_green.input, synapse=0.01)
    nengo.Connection(model.memory_green.output, model.cleanupgreen.input, synapse=0.01)

    model.memory_blue = spa.State(D, vocab=vocab_binary)
    model.cleanupblue = spa.AssociativeMemory(input_vocab=vocab_binary, wta_output=True)
    nengo.Connection(model.cleanupblue.output, model.memory_blue.input, synapse=0.01)
    nengo.Connection(model.memory_blue.output, model.cleanupblue.input, synapse=0.01)
    
    model.memory_yellow = spa.State(D, vocab=vocab_binary)
    model.cleanupyellow = spa.AssociativeMemory(input_vocab=vocab_binary, wta_output=True)
    nengo.Connection(model.cleanupyellow.output, model.memory_yellow.input, synapse=0.01)
    nengo.Connection(model.memory_yellow.output, model.cleanupyellow.input, synapse=0.01)

    model.memory_magenta = spa.State(D, vocab=vocab_binary)
    model.cleanupmagenta = spa.AssociativeMemory(input_vocab=vocab_binary, wta_output=True)
    nengo.Connection(model.cleanupmagenta.output, model.memory_magenta.input, synapse=0.01)
    nengo.Connection(model.memory_magenta.output, model.cleanupmagenta.input, synapse=0.01)



    #All input nodes should feed into one ensemble. Here is how to do this for
    #the radar, see if you can do it for the others
    walldist = nengo.Ensemble(n_neurons=500, dimensions=3, radius=4)
    nengo.Connection(proximity_sensors, walldist)
    #For now, all our agent does is wall avoidance. It uses values of the radar
    #to: a) turn away from walls on the sides and b) slow down in function of 
    #the distance to the wall ahead, reversing if it is really close
    def movement_func(x):
        turn = (x[2] - x[0]) 
        spd = (x[1] - 0.5) / 4
        return spd, turn

    #If switch is active, memorize it
    actions = spa.Actions(
        'dot(red_switch, TRUE) --> memory_red=TRUE',
        'dot(green_switch, TRUE) --> memory_green=TRUE',
        'dot(blue_switch, TRUE) --> memory_blue=TRUE',
        'dot(yellow_switch, TRUE) --> memory_yellow=TRUE',
        'dot(magenta_switch, TRUE) --> memory_magenta=TRUE',
        '0.7 --> '
        )

    model.bg = spa.BasalGanglia(actions)
    model.thalamus = spa.Thalamus(model.bg)


    #the movement function is only driven by information from the radar, so we
    #can connect the radar ensemble to the output node with this function 
    #directly. In the assignment, you will need intermediate steps
    
    #QA-system:
    model.QA_question = spa.State(D, vocab=vocab_color)
    model.QA_answer = spa.State(D, vocab=vocab_binary)
    model.cleanup_qa_answer = spa.AssociativeMemory(input_vocab=vocab_binary, wta_output=True)
    nengo.Connection(model.cleanup_qa_answer.output, model.QA_answer.input, synapse=0.01)
    nengo.Connection(model.QA_answer.output, model.cleanup_qa_answer.input, synapse=0.01)
    #If question is about a certain color, and that color has been seen, return TRUE.
    actions_qa = spa.Actions(
        'dot(QA_question, RED) + dot(memory_red, TRUE) --> QA_answer=10*TRUE - FALSE',
        'dot(QA_question, GREEN) + dot(memory_green, TRUE) --> QA_answer=10*TRUE - FALSE',
        'dot(QA_question, BLUE) + dot(memory_blue, TRUE)--> QA_answer=10*TRUE - FALSE',
        'dot(QA_question, YELLOW) + dot(memory_yellow, TRUE)--> QA_answer=10*TRUE - FALSE',
        'dot(QA_question, MAGENTA) + dot(memory_magenta, TRUE)--> QA_answer=10*TRUE - FALSE',
        '1.35 --> QA_answer=FALSE - TRUE'
        )
        

    model.bg_qa = spa.BasalGanglia(actions_qa)
    model.thalamus_qa = spa.Thalamus(model.bg_qa)
    nengo.Connection(walldist, movement, function=movement_func)
    
    #Colour sequence problem
    #Detect color ahead:
    ahead_color_detector_node = nengo.Node(size_in=6)

    nengo.Connection(ahead_color, ahead_color_detector_node, function=color_detector)

    #Same as before but now for the color ahead
    model.red_ahead_switch = spa.State(D, vocab=vocab_binary)
    model.green_ahead_switch = spa.State(D, vocab=vocab_binary)
    model.blue_ahead_switch = spa.State(D, vocab=vocab_binary)
    model.yellow_ahead_switch = spa.State(D, vocab=vocab_binary)
    model.magenta_ahead_switch = spa.State(D, vocab=vocab_binary)

    deriv_color_ahead = nengo.Ensemble(n_neurons= 1000, dimensions = 6, radius=3)
    deriv_color2_ahead = nengo.Ensemble(n_neurons= 1000, dimensions = 6, radius=3)
    
    nengo.Connection(ahead_color_detector_node, deriv_color_ahead)
    nengo.Connection(ahead_color_detector_node, deriv_color_ahead, transform=-1, synapse=0.05)
    nengo.Connection(deriv_color_ahead, deriv_color2_ahead, function= lambda x: x*5)
    
    nengo.Connection(deriv_color2_ahead[0], model.red_ahead_switch.input, transform=8*vocab_binary["TRUE"].v.reshape(D, 1))
    nengo.Connection(deriv_color2_ahead[1], model.green_ahead_switch.input, transform=8*vocab_binary["TRUE"].v.reshape(D, 1))
    nengo.Connection(deriv_color2_ahead[2], model.blue_ahead_switch.input, transform=8*vocab_binary["TRUE"].v.reshape(D, 1))
    nengo.Connection(deriv_color2_ahead[3], model.yellow_ahead_switch.input, transform=8*vocab_binary["TRUE"].v.reshape(D, 1))
    nengo.Connection(deriv_color2_ahead[4], model.magenta_ahead_switch.input, transform=8*vocab_binary["TRUE"].v.reshape(D, 1))

    #Now only one memory cell which holds the color instead of 5 different memory cells for each color
    model.memory_color_ahead = spa.State(D, vocab=vocab_color)
    model.cleanup_color_ahead = spa.AssociativeMemory(input_vocab=vocab_color, wta_output=True)
    nengo.Connection(model.cleanup_color_ahead.output, model.memory_color_ahead.input, synapse=0.01)
    nengo.Connection(model.memory_color_ahead.output, model.cleanup_color_ahead.input, synapse=0.01)
    
    actions_ahead = spa.Actions(
        'dot(red_ahead_switch, TRUE) --> memory_color_ahead=2*RED-GREEN-BLUE-YELLOW-MAGENTA-WHITE',
        'dot(green_ahead_switch, TRUE) --> memory_color_ahead=2*GREEN-RED-BLUE-YELLOW-MAGENTA-WHITE',
        'dot(blue_ahead_switch, TRUE) --> memory_color_ahead=2*BLUE-GREEN-RED-YELLOW-MAGENTA-WHITE',
        'dot(yellow_ahead_switch, TRUE) --> memory_color_ahead=2*YELLOW-GREEN-BLUE-RED-MAGENTA-WHITE',
        'dot(magenta_ahead_switch, TRUE) --> memory_color_ahead=2*MAGENTA-GREEN-BLUE-RED-YELLOW-WHITE',
        '0.7 --> memory_color_ahead=0.5*WHITE-GREEN-BLUE-RED-YELLOW-MAGENTA'
        )

    model.bg_ahead = spa.BasalGanglia(actions_ahead)
    model.thalamus_ahead = spa.Thalamus(model.bg_ahead)
    
    #Color stepper, commented out because it is noisy. Feel free to uncomment and try it out (it does run)
    # model.color_stepper = spa.State(D, vocab=vocab_color)
    # model.start_stepper = spa.State(D, vocab=vocab_color)
    
    # model.cleanup_stepper = spa.AssociativeMemory(input_vocab=vocab_color, wta_output=True)
    # nengo.Connection(model.cleanup_stepper.output, model.color_stepper.input, synapse=0.01)
    # nengo.Connection(model.color_stepper.output, model.cleanup_stepper.input, synapse=0.01)
    
    # model.cleanup_starter = spa.AssociativeMemory(input_vocab=vocab_color, wta_output=True)
    # nengo.Connection(model.cleanup_starter.output, model.start_stepper.input, synapse = 0.01)
    # nengo.Connection(model.start_stepper.output, model.cleanup_starter.input, synapse = 0.01)
    
    # actions_stepper = spa.Actions(
    # '(dot(color_stepper, GREEN)+dot(green_switch, TRUE))/2 --> start_stepper=10*STARTED-GREEN, color_stepper=YELLOW-GREEN',
    # '(dot(color_stepper, YELLOW)+dot(yellow_switch, TRUE))/2 --> start_stepper=10*STARTED-GREEN, color_stepper=RED-YELLOW',
    # '(dot(color_stepper, RED)+dot(red_switch, TRUE))/2 --> start_stepper=10*STARTED-GREEN, color_stepper=DONE-RED',
    # 'dot(start_stepper, GREEN) --> color_stepper=GREEN, start_stepper=10*STARTED-GREEN',
    # '0.85 --> '
    # )

    # model.bg_stepper = spa.BasalGanglia(actions_stepper)
    # model.thalamus_stepper = spa.Thalamus(model.bg_stepper)
    
    #Navigation, unfinsihed but is able to noisily determine whether the color ahead is safe or not:
    model.color_ahead_safe = spa.State(D, vocab=vocab_binary)
    model.cleanup_color_ahead_safe = spa.AssociativeMemory(input_vocab=vocab_binary, wta_output=True)
    nengo.Connection(model.cleanup_color_ahead_safe.output, model.color_ahead_safe.input, synapse=0.01)
    nengo.Connection(model.color_ahead_safe.output, model.cleanup_color_ahead_safe.input, synapse=0.01)
    
    actions_safe = spa.Actions(
        'dot(memory_color_ahead, RED) + dot(TRUE, memory_red) --> color_ahead_safe=3*FALSE-TRUE',
        'dot(memory_color_ahead, GREEN) + dot(TRUE, memory_green) --> color_ahead_safe=3*FALSE-TRUE',
        'dot(memory_color_ahead, BLUE) + dot(TRUE, memory_blue) --> color_ahead_safe=3*FALSE-TRUE',
        'dot(memory_color_ahead, YELLOW) + dot(TRUE, memory_yellow) --> color_ahead_safe=3*FALSE-TRUE',
        'dot(memory_color_ahead, MAGENTA) + dot(TRUE, memory_magenta) --> color_ahead_safe=3*FALSE-TRUE',
        '1.5 --> color_ahead_safe=0.3*TRUE-FALSE'
        )
 
    model.bg_safe = spa.BasalGanglia(actions_safe)
    model.thalamus_safe = spa.Thalamus(model.bg_safe)