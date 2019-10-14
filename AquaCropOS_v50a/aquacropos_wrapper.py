import matlab.engine
import time
import json
import pathlib
import numpy as np

class AquaCropOSWrapper(object):
    def __init__(self):
        self.eng = matlab.engine.start_matlab()
        clock_struct, initialize_struct = self.initialize_structs()
        self.clock_struct = clock_struct
        self.initialize_struct = initialize_struct
        # Determined by Inputs/Clock.txt duration
        self.max_time_steps = 182
        self.state = np.array([0, 0])

    '''
    Performs multiple timesteps of the simulator at different irrigation levels.

    Will error if the number of timesteps exceeds the number of timesteps left
    in the simulator.

    Parameters:
        irrAmounts - a (1, timesteps) length array of irrigation amounts.
                    The nth amount corresponds to nth timestemp.
        timesteps - the number of timesteps to perform
    Returns:s
        c_struct - clock struct after all timesteps
        i_struct - initialize struct after all timesteps
        states - output states at every timestep
    '''
    def multiple_runs(self, irrAmounts, timesteps):
        c_struct = self.clock_struct
        i_struct = self.initialize_struct
        states = []
        for i in range(timesteps):
            if 'ModelTermination' in self.clock_struct and self.clock_struct['ModelTermination'] == True:
                raise Exception('Number of timesteps exceeds simulator limit.')
            c_struct, i_struct, state = self.single_run(irrAmounts[i])
            states.append(state)
        return c_struct, i_struct, states

    '''
    Performs one timestep of the simulator.
    Creates a clock struct and initialize struct from AOS_Initialize.

    Parameters:
        irrAmount - irrigation amoung
    Returns:
        c_struct - updated clock struct
        i_struct - update initialize struct
        state - state of crop field including canopy cover and water stress level
    ''' 
    def single_run(self, irrAmount):
        c_struct, i_struct, state = self.eng.AOS_PerformUpdate(self.clock_struct, self.initialize_struct, irrAmount, nargout=3)
        self.clock_struct = c_struct
        self.initialize_struct = i_struct
        self.state = state
        return c_struct, i_struct, state

    '''
    Initializes structs according to initial configuration input files.

    Returns:
        clock_struct - initial clock struct
        initialize_struct - initial initialize struct
    '''
    def initialize_structs(self):
        clock_struct, initialize_struct = self.eng.AOS_Initialize(nargout=2)
        return clock_struct, initialize_struct

    '''
        Writes clock and initialize structs to json files.
    '''
    def write_structs(self):
        pathlib.Path('Wrapper_Parameters').mkdir(parents=True, exist_ok=True) 
        # Overwrite existing structs
        f = open('Wrapper_Parameters/clock_struct.json', 'w')
        f.write(json.dumps(self.clock_struct))
        f.close()

        f = open('Wrapper_Parameters/initialize_struct.json', 'w')
        f.write(json.dumps(self.initialize_struct))
        f.close()

    '''
    Reads existing clock and initialize structs.

    Returns;
        clock_struct = previous clock struct
        initialize_struct = previous initialize struct
    '''
    def read_structs(self):
        with open('Wrapper_Parameters/clock_struct.json') as f_in:
            clock_struct = json.load(f_in)
        with open('Wrapper_Parameters/initialize_struct.json') as f_in:
            initialize_struct = json.load(f_in)
        return clock_struct, initialize_struct

    '''
    Writes states and irrigation amounts array to an output file.
    Will create Wrapper_Returns directory in current directory if one does not exist.

    Parameters:
        states - obtained from runs
        irrAmounts - irrigation amounts where the nth amouht corresponds to the nth state
    '''
    def write_Returns(self, states, irrAmounts):
        pathlib.Path('Wrapper_Returns').mkdir(parents=True, exist_ok=True) 
        filename = 'Wrapper_Returns/' + time.strftime('%Y-%m-%d-%H-%M-%S') + '.json'
        f = open(filename, 'w')
        f.write(json.dumps({'irrAmounts': irrAmounts, 'states': states}))
        f.close()

    '''
    Method called by the gym environment to execute an action.

    Parameters:
        action - irrigation amount to apply
    Returns:
        state - state of the environment after irrigation
    '''
    def _take_action(self, action):
        self.single_run(action)
        return self.state