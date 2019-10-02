import matlab.engine
import time
import json
import pathlib

'''
    Performs multiple timesteps of the simulator at different irrigationo levels.
    
    Will error if the number of timesteps exceeds the number of timesteps left
    in the simulator.

    Inputs:
        eng - matlab.engine object
        irrAmounts - a (1, timesteps) length array of irrigation amounts.
                    The nth amount corresponds to nth timestemp.
        clock_struct - clock struct of first timestep
        initialize_struct - initialize struct of first timestep
        timesteps - the number of timesteps to perform
    Outputs:
        c_struct - clock struct after all timesteps
        i_struct - initialize struct after all timesteps
        states - output states at every timestep
'''
def multiple_runs(eng, clock_struct, initialize_struct, irrAmounts, timesteps):
    c_struct = clock_struct
    i_struct = initialize_struct
    states = []
    for i in range(timesteps):
        if 'ModelTermination' in clock_struct and clock_struct['ModelTermination'] == True:
            raise Exception('Number of timesteps exceeds simulator limit.')
        c_struct, i_struct, state = single_run(eng, c_struct, i_struct, irrAmounts[i])
        states.append(state)
    return c_struct, i_struct, states

'''
    Performs one timestep of the simulator.
    Creates a clock struct and initialize struct from AOS_Initialize.

    Inputs:
        eng - matlab.engine object
        irrAmount - irrigation amoung
        clock_struct - initial clock struct
        initialize_struct - initial initialize struct
    Outputs:
        c_struct - updated clock struct
        i_struct - update initialize struct
        state - state of crop field including canopy cover and water stress level
''' 
def single_run(eng, clock_struct, initialize_struct, irrAmount):
    c_struct, i_struct, state = eng.AOS_PerformUpdate(clock_struct, initialize_struct, irrAmount, nargout=3)
    return c_struct, i_struct, state

'''
    Initializes structs according to initial configuration input files.

    Inputs:
        eng - matlab.engine oobject
    Outputs:
        clock_struct - initial clock struct
        initialize_struct - initial initialize struct
'''
def initialize_structs(eng):
    clock_struct, initialize_struct = eng.AOS_Initialize(nargout=2)
    return clock_struct, initialize_struct

'''
    Writes states and irrigation amounts array to an output file.
    Will create Wrapper_Outputs directory in current directory if one does not exist.

    Inputs:
    states - obtained from runs
    irrAmounts - irrigation amounts where the nth amouht corresponds to the nth state
'''
def write_outputs(states, irrAmounts):
    pathlib.Path('Wrapper_Outputs').mkdir(parents=True, exist_ok=True) 
    filename = "Wrapper_Outputs/" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".json"
    f = open(filename, "w")
    f.write(json.dumps({'irrAmounts': irrAmounts, 'states': states}))
    f.close()

    #open and read the file after the appending:
    f = open(filename, "r")
    print(f.read())

if __name__ == '__main__':
    eng = matlab.engine.start_matlab()
    clock_struct, initialize_structs = initialize_structs(eng)
    irrAmounts = [i for i in range(10)]
    clock_struct, initialize_structs, states = multiple_runs(eng, clock_struct, initialize_structs, irrAmounts, 10)
    # Write outputs
    write_outputs(states, irrAmounts)