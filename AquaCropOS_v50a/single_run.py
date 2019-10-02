import matlab.engine

eng = matlab.engine.start_matlab()

clock_struct, initialize_struct = eng.AOS_Initialize(nargout=2)

while 'ModelTermination' not in clock_struct:
    clock_struct, initialize_struct, state = eng.AOS_PerformUpdate(clock_struct, initialize_struct, 10, nargout=3)

eng.AOS_Finish(clock_struct, initialize_struct, nargout=0)