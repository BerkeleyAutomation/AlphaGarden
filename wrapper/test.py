import matlab.engine

eng = matlab.engine.start_matlab()
tf = eng.isprime(37)
print(tf)
