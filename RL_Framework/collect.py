import subprocess
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str) # directory
    parser.add_argument('-n', type=str) # num of batches
    args = parser.parse_args()
    params = vars(args)

    procs = []
    for idx in range(int(params['n'])):
        dir = params['d'] + 'dataset_' + str(idx + 1)
        if not os.path.exists(dir):
            os.mkdir(dir)
        print('DIR', dir)
        args = ('python RL_Framework/data_collection.py' + ' -d' + dir + '/' + ' -s' + str(idx))
        proc = subprocess.Popen(args, shell=True)
        procs.append(proc)

    i = 0
    while(len(procs)):
        i = i % len(procs)
        if procs[i].poll() is None:
            i = (i + 1) % len(procs)
        else:
            del procs[i]

if __name__ == "__main__":
    import os
    cpu_cores = [i for i in range(61, 76)] # Cores (numbered 0-11)
    os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))
    main()
