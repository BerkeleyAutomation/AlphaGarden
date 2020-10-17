import subprocess
import argparse
import os
from multiprocessing import Pool

def open_proc(dir_idx):
    args = ('python Learning/data_collection.py' + ' -d' + dir_idx[0] + '/' + ' -s' + str(dir_idx[1]))
    subprocess.Popen(args, shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str) # directory
    parser.add_argument('-n', type=str) # num of batches
    args = parser.parse_args()
    params = vars(args)

    dir_idxs = []
    for idx in range(int(params['n'])):
        dir = params['d'] + '/dataset_' + str(idx // 12000)
        if not os.path.exists(dir):
            os.makedirs(dir)
        print('DIR', dir)
        dir_idxs.append((dir, idx))
    
    p = Pool(2)
    p.map(open_proc, dir_idxs)

if __name__ == "__main__":
    import os
    cpu_cores = [i for i in range(0, 2)] # Cores (numbered 0-11)
    os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))
    main()
