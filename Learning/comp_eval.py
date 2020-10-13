import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

a_c = np.zeros(100)
a_d = np.zeros(100)

base = '/Users/sebastianoehme/AlphaGarden/Learning/baseline_policy_dataM05_t10_IA0020_trails_random/'
pre = 'data_'
suf = '.pkl'
trails = 20
for i in range(1, trails+1):
    path_to_file = os.path.join(base, pre + str(i) + suf)
    coverage, diversity, _, _ = pickle.load(open(path_to_file, 'rb'))
    a_c += np.array(coverage)
    a_d += np.array(diversity)

a_c = a_c / trails
a_d = a_d / trails

print(sum(a_c[20:70])/50)
print(sum(a_d[20:70])/50)

"""
fig, ax = plt.subplots()
ax.set_ylim([0, 1])
plt.plot(a_c, label='coverage')
plt.plot(a_d, label='diversity')
x = np.arange(len(a_d))
lower = min(a_d) * np.ones(len(a_d))
upper = max(a_d) * np.ones(len(a_d))
plt.plot(x, lower, dashes=[5, 5], label=str(round(min(a_d), 2)))
plt.plot(x, upper, dashes=[5, 5], label=str(round(max(a_d), 2)))
plt.legend()
plt.savefig(base + 'coverage_and_diversity_' + 'companionship' + '.png', bbox_inches='tight', pad_inches=0.02)
plt.clf()
plt.close()"""