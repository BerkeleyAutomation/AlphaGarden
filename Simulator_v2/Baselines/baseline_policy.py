import numpy as np

def baseline_policy(state, step, threshold, amount, irr_threshold):
    state = np.copy(state[0])
    action = np.zeros(state.shape[0:2])
    k = (state.shape[2] - 1) // 2
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            plant_exists = np.amax(state[i,j,0:k])
            if plant_exists:
                water_available = state[i,j,-1]
                if water_available < threshold:
                    action[j,i] = amount
                    for k in range(max(0, i - irr_threshold), min(state.shape[0], i + irr_threshold + 1)):
                        for l in range(max(0, j - irr_threshold), min(state.shape[1], j + irr_threshold + 1)):
                            state[i,j,-1] += amount
    return np.expand_dims(action.flatten(),axis=0)
