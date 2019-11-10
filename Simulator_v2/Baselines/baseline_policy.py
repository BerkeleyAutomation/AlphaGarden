import numpy as np

def baseline_policy(state, step, threshold, amount, irr_threshold):
    state = np.copy(state[0])
    action = np.zeros(state.shape[0:2])
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            radius = np.amax(state[i,j,0:-1])
            if radius > 0:
                lower_x, upper_x = int(max(0, i - (radius // step))), int(min(state.shape[0], i + (radius // step)))
                lower_y, upper_y = int(max(0, j - (radius // step))), int(min(state.shape[1], j + (radius // step)))
                water_available = np.sum(state[lower_x:upper_x+1,lower_y:upper_y+1,-1])
                if water_available < threshold:
                    action[j,i] = amount
                    for k in range(max(0, i - irr_threshold), min(state.shape[0], i + irr_threshold + 1)):
                        for l in range(max(0, j - irr_threshold), min(state.shape[1], j + irr_threshold + 1)):
                            state[i,j,-1] += amount
    return np.expand_dims(action.flatten(),axis=0)
