class DeviceConstants(object):
    CUDA = 'cuda'
    CPU = 'cpu'

class TrainingConstants(object):
    NUM_EPOCHS = 100 
    TOTAL_SIZE = 1.0
    VAL_SIZE = 0.2
    BSZ = 300 
    BASE_LR = 0.1
    LR_STEP_SIZE = 1  # Epochs.
    LR_DECAY_RATE = 0.9
    LOG_INTERVAL = 1  # Batches.
    DEVICE = DeviceConstants.CPU
    OUTPUT_DIR = 'nets'
    NET_NAME = 'baseline_net'
    LOG_DIR = 'logs'
    NET_SAVE_FNAME = 'net.pth'
    NUM_CLASSES = 6
    ACT_DIM = NUM_CLASSES
    FLAT_STATE_DIM = 6219
    # FLAT_STATE_DIM = 6027
    # FLAT_STATE_DIM = 1867