class DeviceConstants(object):
    CUDA = 'cuda'
    CPU = 'cpu'

class TrainingConstants(object):
    NUM_EPOCHS = 60
    TOTAL_SIZE = 1.0
    VAL_SIZE = 0.2
    BSZ = 8
    BASE_LR = 0.1
    LR_STEP_SIZE = 1  # Epochs.
    LR_DECAY_RATE = 0.9
    LOG_INTERVAL = 1  # Batches.
    DEVICE = DeviceConstants.CUDA
    OUTPUT_DIR = 'nets'
    NET_NAME = 'baseline_net'
    LOG_DIR = 'logs'
    NET_SAVE_FNAME = 'net.pth'
    NUM_CLASSES = 3
    ACT_DIM = BSZ * NUM_CLASSES # classes 
    FLAT_STATE_DIM = 15936800
    CC_IMG_DIMS = (3, 235, 499)
    RAW_DIMS = (12, 15, 30)
    GLOBAL_CC_DIMS = (10, 1, 1)