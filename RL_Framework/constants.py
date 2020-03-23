class DeviceConstants(object):
    CUDA = 'cuda'
    CPU = 'cpu'

class TrainingConstants(object):
    NUM_EPOCHS = 50
    TOTAL_SIZE = 1.0
    VAL_SIZE = 0.2
    BSZ = 24 
    BASE_LR = 0.1
    LR_STEP_SIZE = 1  # Epochs.
    LR_DECAY_RATE = 0.9
    LOG_INTERVAL = 1  # Batches.
    DEVICE = DeviceConstants.CPU
    OUTPUT_DIR = 'nets'
    NET_NAME = 'baseline_net'
    LOG_DIR = 'logs'
    NET_SAVE_FNAME = 'net.pth'
    NUM_CLASSES = 3
    ACT_DIM = NUM_CLASSES
    # ORIGINAL
    # FLAT_STATE_DIM = 18532
    # REDUCED ORIGINAL
    # FLAT_STATE_DIM = 5604
    # ORIGINAL WITH POOLING
    # FLAT_STATE_DIM = 8612
    # REDUCED ORIGINAL WITH POOLING
    # FLAT_STATE_DIM = 2564
    # ORIGINAL WITH POOLING AND FC
    # FLAT_STATE_DIM = 4452
    # REDUCED WITH POOLING AND FC
    # FLAT_STATE_DIM = 2564
    # FURTHER REDUCED WITH POOLING AND FC
    # FLAT_STATE_DIM = 1380
    # MINUTE WITH POOLING AND FC
    # FLAT_STATE_DIM = 1188
    CC_IMG_DIMS = (3, 235, 499)
    CC_DS_DIMS = (3, 58, 124)
    RAW_DIMS = (12, 15, 30)
    GLOBAL_CC_DIMS = (10, 1, 1)
