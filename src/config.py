import torch
ROOT_PATH = '/home/nandhini/facial recog/input/facial-keypoints-detection'
OUTPUT_PATH = '/home/nandhini/facial recog/outputs'
BATCH_SIZE = 256
LR = 0.0001
EPOCHS = 300
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#train test split
TEST_SPLIT = 0.2
SHOW_DATASET_PLOT = True