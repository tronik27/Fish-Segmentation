import os

# Paths to data.
PATH = 'path/to/folder/containing/dataset'
PATH_TO_DATA = 'data.json'
WORK_DATA_PATH = 'path/to/folder/containing/images/for/classification'
VALID_SIZE = 0.2
TEST_SIZE = 0.1

# Training parameters.
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 35

# Custom model parameters.
NUM_FILTERS = 128
MODEL_NAME = 'custom_DeeplabV3Plus'
NUM_CLASSES = 9
INPUT_SHAPE = (256, 256, 3)
INPUT_NAME = 'input'
OUTPUT_NAME = 'output'

# Paths to weights and saved model.
WEIGHTS_PATH = 'fish_segmentation_model/weights_3'
MODEL_PATH = '{} (trained_model)'.format(MODEL_NAME)

CLASS_NAMES = os.listdir(PATH)

# augmentation configuration
AUG_CONFIG = ['rotate', 'sharpen', 'rgb_shift', 'brightness_contrast', 'hue_saturation',
              'distortion', 'blur', 'noise', 'crop']

# service parameters
SHOW_LEARNING_CURVES = True
SHOW_IMAGE_DATA = True
NUM_OF_EXAMPLES = 1
