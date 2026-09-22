from pathlib import Path


SEED = 42

# dataset and data loader
ASSIGNMENT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ASSIGNMENT_PATH / "data"
OUTPUT_PATH = ASSIGNMENT_PATH / "outputs"
CHECKPOINT_PATH = ASSIGNMENT_PATH / "checkpoints"

MEAN_TUP = (0.2860,)
SD_TUP = (0.3530,)
VAL_SIZE = 0.1 # 6000 data samples
BATCH_SIZE = 256
NUM_WORKERS = 1

LEARNING_RATE = 0.001
WD = 0.1
NUM_EPOCHS = 10
MAX_ATTEMPT = 10

# LinearModel
INPUT_DIM = 784
OUTPUT_DIM = 10

# MLPModel
DROP_OUT = 0.2
DROP_STEP = 0.1

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
