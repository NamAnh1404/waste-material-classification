RAW_DIR = "data/raw"
SPLIT_DIR = "data/split"
TRAIN_DIR = "data/split/train"
VAL_DIR = "data/split/val"
TEST_DIR = "data/split/test"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MODEL_DIR = "models"
RESULTS_DIR = "results"
TSNE_DIR = "results/tsne"
MISR_DIR = "results/misclassified"

CLASS_INDICES_PATH = "models/class_indices.json"

CNN_MODEL_PATH = "models/cnn_waste_cnn_strong_cw.keras"
EFF_MODEL_PATH = "models/efficientnet_waste.keras"
RESNET_MODEL_PATH = "models/resnet50_waste.keras"

SUMMARY_CSV = "results/experiment_results.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

EPOCHS_CNN = 2
EPOCHS_STAGE_1 = 1
EPOCHS_STAGE_2 = 1
