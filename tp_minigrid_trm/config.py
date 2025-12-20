import torch

# ==============================================================================
# CONFIG GENERALE
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

# ==============================================================================
# CONFIG ENVIRONNEMENT
# ==============================================================================
DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]
GRID_SIZE = 10     
MAX_STEPS = 500

# ==============================================================================
# HYPERPARAMETERS TRAIN
# ==============================================================================
INPUT_DIM = 3       
D_MODEL = 128       
NUM_HEADS = 4       
NUM_ACTIONS = 7     
SEQ_LEN = 100       
N_STEPS = 4         

BATCH_SIZE = 1024   
LEARNING_RATE = 0.0001 
EPOCHS = 300           
EPISODES = 2000     

# ==============================================================================
# SAUVEGARDE
# ==============================================================================
VIDEO_FILENAME = "trm_classic.mp4"
VIDEO_FPS = 15
MAX_VIDEO_STEPS = 500
CHECKPOINT_FILE = "checkpoint_last.pth"
BEST_MODEL_FILE = "best_model_weights.pth"
RESUME_TRAINING = False  

# ==============================================================================
# DATASET
# ==============================================================================
DATASET_FILE = "dataset_10x10_hud.pt"  
FORCE_NEW_DATASET = False 