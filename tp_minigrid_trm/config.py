import torch

# ==============================================================================
# CONFIG GENERALE
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# CONFIG ENVIRONNEMENT
# ==============================================================================
DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]
GRID_SIZE = 14      # <--- TRES GRAND !
MAX_STEPS = 2000    # Il faut du temps pour traverser 3 salles

# ==============================================================================
# HYPERPARAMETERS TRAIN
# ==============================================================================
INPUT_DIM = 3       
D_MODEL = 128       
NUM_HEADS = 4       
NUM_ACTIONS = 7     # 0:left, 1:right, 2:fwd, 3:pickup, 4:drop, 5:toggle, 6:done
SEQ_LEN = 49        
N_STEPS = 4         

BATCH_SIZE = 1024   # On garde la puissance de la 5070 Ti
LEARNING_RATE = 0.001
EPOCHS = 200        # Suffisant avec le HUD
EPISODES = 25000    # Dataset massif pour couvrir les pièges de lave

# ==============================================================================
# CONFIG VIDEO / SAUVEGARDE
# ==============================================================================
VIDEO_FILENAME = "trm_hardcore.mp4"
VIDEO_FPS = 15
MAX_VIDEO_STEPS = 1000
CHECKPOINT_FILE = "checkpoint_last.pth"
BEST_MODEL_FILE = "best_model_weights.pth"
RESUME_TRAINING = False # <--- ON REPART A ZERO PROPREMENT