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
GRID_SIZE = 10      # <--- ON REDUIT A 10 (C'est le secret de la vitesse)
MAX_STEPS = 500     # L'expert met ~40 pas max dans du 10x10. 500 c'est large.

# ==============================================================================
# HYPERPARAMETERS TRAIN
# ==============================================================================
INPUT_DIM = 3       
D_MODEL = 128       
NUM_HEADS = 4       
NUM_ACTIONS = 7     
SEQ_LEN = 100       # <--- IMPORTANT : 10 x 10 = 100 pixels
N_STEPS = 4         

# Force Brute Maximale
BATCH_SIZE = 1024   # Avec du 10x10, ça rentre LARGE dans la 5070 Ti
LEARNING_RATE = 0.0005 # Un peu plus agressif
EPOCHS = 100        # En mode Dieu (FullyObs), 100 suffisent largement.
EPISODES = 2000     # Suffisant.

# ==============================================================================
# CONFIG VIDEO / SAUVEGARDE
# ==============================================================================
VIDEO_FILENAME = "trm_speedrun.mp4"
VIDEO_FPS = 15
MAX_VIDEO_STEPS = 500
CHECKPOINT_FILE = "checkpoint_last.pth"
BEST_MODEL_FILE = "best_model_weights.pth"
RESUME_TRAINING = False 

# ==============================================================================
# CONFIG DATASET
# ==============================================================================
DATASET_FILE = "dataset_10x10_hud.pt"  
FORCE_NEW_DATASET = True               # OBLIGATOIRE car on a changé la taille !