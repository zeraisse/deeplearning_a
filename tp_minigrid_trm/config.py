import torch

# ==============================================================================
# CONFIGURATION GENERALE
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# PARAMETRES POUR gridEnv.py
# ==============================================================================
DIR_TO_VEC = [
    (1, 0), (0, 1), (-1, 0), (0, -1)
]

GRID_SIZE = 12      # <--- 12x12 : C'est grand !
MAX_STEPS = 800     # Plus de temps nécessaire

# ==============================================================================
# HYPERPARAMETERS POUR train.py
# ==============================================================================
INPUT_DIM = 3       
D_MODEL = 128       # Cerveau assez gros pour gérer Murs + Clé + Lave
NUM_HEADS = 4       
NUM_ACTIONS = 7    # 0:left, 1:right, 2:fwd, 3:pickup, 4:drop, 5:toggle
SEQ_LEN = 49        # Toujours une vue locale 7x7
N_STEPS = 4         # 4 cycles de réflexion

BATCH_SIZE = 1024
LEARNING_RATE = 0.001
EPOCHS = 300         # Il faut du temps pour converger
EPISODES = 20000     # Il faut BEAUCOUP d'exemples pour que l'expert montre comment éviter la lave

# ==============================================================================
# PARAMETRES POUR main.py
# ==============================================================================
VIDEO_FILENAME = "trm_lava_result.mp4"
VIDEO_FPS = 10      # Rapide
MAX_VIDEO_STEPS = 1000

# ==============================================================================
# PARAMETRES SAUVEGARDE (CHECKPOINT)
# ==============================================================================
CHECKPOINT_FILE = "checkpoint_last.pth"      # Pour reprendre en cas de crash
BEST_MODEL_FILE = "best_model_weights.pth"   # Le meilleur modèle obtenu
RESUME_TRAINING = True