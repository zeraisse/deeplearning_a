import torch

# ==============================================================================
# CONFIGURATION GENERALE
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# PARAMETRES POUR gridEnv.py (L'Environnement)
# ==============================================================================
# Directions: 0:Right, 1:Down, 2:Left, 3:Up
DIR_TO_VEC = [
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1)
]

GRID_SIZE = 8       # Taille de la grille (8x8)
MAX_STEPS = 200     # Nombre max de pas avant game over

# ==============================================================================
# HYPERPARAMETERS POUR train.py (Le Modèle & L'Entraînement)
# ==============================================================================
# Architecture TRM
INPUT_DIM = 3       # RGB (3 canaux)
D_MODEL = 128       # Taille vecteur latent (augmenté pour Clé/Porte)
NUM_HEADS = 4       # Attention heads
# Actions: 0:left, 1:right, 2:fwd, 3:pickup, 4:drop, 5:toggle
NUM_ACTIONS = 6     # <--- 6 Actions car on a ajouté Pickup/Toggle
SEQ_LEN = 49        # Vue 7x7 pixels
N_STEPS = 4         # Nombre de pas de réflexion (Recurrent steps)

# Entraînement
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 60
EPISODES = 1500     # Taille du dataset

# ==============================================================================
# PARAMETRES POUR main.py (Affichage & Vidéo)
# ==============================================================================
VIDEO_FILENAME = "trm_result.mp4"
VIDEO_FPS = 5
MAX_VIDEO_STEPS = 60