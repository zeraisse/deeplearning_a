import torch
import torch.nn as nn
import torch.nn.functional as F

class DARQN(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_size=512):
        """
        Deep Attention Recurrent Q-Network
        :param input_shape: (Channels, Height, Width) ex: (4, 84, 84)
        :param num_actions: Nombre d'actions possibles (Pacman: 5 ou 9)
        :param hidden_size: Taille de la mémoire LSTM
        """
        super(DARQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        
        # 1. Feature Extractor (CNN standard type Nature-DQN)
        # On réduit l'image en "cartes de caractéristiques" (Feature Maps)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calcul automatique de la taille de sortie du CNN
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)
        
        self.conv_output_size = convw * convh * 64
        # Pour l'attention, on a besoin du nombre de "zones" (grid size) et de la profondeur (channels)
        self.feature_grid_size = convw * convh # L (Locations)
        self.feature_depth = 64                # D (Depth)
        
        # 2. Attention Mechanism
        # On apprend à pondérer les zones de l'image selon l'état caché précédent (h_{t-1})
        self.att_conv = nn.Linear(self.feature_depth, self.feature_depth) 
        self.att_hidden = nn.Linear(hidden_size, self.feature_depth)
        self.att_out = nn.Linear(self.feature_depth, 1)
        
        # 3. Recurrent Layer (LSTM)
        # Input: Features pondérées (Context vector)
        self.lstm = nn.LSTMCell(self.feature_depth, hidden_size)
        
        # 4. Action Head (Q-Values)
        self.fc_adv = nn.Linear(hidden_size, num_actions) # Advantage
        self.fc_val = nn.Linear(hidden_size, 1)           # Value (Dueling DQN structure bonus)

    def forward(self, x, hidden_state):
        """
        :param x: Batch d'images (Batch, C, H, W)
        :param hidden_state: Tuple (h_t, c_t) du pas précédent
        """
        batch_size = x.size(0)
        h_t, c_t = hidden_state
        
        # --- A. Convolution ---
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) 
        # x shape: (Batch, 64, H_grid, W_grid)
        
        # Aplatir spatialement pour l'attention : (Batch, L, D)
        # L = H_grid * W_grid (nombre de zones), D = 64 channels
        features = x.view(batch_size, 64, -1).permute(0, 2, 1) 
        
        # --- B. Attention ---
        # On calcule le score d'attention pour chaque zone L
        # Score = tanh(W_feat * features + W_hid * h_{t-1})
        
        att_features = self.att_conv(features)           # (Batch, L, D)
        att_h = self.att_hidden(h_t).unsqueeze(1)        # (Batch, 1, D)
        
        energy = torch.tanh(att_features + att_h)        # (Batch, L, D)
        attention_weights = F.softmax(self.att_out(energy), dim=1) # (Batch, L, 1)
        
        # Contexte = Somme pondérée des features (Focus)
        context = torch.sum(features * attention_weights, dim=1) # (Batch, D)
        
        # --- C. LSTM ---
        h_new, c_new = self.lstm(context, (h_t, c_t))
        
        # --- D. Dueling Heads ---
        val = self.fc_val(h_new)
        adv = self.fc_adv(h_new)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = val + adv - adv.mean(1, keepdim=True)
        
        return q_values, (h_new, c_new), attention_weights
    
    def init_hidden(self, batch_size, device):
        """Initialise l'état caché (h0, c0) à zéro"""
        return (torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device))