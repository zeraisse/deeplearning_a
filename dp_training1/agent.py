import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from model import DARQN

class RecurrentReplayBuffer:
    def __init__(self, capacity, seq_len=8):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Pour le DARQN, on ne sample pas des images isolées, mais des SÉQUENCES temporelles.
        C'est complexe : on doit prendre des morceaux consécutifs dans la mémoire.
        """
        # Simplification robuste : On prend des indices aléatoires valides
        # et on reconstruit la séquence passée
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []
        
        # On doit s'assurer qu'on a assez de données
        if len(self.buffer) < self.seq_len:
            return None

        for _ in range(batch_size):
            # On choisit un point de fin aléatoire
            end_idx = np.random.randint(self.seq_len, len(self.buffer))
            # On récupère la séquence de longueur seq_len finissant à end_idx
            # Note: Dans une implémentation parfaite, on vérifierait que 'done' n'est pas au milieu
            # Ici on simplifie pour l'apprentissage
            seq = list(self.buffer)[end_idx-self.seq_len:end_idx]
            
            s, a, r, ns, d = zip(*seq)
            state_batch.append(np.array(s))
            action_batch.append(np.array(a))
            reward_batch.append(np.array(r))
            next_state_batch.append(np.array(ns))
            done_batch.append(np.array(d))

        return (
            torch.tensor(np.array(state_batch), dtype=torch.float32),      # (Batch, Seq, C, H, W)
            torch.tensor(np.array(action_batch), dtype=torch.long),        # (Batch, Seq)
            torch.tensor(np.array(reward_batch), dtype=torch.float32),     # (Batch, Seq)
            torch.tensor(np.array(next_state_batch), dtype=torch.float32), # (Batch, Seq, C, H, W)
            torch.tensor(np.array(done_batch), dtype=torch.float32)        # (Batch, Seq)
        )

    def __len__(self):
        return len(self.buffer)

class DARQNAgent:
    def __init__(self, input_shape, num_actions, device, lr=1e-4, batch_size=128):
        self.batch_size = batch_size
        self.device = device
        self.num_actions = num_actions
        
        
        # Réseaux (Policy + Target pour la stabilité)
        self.policy_net = DARQN(input_shape, num_actions).to(device)
        self.target_net = DARQN(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimiseur
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Mémoire
        self.memory = RecurrentReplayBuffer(capacity=50000, seq_len=4) # Seq courte pour commencer
        
        # Hyperparamètres
        self.gamma = 0.99
        
        # Compilation (Optionnel: Boost RTX 5070 Ti)
        
        # print("🔥 Compilation du modèle (Blackwell Mode)...")
        # self.policy_net = torch.compile(self.policy_net)
        # self.target_net = torch.compile(self.target_net)
        
    def select_action(self, state, hidden, epsilon):
        # state shape: (C, H, W) -> besoin de (1, C, H, W) pour le modèle
        state_t = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        
        # --- BRANCHE 1 : EXPLOITATION (INTELLIGENTE) ---
        if random.random() > epsilon:
            with torch.no_grad():
                # Mode Turbo (BF16) activé
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    q_values, next_hidden, _ = self.policy_net(state_t, hidden)
                return q_values.argmax().item(), next_hidden
        
        # --- BRANCHE 2 : EXPLORATION (ALÉATOIRE) ---
        else:
            # Action aléatoire, mais on doit calculer le hidden state suivant !
            with torch.no_grad():
                # CORRECTION : On active AUSSI le BF16 ici pour gérer les formats entrants
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                     _, next_hidden, _ = self.policy_net(state_t, hidden)
            
            return random.randrange(self.num_actions), next_hidden
    def train_step(self):
        if len(self.memory) < 1000: return None # Warmup

        batch = self.memory.sample(self.batch_size)
        if not batch: return None
        
        states, actions, rewards, next_states, dones = [b.to(self.device) for b in batch]
        # states shape: (Batch, Seq, C, H, W)
        
        # Initialisation hidden states à 0 pour le début de séquence (Simplification)
        hidden = self.policy_net.init_hidden(self.batch_size, self.device)
        target_hidden = self.target_net.init_hidden(self.batch_size, self.device)
        
        loss = 0
        
        # Optimisation BLACKWELL : Mixed Precision (BF16)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # On itère sur la séquence temporelle
            # Note: C'est une boucle, donc lent en Python, mais indispensable pour le LSTM
            for t in range(self.memory.seq_len):
                # 1. Q-Values actuelles
                q_values, hidden, _ = self.policy_net(states[:, t], hidden)
                state_action_values = q_values.gather(1, actions[:, t].unsqueeze(1))
                
                # 2. Target Q-Values (Double DQN logic possible ici)
                with torch.no_grad():
                    next_q_values, target_hidden, _ = self.target_net(next_states[:, t], target_hidden)
                    next_max_q = next_q_values.max(1)[0].detach()
                    expected_q_values = rewards[:, t] + (self.gamma * next_max_q * (1 - dones[:, t]))
                
                # 3. Accumulation de la loss sur la séquence
                loss += F.smooth_l1_loss(state_action_values, expected_q_values.unsqueeze(1))
        
        # Backpropagation
        self.optimizer.zero_grad()
        # Scaler non nécessaire pour bfloat16 (native range), mais bonne pratique
        loss.backward()
        # Gradient clipping pour éviter l'explosion dans le LSTM
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()