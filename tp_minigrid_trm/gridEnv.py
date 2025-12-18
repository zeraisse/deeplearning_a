import numpy as np

def get_expert_action(env):
    # On récupère l'environnement de base (sans les wrappers Gymnasium)
    # pour accéder à la grille et la position de l'agent.
    base_env = env.unwrapped
    
    agent_pos = base_env.agent_pos
    agent_dir = base_env.agent_dir
    goal_pos = None
    
    # Scan de la grille pour trouver le Goal
    grid = base_env.grid
    for x in range(base_env.width):
        for y in range(base_env.height):
            # minigrid stocke les objets dans grid.get(x, y)
            obj = grid.get(x, y)
            if obj and obj.type == 'goal':
                goal_pos = (x, y)
                break
    
    # Si pas de goal trouvé (rare), action aléatoire
    if not goal_pos: 
        return base_env.action_space.sample()

    # Calcul du vecteur direction
    dx = goal_pos[0] - agent_pos[0]
    dy = goal_pos[1] - agent_pos[1]
    
    # Détermination de l'orientation cible (0:Right, 1:Down, 2:Left, 3:Up)
    target_dir = -1
    if abs(dx) > abs(dy): 
        target_dir = 0 if dx > 0 else 2
    else: 
        target_dir = 1 if dy > 0 else 3

    # Choix de l'action (0: Left, 1: Right, 2: Forward)
    # Attention: Les indices d'action peuvent varier selon les versions, 
    # mais pour MiniGrid standard : 0=left, 1=right, 2=forward
    if agent_dir == target_dir: 
        return 2 # Forward
    elif (agent_dir + 1) % 4 == target_dir: 
        return 1 # Right
    else: 
        return 0 # Left