import numpy as np
import minigrid
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv

class gridEnv(MiniGridEnv):
    """
    Environnement personnalisé (anciennement SimpleGridEnv).
    """
    def __init__(self, size=6, **kwargs):
        self.agent_start_pos = (1, 1)
        self.agent_start_dir = 0
        mission_space = MissionSpace(mission_func=lambda: "go to goal")
        
        # Appel au constructeur parent avec le nom de la mission
        super().__init__(mission_space=mission_space, grid_size=size, max_steps=100, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.place_agent()
        self.place_obj(Goal())
        self.mission = "go to goal"

def get_expert_action(env):
    """
    L'Expert (Le Professeur) pour l'apprentissage par imitation.
    Génère le 'y_true' (label).
    """
    agent_pos = env.agent_pos
    agent_dir = env.agent_dir
    goal_pos = None
    
    # Recherche de l'objectif (triche : accès total à la grid)
    for x in range(env.width):
        for y in range(env.height):
            obj = env.grid.get(x, y)
            if obj and obj.type == 'goal':
                goal_pos = (x, y)
                break
    
    if not goal_pos: return env.action_space.sample()

    # Calcul vecteur direction
    dx = goal_pos[0] - agent_pos[0]
    dy = goal_pos[1] - agent_pos[1]
    
    # Détermination de l'orientation cible (0:Right, 1:Down, 2:Left, 3:Up)
    target_dir = -1
    if abs(dx) > abs(dy): target_dir = 0 if dx > 0 else 2
    else: target_dir = 1 if dy > 0 else 3

    # Choix action (2: Forward, 1: Right, 0: Left)
    if agent_dir == target_dir: return 2 
    elif (agent_dir + 1) % 4 == target_dir: return 1 
    else: return 0