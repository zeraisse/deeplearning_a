import numpy as np
import minigrid
# Import correct pour les versions récentes de minigrid
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal

# --- HYPERPARAMETERS ENV ---
# Tu as la main ici : change 6 en 8, 10, 16...
GRID_SIZE = 12 
MAX_STEPS = 200

class gridEnv(MiniGridEnv):
    def __init__(self, size=GRID_SIZE, **kwargs):
        self.agent_start_pos = (1, 1)
        self.agent_start_dir = 0
        mission_space = MissionSpace(mission_func=lambda: "go to goal")
        super().__init__(mission_space=mission_space, grid_size=size, max_steps=MAX_STEPS, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.place_agent()
        self.place_obj(Goal())
        self.mission = "go to goal"

def get_expert_action(env):
    """
    Expert Naïf (Ligne droite).
    Fonctionne PARFAITEMENT pour cet environnement vide.
    Ne marchera pas s'il y a des murs.
    """
    agent_pos = env.agent_pos
    agent_dir = env.agent_dir
    goal_pos = None
    
    # Scan de la grille
    for x in range(env.width):
        for y in range(env.height):
            obj = env.grid.get(x, y)
            if obj and obj.type == 'goal':
                goal_pos = (x, y)
                break
    
    if not goal_pos: return env.action_space.sample()

    dx = goal_pos[0] - agent_pos[0]
    dy = goal_pos[1] - agent_pos[1]
    
    target_dir = -1
    if abs(dx) > abs(dy): target_dir = 0 if dx > 0 else 2
    else: target_dir = 1 if dy > 0 else 3

    if agent_dir == target_dir: return 2 
    elif (agent_dir + 1) % 4 == target_dir: return 1 
    else: return 0