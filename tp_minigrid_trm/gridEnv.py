import numpy as np
import collections
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Door, Wall

from config import DIR_TO_VEC, GRID_SIZE, MAX_STEPS

class gridEnv(MiniGridEnv):
    def __init__(self, size=GRID_SIZE, **kwargs):
        self.agent_start_pos = (1, 1)
        self.agent_start_dir = 0
        mission_space = MissionSpace(mission_func=lambda: "get key, open door, go to goal")
        super().__init__(mission_space=mission_space, grid_size=size, max_steps=MAX_STEPS, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        # 1. Le Mur Vertical
        splitIdx = width // 2
        for i in range(0, height):
            self.grid.set(splitIdx, i, Wall())
        
        # 2. La Porte Jaune
        doorIdx = height // 2
        self.grid.set(splitIdx, doorIdx, Door('yellow', is_locked=True))
        
        # 3. Le Goal (À droite)
        self.place_obj(Goal(), top=(splitIdx + 1, 0), size=(width - splitIdx - 1, height))
        
        # 4. L'Agent
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        
        # 5. La Clé (Tout en bas à gauche)
        self.place_obj(Key('yellow'), top=(1, height - 3), size=(splitIdx - 1, 2))
        
        self.mission = "get key, open door, go to goal"

# --- EXPERT INTELLIGENT ---
def bfs_path(env, start_pos, start_dir, target_pos):
    queue = collections.deque([(start_pos[0], start_pos[1], start_dir, [])])
    visited = set([(start_pos[0], start_pos[1], start_dir)])
    
    while queue:
        x, y, d, actions = queue.popleft()
        
        if (x, y) == target_pos:
            return actions
            
        possible_moves = [(0, 'left'), (1, 'right'), (2, 'fwd')]
        
        for action, name in possible_moves:
            nx, ny, nd = x, y, d
            
            if action == 0: nd = (d - 1) % 4
            elif action == 1: nd = (d + 1) % 4
            elif action == 2:
                dx, dy = DIR_TO_VEC[d]
                nx, ny = x + dx, y + dy

            if nx < 0 or nx >= env.width or ny < 0 or ny >= env.height: continue

            cell = env.grid.get(nx, ny)
            is_obstacle = (cell is not None and cell.type == 'wall')
            
            if cell and cell.type == 'door' and cell.is_locked:
                if (nx, ny) != target_pos: is_obstacle = True

            if not is_obstacle and (nx, ny, nd) not in visited:
                visited.add((nx, ny, nd))
                new_actions = actions + [action]
                queue.append((nx, ny, nd, new_actions))
    return None

def get_expert_action(env):
    base_env = env.unwrapped
    agent_pos = base_env.agent_pos
    agent_dir = base_env.agent_dir
    carrying = base_env.carrying
    
    key_pos = None
    door_pos = None
    goal_pos = None
    door_is_open = False
    
    for x in range(base_env.width):
        for y in range(base_env.height):
            obj = base_env.grid.get(x, y)
            if obj:
                if obj.type == 'key': key_pos = (x, y)
                elif obj.type == 'door': 
                    door_pos = (x, y)
                    door_is_open = obj.is_open
                elif obj.type == 'goal': goal_pos = (x, y)

    target = None
    action_at_target = None
    
    if not carrying:
        target, action_at_target = key_pos, 3 # Pickup
    elif carrying and not door_is_open:
        target, action_at_target = door_pos, 5 # Toggle
    else:
        target, action_at_target = goal_pos, 2 # Walk

    if not target: return base_env.action_space.sample()

    dx, dy = DIR_TO_VEC[agent_dir]
    front_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
    
    if front_pos == target and action_at_target in [3, 5]:
        return action_at_target

    actions = bfs_path(base_env, agent_pos, agent_dir, target)
    if actions and len(actions) > 0:
        if len(actions) == 1 and action_at_target in [3, 5] and actions[0] == 2:
             return action_at_target
        return actions[0]
    
    return base_env.action_space.sample()