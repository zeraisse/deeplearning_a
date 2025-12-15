import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation, 
    ResizeObservation, 
    FrameStackObservation, 
    TransformObservation
)
import numpy as np
import torch
import ale_py

gym.register_envs(ale_py)

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        """On saute des frames pour accélérer l'entraînement. 
        L'IA joue une fois, l'action est répétée 'skip' fois."""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info

def make_env(env_name="ALE/Pacman-v5", render_mode="rgb_array"):
    """
    Configure l'environnement pour le Deep Learning.
    """
    env = gym.make(env_name, render_mode=render_mode)

    # 1. Skip Frame : On accélère le jeu (Standard Atari)
    env = SkipFrame(env, skip=4)

    # 2. Grayscale : (210, 160, 3) -> (210, 160)
    # keep_dim=False car FrameStack va rajouter la dimension des canaux
    env = GrayscaleObservation(env, keep_dim=False)

    # 3. Resize : (210, 160) -> (84, 84)
    env = ResizeObservation(env, (84, 84))

    # 4. Stack : Empile 4 frames -> (4, 84, 84)
    # C'est ici qu'on crée la notion de temps/mouvement
    env = FrameStackObservation(env, stack_size=4)

    return env