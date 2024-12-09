import gym
from gym import spaces
import numpy as np
import random

class MazeEnv(gym.Env):
    def __init__(self, size=4, obstacle_density=0.1):
        super(MazeEnv, self).__init__()
        self.size = size
        self.obstacle_density = obstacle_density
        self.action_space = spaces.Discrete(4)  # [Up, Down, Left, Right]
        self.observation_space = spaces.Box(low=0, high=2, shape=(size, size), dtype=np.int32)
        self.reset()

    def reset(self):
        self.maze = np.zeros((self.size, self.size), dtype=int)
        self.start = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        self.goal = self.start
        while self.goal == self.start:
            self.goal = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        self.agent_pos = list(self.start)
        self._add_obstacles()

        # Set initial positions for agent and goal
        self.maze[self.agent_pos[0], self.agent_pos[1]] = 1
        self.maze[self.goal] = 2  # Goal
        return self._get_state()

    def step(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # [Up, Down, Left, Right]
  # [Up, Down, Left, Right]
        new_pos = [self.agent_pos[0] + moves[action][0], self.agent_pos[1] + moves[action][1]]

        # Update agent position if within bounds and not on an obstacle
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size and self.maze[new_pos[0], new_pos[1]] != -1:
            # Clear the old position
            self.maze[self.agent_pos[0], self.agent_pos[1]] = 0
            # Move to the new position
            self.agent_pos = new_pos
            # Mark the new position as the agent's
            self.maze[self.agent_pos[0], self.agent_pos[1]] = 1

        done = self.agent_pos == list(self.goal)
        reward = 10 if done else -0.01
        return self._get_state(), reward, done, {}

    def _add_obstacles(self):
        num_obstacles = int(self.size * self.size * self.obstacle_density)
        for _ in range(num_obstacles):
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if (x, y) != self.start and (x, y) != self.goal:
                self.maze[x, y] = -1  # Obstacle

    def _get_state(self):
        state = np.zeros_like(self.maze)
        state[self.agent_pos[0], self.agent_pos[1]] = 1
        return state

    def increase_difficulty(self):
        """Increase the difficulty by adjusting the size and obstacle density."""
        self.size = min(self.size + 1, 10)  # Limit the size to a maximum of 10x10
        self.obstacle_density = min(self.obstacle_density + 0.05, 0.3)  # Cap density at 0.3
        self.reset()  # Reinitialize the environment with the new parameters


    def render(self, mode='human'):
        print(self.maze)
        print(f"Agent position: {self.agent_pos}")
