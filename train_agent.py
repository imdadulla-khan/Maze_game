from visualizer import MazeVisualizer
from maze_env import MazeEnv
from dqn_agent import Agent
import torch
import time

def increase_difficulty(self):
    """Increase the difficulty by adjusting the size and obstacle density."""
    self.size = min(self.size + 1, 10)  # Limit the size to a maximum of 10x10
    self.obstacle_density = min(self.obstacle_density + 0.05, 0.3)  # Cap density at 0.3
    self.reset()  # Reinitialize the environment with the new parameters

def train_agent():
    initial_size = 4
    initial_density = 0.1
    max_size = 10
    max_density = 0.3
    episodes_per_level = 50
    total_episodes = 300
    agent = Agent(state_dim=initial_size * initial_size, action_dim=4)

    env = MazeEnv(size=initial_size, obstacle_density=initial_density)
    visualizer = MazeVisualizer(env)

    current_level = 1
    episode_count = 0


    for episode in range(total_episodes):
        # Reset the environment
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.act(state.flatten())
            next_state, reward, done, _ = env.step(action)
            agent.store((state.flatten(), action, reward, next_state.flatten(), done))
            agent.train()
            state = next_state
            total_reward += reward
            steps += 1

            visualizer.draw()  # Visualize during training

            if steps >= env.size ** 2 * 2:  # Limit steps to twice the grid size
                break

        agent.update_target_network()
        episode_count += 1
        print(f"Episode {episode}: Total Reward: {total_reward}, Level: {current_level}")

        # Increase difficulty every 'episodes_per_level'
        if episode_count >= episodes_per_level:
            env.increase_difficulty()  # Dynamically increase difficulty
            agent.reinitialize_networks(env.size * env.size)
            agent.clear_memory()  # Clear memory to avoid shape mismatch

            episode_count = 0
            current_level += 1
            env.reset()  # Reinitialize environment with new difficulty

    visualizer.close()
    return agent



def save_agent(agent, file_path="trained_agent.pth"):
    torch.save(agent.policy_net.state_dict(), file_path)

if __name__ == "__main__":
    agent = train_agent()
    save_agent(agent)
