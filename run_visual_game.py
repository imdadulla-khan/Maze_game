import torch
from maze_env import MazeEnv
from visualizer import MazeVisualizer
from dqn_agent import Agent

def load_agent(file_path="trained_agent.pth"):
    # Load the saved state dictionary
    state_dict = torch.load(file_path)
    
    # Extract the input and output dimensions from the state dictionary
    input_dim = state_dict['network.1.weight'].shape[1]  # Input size from the first layer
    action_dim = state_dict['network.5.bias'].shape[0]   # Output size from the last layer

    # Initialize the agent with matching dimensions
    agent = Agent(state_dim=input_dim, action_dim=action_dim)
    agent.policy_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    return agent

def run_visual_game():
    # Load the trained agent
    agent = load_agent()
    
    # Initialize the environment
    env = MazeEnv(size=10, obstacle_density=0.3)  # Adjust to the final training size
    visualizer = MazeVisualizer(env)
    
    # Reset the environment and visualize
    state = env.reset()
    visualizer.draw()
    done = False

    print("AI is navigating the maze. Watch the movements!")
    while not done:
        action = agent.act(state.flatten())
        state, _, done, _ = env.step(action)
        visualizer.draw()

    print("AI has completed the maze!")
    visualizer.close()

if __name__ == "__main__":
    run_visual_game()
