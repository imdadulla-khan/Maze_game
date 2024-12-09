import torch
from maze_env import MazeEnv
from train_agent import Agent
from visualizer import MazeVisualizer
import sys

def load_agent(agent, file_path="trained_agent.pth"):
    # Load the state dictionary safely
    state_dict = torch.load(file_path, weights_only=True)
    print("State Dict Keys:", state_dict.keys())  # Debug: Inspect available keys

    # Extract input dimension from the first weight matrix
    input_dim = None
    for key in state_dict.keys():
        if "weight" in key and "1" in key:  # Adjust to match the first layer naming
            input_dim = state_dict[key].shape[1]
            break

    if input_dim is None:
        raise KeyError("Input dimension could not be determined from the state dictionary.")

    # Extract output dimension from the final bias layer
    action_dim = state_dict['network.5.bias'].shape[0]  # Matches the last layer's bias size

    # Reinitialize the agent with matching dimensions
    agent = Agent(state_dim=input_dim, action_dim=action_dim)
    agent.policy_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    return agent

def challenge_user():
    env = MazeEnv(size=4, obstacle_density=0.1)  # Initial settings
    visualizer = MazeVisualizer(env)
    level = 1
    print("\nInstructions:")
    print("1. Navigate the maze using the following keys:")
    print("   - W: Move Up")
    print("   - S: Move Down")
    print("   - A: Move Left")
    print("   - D: Move Right")
    print("2. Colors in the game:")
    print("   - Yellow: Your Position")
    print("   - Green: The Goal")
    print("   - White: Empty Space")
    print("   - Black: Obstacles")
    print("\nPress Ctrl+C anytime to exit the game.")
    print("=" * 50)
    try:
        while True:
            print(f"Level {level}: Navigate the maze using WASD keys.")
            state = env.reset()
            visualizer.draw()

            done = False
            while not done:
                action = None
                while action not in {'w', 'a', 's', 'd'}:
                    action = input("Move (w/a/s/d): ").strip().lower()
                    if action not in {'w', 'a', 's', 'd'}:
                        print("Invalid move. Use WASD keys.")

                action_map = {'a': 2, 'd': 3, 'w': 0, 's': 1}  # Updated directions
                _, _, done, _ = env.step(action_map[action])
                visualizer.draw()

            print(f"Congratulations! You cleared Level {level}.")
            level += 1
            env.increase_difficulty()  # Increase difficulty for the next level

    except KeyboardInterrupt:
        print("\nGame interrupted by user. Exiting...")
    finally:
        visualizer.close()
        print("Thank you for playing! Goodbye.")
        sys.exit(0)

if __name__ == "__main__":
    agent = load_agent(None)
    challenge_user()
