import tkinter as tk
import sys

class MazeVisualizer:
    def __init__(self, env):
        self.env = env
        self.cell_size = 50  # Size of each cell in pixels
        self.root = tk.Tk()
        self.root.title("Maze Game")
        self.canvas = tk.Canvas(self.root, width=env.size * self.cell_size, height=env.size * self.cell_size)
        self.canvas.pack()
        self.colors = {
            0: "white",   # Free space
            -1: "black",  # Obstacle
            1: "yellow",  # Agent
            2: "green"    # Goal
        }
        self.root.protocol("WM_DELETE_WINDOW", self.close)  # Bind close button to cleanup

    def resize_canvas(self):
        """Resize the canvas to match the new maze size."""
        self.canvas.config(width=self.env.size * self.cell_size, height=self.env.size * self.cell_size)

    def draw(self):
        """Draw the maze grid with the current positions of the agent and the goal."""
        self.resize_canvas()  # Adjust canvas size if maze size has changed
        self.canvas.delete("all")
        for row in range(self.env.size):
            for col in range(self.env.size):
                # Get the value of the current cell and determine its color
                cell_value = self.env.maze[row, col]
                color = self.colors[cell_value]
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                # Draw the cell as a rectangle
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
        self.root.update()

    def close(self):
        """Close the Tkinter window."""
        print("Closing the game...")
        self.root.destroy()
        sys.exit(0)