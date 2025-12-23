# NeuralNav: Autonomous Car Navigation using DQN

## 1. Project Objective
The objective of this project is to implement and train a self-driving car agent to navigate a city map using Deep Q-Learning (DQN). The car must learn to:
1.  Navigate from a start point to multiple sequentially generated target points.
2.  Avoid obstacles (white areas on the map).
3.  Optimize its path for efficiency.

The core challenge was to tune a set of initially incorrect "broken" parameters (physics, sensor settings, and reinforcement learning hyperparameters) to enable stable and effective learning.

## 2. Key Reinforcement Learning Concepts

*   **Deep Q-Network (DQN)**: A neural network that approximates the Q-value function, predicting the expected future reward for taking a specific action in a given state.
*   **State Space**: The car's inputs, which include:
    *   7 Distance Sensor readings (detecting obstacles).
    *   Angle to the target.
    *   Distance to the target.
*   **Action Space**: The possible moves the car can make: Left, Right, Straight, Sharp Left, Sharp Right.
*   **Reward Function**: A mechanism to guide learning:
    *   **Positive Reward (+100)**: Reaching a target.
    *   **Negative Reward (-100)**: Crashing into an obstacle.
    *   **Shaping Reward**: Small rewards/penalties for moving closer to or further from the target.
*   **epsilon-Greedy Strategy**: Balancing **exploration** (random actions) and **exploitation** (using the best known action). Epsilon starts at 1.0 and decays over time.
*   **Experience Replay**: Storing past experiences (State, Action, Reward, Next State) in a memory buffer and sampling them randomly to train the network. This breaks correlations between consecutive samples and stabilizes training.

## 3. Project Structure

*   **`citymap_assignment-v2.py`**: The main application file containing all components:
    *   **`CarBrain` Class**: Manages the physics, RL logic, memory buffers, and training loops.
    *   **`DrivingDQN` Class**: The PyTorch neural network architecture (Multi-layer Perceptron).
    *   **`NeuralNavApp` Class**: The PyQt6 GUI that renders the map, car, and charts.
    *   **Physics Engine**: Handles movement, collision detection, and sensor calculations.

## 4. DQN Architecture (V2 Evolution)

The V2 simulation features an upgraded **DrivingDQN** architecture. While the original model used a simpler MLP, the V2 model incorporates an additional deeper layer to handle the complexities of the Parisian radial map and cyclic navigation.

### Network Layout
- **Input Layer**: 9 nodes (7 sensors + distance + angle).
- **Hidden Layer 1**: 128 units, ReLU activation.
- **Hidden Layer 2**: 256 units, ReLU activation.
- **Deeper Layer (New)**: 256 units, ReLU activation — *Added in V2 to increase representational capacity for complex city intersections.*
- **Hidden Layer 3**: 256 units, ReLU activation.
- **Hidden Layer 4**: 128 units, ReLU activation.
- **Output Layer**: 5 nodes (Left, Right, Straight, Sharp Left, Sharp Right).

### Why the Deeper Net?
The addition of the extra 256-unit layer allows the network to capture higher-level spatial relationships between the 7 sensors and the cyclic target coordinates. This is particularly effective for the "Paris" map where roads radiate at different angles, requiring more nuanced steering decisions than a standard grid.

## 5. Hyperparameter Fixes

The following parameters were identified as incorrect and have been fixed to enable proper learning:

| Parameter | Type | Fixed Value | Reason for Choice |
| :--- | :--- | :--- | :--- |
| **SENSOR_DIST** | Physics | `15` | Short-range sensors allow for immediate obstacle detection near walls. |
| **SENSOR_ANGLE**| Physics | `15` | A 15-degree spread provides focused frontal coverage (approx 90° total FOV). |
| **SPEED** | Physics | `5` | Lower speed (5 px/step) prevents overshooting and allows enough time for the agent to react. |
| **TURN_SPEED** | Physics | `5` | Matches the speed to create a tighter turning radius (~1.0), essential for city corners. |
| **SHARP_TURN** | Physics | `20` | Allows for effective evasive maneuvers when stuck or facing a sharp corner. |
| **BATCH_SIZE** | RL | `256` | A larger batch size stabilizes gradient updates by averaging more examples. |
| **GAMMA** | RL | `0.98` | High discount factor makes the agent value long-term rewards (reaching the target) over immediate ones. |
| **LR** | RL | `0.0005` | Moderate learning rate (5e-4) avoids unstable updates while learning fast enough. |
| **TAU** | RL | `0.005` | Soft update rate for the target network ensures stable learning without drastic shifts. |
| **EPSILON** | RL | `1.0` | Starts at 100% exploration to ensure the agent discovers the map mechanics before exploiting. |

## 6. How to Run

1.  Ensure you have the required dependencies installed:
    ```bash
    pip install numpy torch PyQt6
    ```
2.  Run the application:
    ```bash
    python "citymap_assignment - v2.py"
    ```
3. Load either one of the maps provided - paris_citymap1, paris_citymap2, mumbaicitymap3
   
4.  **In the App**:
    *   **Left Click** on the map to place the **Car**.
    *   **Left Click** again (multiple times) to place **Targets**.
    *   **Right Click** to finish setup.
    *   Press **SPACE** or click **START** to begin training.

## 7. Demo

<img width="1914" height="1137" alt="image" src="https://github.com/user-attachments/assets/7b24d39b-53cb-43c2-b145-fe004b2968b2" />

https://youtu.be/Q16biHWqxig




