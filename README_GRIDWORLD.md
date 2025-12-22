# GridWorld Policy Evaluation Assignment

## Objective
This project implements **Iterative Policy Evaluation** for a 4x4 GridWorld environment.
The agent starts at the top-left (0,0) and navigates to the bottom-right (3,3).

## Problem Description
*   **Grid**: 4x4
*   **Start State**: (0, 0)
*   **Terminal State**: (3, 3) a.k.a state 15
*   **Actions**: Up, Down, Left, Right (25% probability each)
*   **Rewards**: -1 for each step, 0 at terminal.
*   **Gamma (Discount)**: 1.0

## Algorithm
The Value Function $V(s)$ is initialized to 0. We iteratively apply the Bellman Equation:

$$V_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$$

Until $\max|V_{k+1}(s) - V_k(s)| < 0.0001$.

## Final Output
The value function converged with the following values:

```
[[-59.42367735 -57.42387125 -54.2813141  -51.71012579]
 [-57.42387125 -54.56699476 -49.71029394 -45.13926711]
 [-54.2813141  -49.71029394 -40.85391609 -29.99766609]
 [-51.71012579 -45.13926711 -29.99766609   0.        ]]
```

## How to Run
1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook gridworld_assignment.ipynb
    ```
2.  Run all cells to verify the calculation.
