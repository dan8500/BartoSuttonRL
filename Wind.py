import numpy as np
import matplotlib.pyplot as plt

wind_grid = np.zeros((7, 10), dtype=int)

cols_one = [3, 4, 5, 8]
wind_grid[:, 6:8] = -2
wind_grid[:, cols_one] = -1

s = [3,0] #starting state
goal = [3,9]

moves = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1), "NE": (-1, 1), "SE": (1, 1), "SW": (1, -1), "NW": (-1, -1)} #in python we count from the upper left corner of the array, so to move north you decrease the column index
move_keys = list(moves.keys())

Q = np.zeros((len(wind_grid), len(wind_grid[0]), len(moves))) #numpy breaks first dimension into slices, fyi

epsilon = 1 #0.1, changed to 1 for testing

rewards = np.ones_like(wind_grid) *-1
rewards[*goal] = 0

#I will start by making a loop for just one episode
while s != goal:
    chooser = np.random.rand()
    if chooser < epsilon:
        a = np.random.choice(move_keys)
    else:
        q_values_to_select_from = Q[s[0], s[1], :]
        best_action_index = np.argmax(q_values_to_select_from)
        a = move_keys[best_action_index]

    print("Chosen action:", a)

    rows, cols = wind_grid.shape

    # Current position
    row, col = s

    # Move delta for action 'a'
    delta_row, delta_col = moves[a]

    # New position after move
    new_row_without_wind = row + delta_row 
    new_col = col + delta_col

    safe_row = max(0, min(rows - 1, new_row_without_wind))
    safe_col = max(0, min(cols - 1, new_col))

    wind = wind_grid[safe_row, safe_col]

    new_row = new_row_without_wind + wind

    new_row = max(0, min(rows - 1, new_row))
    new_col = max(0, min(cols - 1, new_col))

    # Keep new position inside the grid boundaries
    new_row = max(0, min(rows - 1, new_row))
    new_col = max(0, min(cols - 1, new_col))

    # Update state
    s = [int(new_row), int(new_col)]

    print("New coordinate:", s)