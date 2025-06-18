import numpy as np
import matplotlib.pyplot as plt

ph = 0.4

state_space = np.arange(1, 100)

goal = 100

threshold = 10**(-11)

v = np.zeros(goal+1) #array of v(s) for all s
v[goal] = 1

policy = np.zeros(goal+1, dtype=int)

#value iteration (combining evlaution and improvement)
while True:
    delta = 0
    for s in state_space:
        max_s = min(s,goal-s)
        action_space = np.arange(0, max_s+1)
        action_returns = np.zeros_like(action_space, dtype=float) 
        v_old = v[s]
        for a in action_space:
            action_returns[a] = ph*v[s+a]+(1-ph)*v[s-a]
        v[s] = np.max(action_returns)
        best_action_index = np.argmax(action_returns)
        policy[s] = action_space[best_action_index]
        delta = max(delta,abs(v[s]-v_old))
    if (delta < threshold):
        break

print(policy)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Optimal Policy
ax1.plot(state_space, policy[state_space], marker='o', linestyle='-', color='blue')
ax1.set_title('Optimal Policy (Stake) vs State (Capital)')
ax1.set_xlabel('Capital (State s)')
ax1.set_ylabel('Optimal Stake (Action a)')
ax1.grid(True)

# Plot Value Function
ax2.plot(range(goal + 1), v, linestyle='-', color='green')
ax2.set_title('Value Function vs State')
ax2.set_xlabel('Capital (State s)')
ax2.set_ylabel('Value v(s)')
ax2.grid(True)

plt.tight_layout()
plt.show()