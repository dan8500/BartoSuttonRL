import numpy as np
import matplotlib.pyplot as plt

ph = 0.4

goal = 127

state_space = np.arange(1, goal)


threshold = 10**(-8)

v = np.zeros(goal+1) #array of v(s) for all s
v[goal] = 1

policy = np.zeros(goal+1, dtype=int)

#value iteration (combining evlaution and improvement)
while True:
    delta = 0
    for s in state_space:
        max_s = min(s,goal-s)
        action_space = np.arange(1, max_s+1) #don't need 0 because it doesn't do anything
        action_returns = np.zeros_like(action_space, dtype=float) 
        v_old = v[s]
        for idx, a in enumerate(action_space):
            #reward = 1 if s + a == goal else 0
            #action_returns[idx] = ph*(reward+v[s+a])+(1-ph)*v[s-a]
            if s + a == goal:
                action_returns[idx] = ph * 1 + (1 - ph) * v[s - a]  # no v[goal] added!
            else:
                action_returns[idx] = ph * v[s + a] + (1 - ph) * v[s - a]
        best_action_index = np.argmax(action_returns)
        best_actions = np.flatnonzero(action_returns == np.max(action_returns))
        #policy[s] = int(np.mean(action_space[best_actions]))
        policy[s] = action_space[best_action_index]
        v[s] = action_returns[best_action_index]
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