import numpy as np
import matplotlib.pyplot as plt

G = 1.0
M = 1.0

def derivatives(state):
    #x,y coords and their velocities
    x, y, vx, vy = state
    #distance
    r = np.sqrt(x**2 + y**2)

    #cool formulas
    ax = -G * M * x / r**3
    ay = -G * M * y / r**3

    return np.array([vx, vy, ax, ay])

def rk4_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)

    #takes the average step in which to make the small change rather than treating it like a stright line(gravity changes stuff)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

initial_state = np.array([1.0, 0.1, 0, 1])

dt = 0.01
total_steps = 2000

positions_x = []
positions_y = []

current_state = initial_state
for _ in range(total_steps):
    positions_x.append(current_state[0])
    positions_y.append(current_state[1])

    current_state = rk4_step(current_state, dt)

plt.figure(figsize=(6, 6))
plt.plot(positions_x, positions_y, label="Particle Orbit", color="cyan")
plt.scatter(0, 0, color="orange", s=200, label="Star (M)")  # Central body
plt.title("Step 1: Flat-Space Newtonian Orbit (RK4)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.axis("equal")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()