import numpy as np
import matplotlib.pyplot as plt

M = 1.0  # Mass of the black hole
# In these units, the Event Horizon (Schwarzschild Radius) is at r = 2*M = 2.0

def geodesics(state):
    """
    r - distance between the center of the black hole and the object
    phi - angle used for the polar coordinat form
    vr - velocity 
    vphi - angular velocity
    """
    r, phi, vr, vphi = state
    
    # If the particle falls into the black hole, stop calculating to avoid dividing by zero
    if r <= 2.0 * M:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    #velocity is the first derivative
    dr = vr
    dphi = vphi
    
    # Schwarzschild Geodesic Equation
    dvr = -M / r**2 + r * vphi**2 - 3.0 * M * vphi**2
    
    # Angular Acceleration
    dvphi = -2.0 * vr * vphi / r
    
    return np.array([dr, dphi, dvr, dvphi])

# The RK4 Integrator Core 
def rk4_step(state, dt):
    k1 = geodesics(state)
    k2 = geodesics(state + 0.5 * dt * k1)
    k3 = geodesics(state + 0.5 * dt * k2)
    k4 = geodesics(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

r_init = 8.43 
vphi_init = np.sqrt(M / (r_init**3 - 3.0 * M * r_init**2))
initial_state = np.array([r_init, 0.0, 0.0, vphi_init])

dt = 0.05
total_steps = 12000

history_r = []
history_phi = []

current_state = initial_state
for _ in range(total_steps):
    history_r.append(current_state[0])
    history_phi.append(current_state[1])
    
    current_state = rk4_step(current_state, dt)
    if current_state[0] <= 2.0 * M:
        print("The particle was swallowed by the black hole!")
        break

# Convert Polar Coordinates back to Cartesian
history_r = np.array(history_r)
history_phi = np.array(history_phi)
positions_x = history_r * np.cos(history_phi)
positions_y = history_r * np.sin(history_phi)

plt.figure(figsize=(7, 7))
plt.plot(positions_x, positions_y, label="Relativistic Orbit", color="magenta")

horizon = plt.Circle((0, 0), 2.0*M, color='black', label="Event Horizon (r=2M)")
plt.gca().add_patch(horizon)

plt.title("Step 2: Curved-Spacetime Schwarzschild Geodesic")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.axis("equal")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.show()
