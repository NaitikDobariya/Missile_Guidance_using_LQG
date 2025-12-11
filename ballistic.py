import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

# Time settings
dt = 0.01
T = 20
t = np.arange(0, T, dt)

# Target trajectory (sinusoidal path)
def target_trajectory(t):
    x = t
    y = 5 * np.sin(0.2 * t)
    return np.array([x, y])

# Missile dynamics: [x, y, vx, vy]
# Missile input: acceleration in x and y: [ax, ay]
A = np.array([[0, 0, 1, 0],
              [0, 0, 0.09, 1.2],
              [0, 0, 0, 0.2],
              [0, 0, 0, 0]])
B = np.array([[0, 0],
              [0, 0],
              [1, 0],
              [0, 1]])
C = np.eye(4)

# LQR setup
Q = np.diag([100, 100, 10, 10])
R = np.diag([1, 1])
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

# Kalman filter setup
Qo = np.eye(4) * 0.1
Ro = np.eye(4) * 0.5
Po = solve_continuous_are(A.T, C.T, Qo, Ro)
L = Po @ C.T @ np.linalg.inv(Ro)

# Initial states
x_missile = np.array([0.0, 0.0, 0.0, 0.0])
x_hat = x_missile.copy()

missile_path = []
estimated_path = []
target_path = []
path_angle = []
actuator_output = []

for ti in t:
    # Target state
    target = target_trajectory(ti)
    target_state = np.array([target[0], target[1], 0, 0])
    target_path.append(target)

    # Control input from estimated state
    u = -K @ (x_hat - target_state)
    actuator_output.append(u.copy())

    # True missile state update
    x_dot = A @ x_missile + B @ u
    x_missile += x_dot * dt

    # Measurement with noise
    y = C @ x_missile + np.random.multivariate_normal(np.zeros(4), Ro)

    # Kalman filter update
    x_hat_dot = A @ x_hat + B @ u + L @ (y - C @ x_hat)
    x_hat += x_hat_dot * dt

    # Logs
    missile_path.append(x_missile.copy())
    estimated_path.append(x_hat.copy())
    angle = np.arctan2(x_missile[3], x_missile[2])
    path_angle.append(np.degrees(angle))

# Convert to arrays
missile_path = np.array(missile_path)
estimated_path = np.array(estimated_path)
target_path = np.array(target_path)
path_angle = np.array(path_angle)
actuator_output = np.array(actuator_output)

# ----------------- Plotting -----------------

plt.figure(figsize=(15, 5))

# Trajectories
plt.subplot(1, 3, 1)
plt.plot(target_path[:, 0], target_path[:, 1], 'r--', label='Target')
plt.plot(missile_path[:, 0], missile_path[:, 1], 'b', label='True Missile')
plt.plot(estimated_path[:, 0], estimated_path[:, 1], 'g--', label='Estimated Missile')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.title("Target vs True vs Estimated Missile Trajectory")

# Path angle
plt.subplot(1, 3, 2)
plt.plot(t, path_angle, 'm')
plt.xlabel("Time (s)")
plt.ylabel("Missile Angle (deg)")
plt.grid()
plt.title("Missile Heading Angle Over Time")

# Control input
plt.subplot(1, 3, 3)
plt.plot(t, actuator_output[:, 0], label="ax (x dir)")
plt.plot(t, actuator_output[:, 1], label="ay (y dir)")
plt.xlabel("Time (s)")
plt.ylabel("Control Input (m/s²)")
plt.legend()
plt.grid()
plt.title("LQR Control Inputs")

plt.tight_layout()
plt.show()
