import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import solve_continuous_are

# Time settings
dt = 0.005
T = 20
t = np.arange(0, T, dt)

# Target trajectory in 3D
def target_trajectory(t):
    x = t
    y = 5 * np.sin(0.2 * t) * t ** 1.1 + 2 * np.cos(0.6 * t) * t ** 1.2
    z = 3 * np.cos(0.3 * t) * t ** 1.2 + 5 * np.cos(0.8 * t) * t ** 1.3
    return np.array([x, y, z])

# State-space model: [x, y, z, vx, vy, vz]
# Inputs: [ax, ay, az]
A = np.block([
    [np.zeros((3, 3)), np.eye(3)],
    [np.zeros((3, 3)), np.zeros((3, 3))]
])
B = np.block([
    [np.zeros((3, 3))],
    [np.eye(3)]
])
D = np.eye(6, 3) * 0.05  # Direct feedthrough (small influence for realism)

C = np.eye(6)  # Full state measurement

# LQR setup
Q = np.diag([200, 200, 200, 10, 10, 10])
R = np.eye(3)
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

# Kalman filter setup
Qo = np.eye(6) * 0.1
Ro = np.eye(6) * 0.5
Po = solve_continuous_are(A.T, C.T, Qo, Ro)
L = Po @ C.T @ np.linalg.inv(Ro)

# Initial states
x_missile = np.array([0, -50, -100, 0, 0, 0], dtype = np.float64)
x_hat = x_missile.copy()

missile_path = []
estimated_path = []
target_path = []
azimuth = []
elevation = []
actuator_output = []

for ti in t:
    # Target state
    target = target_trajectory(ti)
    target_state = np.array([target[0], target[1], target[2], 0, 0, 0])
    target_path.append(target)

    # Control input
    u = -K @ (x_hat - target_state)
    actuator_output.append(u.copy())

    # Missile state update with D matrix
    x_dot = A @ x_missile + B @ u 
    x_missile += x_dot * dt

    # Measurement with D influence and noise
    y = C @ x_missile + D @ u + np.random.multivariate_normal(np.zeros(6), Ro)

    # Kalman filter update
    x_hat_dot = A @ x_hat + B @ u + L @ (y - (C @ x_hat + D @ u))
    x_hat += x_hat_dot * dt

    # Logging
    missile_path.append(x_missile.copy())
    estimated_path.append(x_hat.copy())

    vx, vy, vz = x_missile[3:6]
    az = np.degrees(np.arctan2(vy, vx))
    el = np.degrees(np.arctan2(vz, np.sqrt(vx**2 + vy**2)))
    azimuth.append(az)
    elevation.append(el)

# Convert to arrays
missile_path = np.array(missile_path)
estimated_path = np.array(estimated_path)
target_path = np.array(target_path)
azimuth = np.array(azimuth)
elevation = np.array(elevation)
actuator_output = np.array(actuator_output)

# ---------------------- PLOTTING ----------------------

# 1. 3D Trajectory
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(221, projection='3d')
ax.plot(target_path[:, 0], target_path[:, 1], target_path[:, 2], 'r--', label='Target')
ax.plot(missile_path[:, 0], missile_path[:, 1], missile_path[:, 2], 'b', label='Missile')
ax.plot(estimated_path[:, 0], estimated_path[:, 1], estimated_path[:, 2], 'g--', label='Estimated')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Missile & Target Trajectory")
ax.legend()
ax.grid()

# 2. Projections on Planes
plt.subplot(222)
plt.plot(target_path[:, 0], target_path[:, 1], 'r--', label='Target XY')
plt.plot(missile_path[:, 0], missile_path[:, 1], 'b', label='Missile XY')
plt.xlabel("X"); plt.ylabel("Y")
plt.title("XY Plane")
plt.grid(); plt.legend()

plt.subplot(223)
plt.plot(target_path[:, 1], target_path[:, 2], 'r--', label='Target YZ')
plt.plot(missile_path[:, 1], missile_path[:, 2], 'b', label='Missile YZ')
plt.xlabel("Y"); plt.ylabel("Z")
plt.title("YZ Plane")
plt.grid(); plt.legend()

plt.subplot(224)
plt.plot(target_path[:, 0], target_path[:, 2], 'r--', label='Target XZ')
plt.plot(missile_path[:, 0], missile_path[:, 2], 'b', label='Missile XZ')
plt.xlabel("X"); plt.ylabel("Z")
plt.title("XZ Plane")
plt.grid(); plt.legend()

plt.tight_layout()
plt.show()

# 3. Angles and Control Inputs
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(t, azimuth, label='Azimuth (deg)')
plt.plot(t, elevation, label='Elevation (deg)')
plt.xlabel("Time (s)")
plt.ylabel("Angle (deg)")
plt.legend(); plt.grid()
plt.title("Missile Orientation Over Time")

plt.subplot(1, 2, 2)
plt.plot(t, actuator_output[:, 0], label='ax')
plt.plot(t, actuator_output[:, 1], label='ay')
plt.plot(t, actuator_output[:, 2], label='az')
plt.xlabel("Time (s)")
plt.ylabel("Control Input (m/s²)")
plt.legend(); plt.grid()
plt.title("LQR Control Inputs")

plt.tight_layout()
plt.show()
