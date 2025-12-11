import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.signal import place_poles

# Missile inner-loop dynamics (AoA and pitch rate)
A_missile = np.array([
    [-1.064, 1.000],
    [290.26, 0.00]
])
B_missile = np.array([
    [-0.25],
    [-331.40]
])

# Only AoA is measured
C_missile = np.array([
    [-123.34, 0.0],
    [0.0, 1.0]
])
D_missile = np.array([
    [-13.41],
    [0.0]
])

# LQR Design
Q_missile = np.diag([10.0, 1.0])
R_missile = np.array([[2.0]])
P_missile = solve_continuous_are(A_missile, B_missile, Q_missile, R_missile)
K_missile = np.linalg.inv(R_missile) @ B_missile.T @ P_missile

# Kalman filter design (optimal observer)
Q_kalman = np.diag([0.5, 0.5])   # Process noise covariance
R_kalman = np.diag([0.1, 0.1])   # Measurement noise covariance

P_kalman = solve_continuous_are(A_missile.T, C_missile.T, Q_kalman, R_kalman)
L_missile = P_kalman @ C_missile.T @ np.linalg.inv(R_kalman)


# Simulation parameters
dt = 0.01
T = 500
t = np.arange(0, T, dt)

# Target trajectory
def target_trajectory(t):
    x = t
    y = 5 * np.sin(0.04 * t) + 2 * np.cos(0.06 * t)
    return np.array([x, y])

# Initialization
x_pos = np.array([0.0, -5.0])
v = 1.0
flight_path_angle = np.radians(20.0)

x_true = np.array([0.0, 0.0])  # [alpha, q]
x_hat = np.array([0.0, 0.0])   # Observer estimate

# Logs
path, target_path = [], []
delta_c_list, flight_path_angles, az_error = [], [], []

for i, ti in enumerate(t):
    # Target
    lookahead = 3.0
    future_time = min(ti + lookahead, T)
    target = target_trajectory(future_time)
    target_now = target_trajectory(ti)

    # LOS angle
    delta = target - x_pos
    theta_ref = np.arctan2(delta[1], delta[0])

    # Current FPA (from velocity)
    vx = v * np.cos(flight_path_angle)
    vy = v * np.sin(flight_path_angle)
    current_fpa = np.arctan2(vy, vx)

    # Measured AoA and pitch
    alpha = x_true[0]
    theta = current_fpa + alpha

    # Angle error (LOS - FPA)
    angle_error = theta_ref - current_fpa
    az_error.append(np.degrees(angle_error))

    # Measurement: only alpha
    y_meas = C_missile @ x_true

    # Compute control input (LQR using estimated state)
    ref_state = np.array([angle_error, 0.0])
    u_raw = -K_missile @ (x_hat - ref_state)

    # Smooth saturation
    delta_c = 25 * np.pi / 180 * np.tanh(u_raw / (25 * np.pi / 180))
    delta_c_val = delta_c.item()
    delta_c_list.append(np.degrees(delta_c_val))

    # Observer update
    x_hat_dot = A_missile @ x_hat + B_missile.flatten() * delta_c_val + \
                L_missile @ (y_meas - C_missile @ x_hat)
    x_hat += x_hat_dot * dt

    # True dynamics update
    x_dot_true = A_missile @ x_true + B_missile.flatten() * delta_c_val
    x_true += x_dot_true * dt

    # Update flight angle
    flight_path_angle = current_fpa + x_true[0]

    # Update position
    dx = v * np.cos(flight_path_angle) * dt
    dy = v * np.sin(flight_path_angle) * dt
    x_pos += np.array([dx, dy])

    # Log data
    path.append(x_pos.copy())
    target_path.append(target_now.copy())
    flight_path_angles.append(np.degrees(flight_path_angle))

# Convert logs to arrays
path = np.array(path)
target_path = np.array(target_path)
flight_path_angles = np.array(flight_path_angles)
delta_c_list = np.array(delta_c_list)
az_error = np.array(az_error)

# Plotting
plt.figure(figsize=(15, 10))

# 1. Trajectory
plt.subplot(2, 2, 1)
plt.plot(target_path[:, 0], target_path[:, 1], 'r--', label='Target')
plt.plot(path[:, 0], path[:, 1], 'b', label='Missile')
plt.xlabel("X (m)"); plt.ylabel("Y (m)")
plt.title("Missile vs Target Trajectory (2D)")
plt.legend(); plt.grid()

# 2. Flight Path Angle
plt.subplot(2, 2, 2)
plt.plot(t, flight_path_angles, label='Flight Path Angle (deg)')
plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
plt.title("Flight Path Angle over Time")
plt.legend(); plt.grid()

# 3. Control Surface Deflection
plt.subplot(2, 2, 3)
plt.plot(t, delta_c_list, label='Control Input δ_c (deg)')
plt.xlabel("Time (s)"); plt.ylabel("δ_c (deg)")
plt.title("Missile Control Surface Deflection")
plt.legend(); plt.grid()

# 4. LOS vs FPA Error
plt.subplot(2, 2, 4)
plt.plot(t, az_error, label='LOS vs FPA Error (deg)')
plt.xlabel("Time (s)"); plt.ylabel("Angle Error (deg)")
plt.title("Line-of-Sight vs Flight Path Angle Error")
plt.legend(); plt.grid()

plt.tight_layout()
plt.show()
