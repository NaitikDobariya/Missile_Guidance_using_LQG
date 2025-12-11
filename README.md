# Missile Guidance & Navigation Control (GNC) System

Video

[![GNC Missile Demo](https://img.youtube.com/vi/TMJZxYzgdYg/maxresdefault.jpg)](https://youtu.be/TMJZxYzgdYg)

## What This Does
A basic attempt at making a cruise missile, what we are trying to do is simulate terrain hugging. We've got realistic state-space models (A, B, C, D matrices), LQR optimal controller, Kalman filter for state estimation, and it works for both 2D and 3D. 

<img width="1450" height="815" alt="Screenshot from 2025-12-12 01-11-57" src="https://github.com/user-attachments/assets/62db8eb1-1967-431f-b9cf-42ed302a78f6" />

The missile follows terrain-hugging paths or ballistic trajectories by minimizing the error between **flight path angle** and **line-of-sight angle** to a lookahead point on the reference trajectory. The erro is given by `theta_error = theta_LOS - theta_FPA`

<img width="478" height="110" alt="Screenshot from 2025-12-12 02-53-19" src="https://github.com/user-attachments/assets/ff154d39-f9d4-4812-b175-f1fd57f93a96" />

LQR drives this error to zero by computing optimal control inputs `u = -K(x - x_ref)`


## Core Control Architecture

**1. State Estimation (Kalman Filter) plus Optimal Control (LQR)**

<img width="966" height="577" alt="Screenshot from 2025-12-12 01-12-59" src="https://github.com/user-attachments/assets/eb718962-66a8-4504-9e35-15d04f1c42ba" />

**2. Guidance Law**

Line-of-sight to lookahead point (Ts ahead):

<img width="759" height="520" alt="Screenshot from 2025-10-24 20-00-54" src="https://github.com/user-attachments/assets/754e8608-d35a-4b4b-b3b8-2cf3225c6545" />

theta_LOS = arctan((y_ref(t+T) - y_m)/ (x_ref(t+T) - x_m))

Controller aligns flight path angle theta_FPA with theta_LOS.

## Key Results

### Mountainous terrain hugging trajectory (terrain_hugging.py)
Missile hugs very closely a multi-peaked terrain path (mountainous terrain proxy) without deviation, and hits the endpoint precisely.
<img width="1821" height="947" alt="Figure_4" src="https://github.com/user-attachments/assets/027020a5-06ab-4a69-981e-f792dc30516b" />


### 3D trajectory example (3d_trajectory.py)
Complex 3D path with sinusoidal variation in all axes. XY/YZ/XZ projections and azimuth/elevation plots show perfect multi-axis tracking from an initial offset.
<img width="1821" height="947" alt="Figure_2" src="https://github.com/user-attachments/assets/02d1c559-0bee-41fd-9877-fefdd78759d5" />


### Ballistic trajectory (ballistic.py)
<img width="1499" height="499" alt="Figure_1" src="https://github.com/user-attachments/assets/5eec2a3c-6286-4eb8-b511-55dc5d24deac" />


## How the guidance logic had been designed 

1. Pick a lookahead point on the reference trajectory
2. Compute LOS angle from missile to lookahead
3. LQR drives flight path angle → LOS angle
4. Control surfaces execute via inner-loop dynamics

## Setup & Run

```bash
# Clone the repo 
git clone

# enter the directory
python3 build -m venv venv

# Activate environment
source venv/bin/activate

# Terrain-hugging trajectory example
python3 terrain_hugging.py

# 3D trajectory following example
python3 3d_trajectory.py

# Ballistic trajectory example
python3 ballistic.py
```

## Future goals 
- Automatic trajectory generation using terrain height data and missile kinematics constraints.
- GPS denied navigation using visual odometry, and other advanced techniques like TERCOM, DSMAC
- Integrate and test it with a high-fidelity simulator for missiles. (Suggestions open)
- Clean modular code

Suggest improvements, and your ideas; this thing has a lot of potential.  
