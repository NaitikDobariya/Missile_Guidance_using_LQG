%% Missile GNC Setup
clc; close all; clear;
format short;

%% System Matrices (LTI State-Space)
A = [-1.064, 1.000; 290.26, 0.00];
B = [-0.25; -331.40];
C = [-123.34, 0.00; 0.00, 1.00];
D = [-13.41; 0.00];

states = {'AoA', 'q'};
inputs = {'\delta_c'};
outputs = {'Az', 'q'};

sys = ss(A, B, C, D, 'statename', states, ...
               'inputname', inputs, ...
               'outputname', outputs);

%% Transfer Function (control q)
TFs = tf(sys);
TF = TFs(2,1);
disp('Poles of open-loop transfer function:');
disp(pole(TF));

%% LQR Design
Q = [0.1, 0; 0, 0.1];     % State weighting
R = 0.5;                  % Control weighting

[K, S, e] = lqr(A, B, Q, R);

disp('Eigenvalues of A - BK:');
disp(eig(A - B*K));
disp('LQR Gain K:');
disp(K);

%% Closed-Loop System (A - BK)
Acl = A - B*K;
Bcl = B;
Ccl = C;
Dcl = D;

sys_cl = ss(Acl, Bcl, Ccl, Dcl, ...
    'statename', states, ...
    'inputname', inputs, ...
    'outputname', outputs);

TF_cl = tf(sys_cl);
TFc = TF_cl(2,1);
disp('Poles of closed-loop transfer function:');
disp(pole(TFc));

%% Kalman Filter Design (LQG)
G = eye(2);         % Process noise gain
H = zeros(2);       % Measurement noise gain

Qbar = diag([0.00015, 0.00015]);   % Process noise covariance
Rbar = diag([0.55, 0.55]);         % Measurement noise covariance

sys_n = ss(A, [B G], C, [D H]);    % Augmented system
[Kest, L, P] = kalman(sys_n, Qbar, Rbar, 0);

Aob = A - L*C;

disp('Observer Eigenvalues (A - LC):');
disp(eig(Aob));

%% Noise Sample Times
dT1 = 0.75;   % Process noise update time
dT2 = 0.25;   % Measurement noise update time

%% Missile Initial Conditions and Target Info
R_earth = 6371e3;         % Earth radius [m]
vel = 1021.08;            % Missile speed [m/s]
m2f = 3.2811;             % Meters to feet conversion
d2r = pi/180;             % Degrees to radians

% Target location
LAT_TARGET = 34.6588; %34.6588
LON_TARGET = -118.769745; %-118.769745
ELEV_TARGET = 795;        % m

% Initial launch location
LAT_INIT = 34.2329;
LON_INIT = -119.4573;
ELEV_INIT = 10000;        % m

% Obstacle location
LAT_OBS = 34.61916;
LON_OBS = -118.8429;

%% Distance Calculation (INIT to OBS)
l1 = LAT_INIT * d2r;
u1 = LON_INIT * d2r;
l2 = LAT_OBS  * d2r;
u2 = LON_OBS  * d2r;

dl = l2 - l1;
du = u2 - u1;

% Haversine formula
a = sin(dl/2)^2 + cos(l1)*cos(l2)*sin(du/2)^2;
c = 2 * atan2(sqrt(a), sqrt(1 - a));
d_horiz = R_earth * c;

% Slant range from initial to target
r = sqrt(d_horiz^2 + (ELEV_TARGET - ELEV_INIT)^2);

%% Initial Azimuth (Yaw)
lat1 = LAT_INIT * d2r;
lat2 = LAT_TARGET * d2r;
dLon = (LON_TARGET - LON_INIT) * d2r;

y = sin(dLon) * cos(lat2);
x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon);
yaw_init = atan2(y, x);   % radians

%% Initial Flight Path Angle
dh = abs(ELEV_TARGET - ELEV_INIT);
FPA_INIT = atan(dh / d_horiz);   % radians

assignin('base', 'A', A);
assignin('base', 'B', B);
assignin('base', 'C', C);
assignin('base', 'D', D);
assignin('base', 'K', K);
assignin('base', 'Acl', Acl);
assignin('base', 'L', L);
assignin('base', 'Aob', Aob);
