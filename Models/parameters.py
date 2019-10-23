import numpy as np
from global_parameters import *
import numpy.matlib

# Weight constants
w_nu = 1e5  # virtual control
w_ey = 1
w_epsi = 1

w_jerk = 0.5
w_deltadot = 0.5

w_speed = 0.1
w_speed_terminal = 1
w_soft = 5

# initial trust region radius
tr_radius = 5

# trust region variables
rho_0 = 0.0
rho_1 = 0.25
rho_2 = 0.9
alpha = 2.0
beta = 3.2


def compute_scalers(bounds):
    '''

    x_hat = D_x*x + Cx  # x is the original variable and x_hat \in [-1, 1]
    u_hat = D_u*u + Cu

    // inverse of above
    x = D_xhat * x_hat + C_xhat
    u = D_uhat * u_hat + C_uhat

    Computing D_xhat, .. params is easier then computing Dx, Du, we first comput xhat coefficients

    :param bounds: [lower, upper]
    :return: Dx,u and Cx,u
    '''

    lower, upper = bounds

    # extract values of upper and lower bounds : vectors only include nonzero values for the scaled states
    upper = np.array([vals for _, vals in upper.items()])  # nonscaled variables's value is zero
    lower = np.array([vals for _, vals in lower.items()])

    # Inverse scaling coefficients
    D_hat = (upper - lower) / 2
    C_hat = (((upper + lower) / 2))[:, None]

    D_hat[D_hat == 0] = 1.0  # non-scaled variable scales are 1
    D_hat = np.diag(D_hat)

    # Original scaling
    D = np.linalg.inv(D_hat)
    C = -D @ C_hat

    return D, C, D_hat, C_hat


## STATE BOUNDS
states = ['xw', 'yw', 'psi', 's', 'ey', 'epsi', 'Vx', 'delta', 'ts']
upper_bounds_states = {var_name: 0 for var_name in states}
lower_bounds_states = {var_name: 0 for var_name in states}

upper_bounds_states['s'] = 20
lower_bounds_states['s'] = 0

upper_bounds_states['delta'] = np.deg2rad(40)  # deg/sec to rad/sec
lower_bounds_states['delta'] = np.deg2rad(-40)  # deg/sec to rad/sec

upper_bounds_states['ey'] = 2
lower_bounds_states['ey'] = -2

upper_bounds_states['epsi'] = np.deg2rad(3)
lower_bounds_states['epsi'] = -np.deg2rad(3)

upper_bounds_states['Vx'] = 20  # m/s
lower_bounds_states['Vx'] = 5

upper_bounds_states['ts'] = 10  # time scale
lower_bounds_states['ts'] = 0

## CONTROL BOUNDS
controls = ['acc_x', 'delta_dot']
upper_bounds_controls = {var_name: 0 for var_name in controls}
lower_bounds_controls = {var_name: 0 for var_name in controls}

upper_bounds_controls['acc_x'] = 5  # acceleration bound 5 m/s/s
lower_bounds_controls['acc_x'] = -5

upper_bounds_controls['delta_dot'] = np.deg2rad(60)  # steering rate
lower_bounds_controls['delta_dot'] = -np.deg2rad(60)

'''
    x_hat = D_x*x + Cx  # x is the original variable and x_hat \in [-1, 1]
    u_hat = D_u*u + Cu

    // inverse of above
    x = D_xhat * x_hat + C_xhat
    u = D_uhat * u_hat + C_uhat
'''
# State Scaling Matrix
D_x, C_x, D_xhat, C_xhat = compute_scalers([lower_bounds_states, upper_bounds_states])

# Control Scaling Matrix
D_u, C_u, D_uhat, C_uhat = compute_scalers([lower_bounds_controls, upper_bounds_controls])

# Scaling Centers, will change later
C_x = np.matlib.repmat(C_x, 1, K)
C_u = np.matlib.repmat(C_u, 1, K)

C_xhat = np.matlib.repmat(C_xhat, 1, K)
C_uhat = np.matlib.repmat(C_uhat, 1, K)
