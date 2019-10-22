import numpy as np
import cvxpy as cvx
import sympy as sp

from global_parameters import K
from Models.parameters import *
from Optimization.utils import *


class Model:
    g = 9.81  # gravity acceleration

    def __init__(self, rpath, target_distance, target_velocity):
        '''
             State and Control Variables
        '''

        self.states_var_names = ['xw', 'yw', 'psi', 's', 'ey', 'epsi', 'Vx', 'delta', 'ts']
        self.controls_var_names = ['acc_x', 'delta_dot']  # acceleration-brake
        self.parameters_var_names = ['curvature']  # curvature enters to the model as reference

        # Parameters
        self.lr = 1.4  # the distance of real axle center to the center of gravity
        self.l = 2.9  # the distance between the axles
        self.mu = 0.6  # road friction
        self.mass = 1500  # kg

        # Fixed_Time Variables
        self.nx = 9  # number of states ; (integrated variables)
        self.nu = 2  # number of inputs ; steering and gas-brake
        self.nr = 1  # number of reference parameters; curvature and velocity

        # Optimization paramaters
        self.K = K

        # Reference Path
        ''' 
             Reference Path 
             path states = s, xw, yw, Ψ, κ, Δs, Δx_ds, Δy_ds
        '''
        self.rpath = rpath

        ## MAP RELATED
        self.current_curvature_map = None
        self.current_reference_map = None
        self.current_station_index = 0  # keeps the station
        self.current_station_dist = 0  # the distance travelled on the reference map

        self.st_interval = self.rpath[0, 5]  # distance between two stations in the path file
        self.st_index_horizon = int(50 / self.st_interval)  # how many station far ahead to get from the file as
        # reference

        # set initial states
        self.x_init = np.zeros((self.nx,))
        self.x_init[0:3] = self.rpath[0, (1, 2, 3)]  # set x, y, psi
        self.x_init[6] = target_velocity  # set speed m/s

        # set final states
        self.x_final = np.zeros((self.nx,))
        self.x_final[[3, 6]] = target_distance, target_velocity  # s-Vx
        self.x_final[0] = np.interp(target_distance, rpath[0:100, 0], rpath[0:100, 1])
        self.x_final[1] = np.interp(target_distance, rpath[0:100, 0], rpath[0:100, 2])
        self.x_final[2] = np.interp(target_distance, rpath[0:100, 0], rpath[0:100, 3])

        # self.tf_guess = self.x_final[3]

        # set slacks
        # Slack variables for the soft constraints
        self.s_prime_delta = cvx.Variable((K,), nonneg=True)
        self.s_prime_speed = cvx.Variable((K,), nonneg=True)
        self.s_prime_dist = cvx.Variable((K,), nonneg=True)
        self.s_prime_acc = cvx.Variable((K,), nonneg=True)

    def get_equations(self):
        """
            Vehicle Dynamical Equations in Vector Function

            :param x0; Current states - numpy column array
            :param u0: Current Inputs - numpy column array
            :param refs : Curvature table - kappa

        """
        """ 

            ** Kinematic Vehicle Model 
            STATES: 
                xw, yw, ψ, ψdot, s, ey, epsi, Vx, delta, ts 

            INPUTS:             
            acceleration   - acc_x     # desired braking and anc acceleration     
            steering input - deltadot  # Steering Velocity is the input

            x0 = xw 
            x1 = yw  
            x2 = psi
            x3 = s
            x4 = ey
            x5 = epsi
            x6 = Vx
            x7 = delta
            x8 = ts            

            u0 = acc_x 
            u1 = delta_dot
            

        """
        f = sp.zeros(self.nx, 1)

        x = sp.Matrix(sp.symbols('xw, yw, ψ, s, ey, epsi, Vx, delta, ts', real=True))
        u = sp.Matrix(sp.symbols('acc_x, deltadot', real=True))

        ## ROAD COORDINATES s parametrization
        p = sp.symbols('kappa', real=True)  # external reference as the parameter input - curvature

        # ## ROAD COORDINATES s parametrization
        # beta = sp.atan(self.lr * sp.tan(x[7, 0]) / self.l)
        # sdot = x[6, 0] * sp.cos(x[5, 0] + beta) / (1 - p * x[4, 0])
        # f[3, 0] = 1  # x[6, 0] * sp.cos(x[5, 0] + beta) / (1 - p * x[4, 0])  # sdot
        #
        # f[0, 0] = x[6, 0] * sp.cos(x[2, 0] + beta) / sdot  # Xw = V*cos(psi + beta)
        # f[1, 0] = x[6, 0] * sp.sin(x[2, 0] + beta) / sdot  # Yw = V*sin(psi + beta)
        # f[2, 0] = x[6, 0] * sp.sin(beta) / (self.lr * sdot)  # Psidot = psidot
        #
        # f[4, 0] = x[6, 0] * sp.sin(x[5, 0] + beta) / sdot  # eydot
        # f[5, 0] = f[2, 0] - p  # epsi_dot
        # f[6, 0] = u[0, 0] / sdot  # Vxdot = acc_x \input
        # f[7, 0] = u[1, 0] / sdot  # delta_dot = delta_dot \input
        # f[8, 0] = 1 / sdot  # delta_dot = delta_dot \input

        ## WITHOUT BETA
        sdot = x[6, 0] * sp.cos(x[5, 0]) / (1 - p * x[4, 0])

        f[3, 0] = 1  # sdot

        f[0, 0] = x[6, 0] * sp.cos(x[2, 0]) / sdot  # Xw = V*cos(psi)
        f[1, 0] = x[6, 0] * sp.sin(x[2, 0]) / sdot  # Yw = V*sin(psi)
        f[2, 0] = x[6, 0] * sp.tan(x[7, 0]) / (sdot * self.l)  # Psidot = psidot = (V/lr)tan(delta)

        f[4, 0] = x[6, 0] * sp.sin(x[5, 0]) / sdot  # eydot
        f[5, 0] = f[2, 0] - p  # epsi_dot
        f[6, 0] = u[0, 0] / sdot  # Vxdot = acc_x \input
        f[7, 0] = u[1, 0] / sdot  # delta_dot = delta_dot \input
        f[8, 0] = 1 / sdot

        ## TIME PARAMETRIZATION
        # f[3, 0] = x[6, 0] * sp.cos(x[5, 0]) / (1 - p * x[4, 0])  # sdot
        #
        # f[0, 0] = x[6, 0] * sp.cos(x[2, 0])  # Xw = V*cos(psi)
        # f[1, 0] = x[6, 0] * sp.sin(x[2, 0])  # Yw = V*sin(psi)
        # f[2, 0] = x[6, 0] * sp.tan(x[7, 0]) / (self.lr)  # Psidot = psidot = (V/lr)tan(delta)
        #
        # f[4, 0] = x[6, 0] * sp.sin(x[5, 0])  # eydot
        # f[5, 0] = f[2, 0] - p * f[3, 0]  # epsi_dot
        # f[6, 0] = u[0, 0]  # Vxdot = acc_x \input
        # f[7, 0] = u[1, 0]  # delta_dot = delta_dot \input

        f = sp.simplify(f)
        A = sp.simplify(f.jacobian(x))
        B = sp.simplify(f.jacobian(u))

        f_func = sp.lambdify((x, u, p), f, 'numpy')
        A_func = sp.lambdify((x, u, p), A, 'numpy')
        B_func = sp.lambdify((x, u, p), B, 'numpy')

        return f_func, A_func, B_func

    def get_current_map(self):
        # path states = s, xw, yw, Ψ, κ, Δs, Δx_ds, Δy_ds
        return self.current_reference_map

    def get_curvature_ref(self):
        '''
             Get the reference curvature for a specific horizon length
             path states = s, xw, yw, Ψ, κ, Δs, Δx_ds, Δy_ds

             :return : curvature reference κ from [current station --> end station]
        '''

        # s0 = self.current_station_index

        if self.current_station_index <= 5:
            s0 = self.current_station_index
        else:
            s0 = self.current_station_index - 1

        sf = s0 + self.st_index_horizon

        index_interval = slice(s0, sf)

        self.current_reference_map = self.rpath[index_interval]  # containts all the ref values for the given horizon
        self.current_curvature_map = self.current_reference_map[:, (0, 4)]

        # plt.plot(current_curvature_ref[:, 0], current_curvature_ref[:, 1])
        # plt.show()
        return self.current_curvature_map

    def update_last_station_index(self, distantace_travelled):
        self.current_station_dist = distantace_travelled

        last_distance = self.current_reference_map[self.current_reference_map[:, 0] <= distantace_travelled][-1, 0]
        last_index = np.floor(last_distance / self.st_interval).astype(int)
        self.current_station_index = last_index

    def initialize_trajectory(self, X, U):
        """
        Initialize the trajectory.

        :param X: Numpy array of states to be initialized
        :param U: Numpy array of inputs to be initialized
        :return: The initialized X and U
        """
        K = X.shape[1]
        sf = self.x_final[3]
        svec = np.linspace(0, sf, self.K)

        rfinal = 120
        Xest = np.interp(svec, self.rpath[0:rfinal, 0], self.rpath[0:rfinal, 1])
        Yest = np.interp(svec, self.rpath[0:rfinal, 0], self.rpath[0:rfinal, 2])
        Psi_est = np.interp(svec, self.rpath[0:rfinal, 0], self.rpath[0:rfinal, 3])
        kappa_est = np.interp(svec, self.rpath[0:rfinal, 0], self.rpath[0:rfinal, 4])
        delta_est = np.arctan(self.l * kappa_est)

        for k in range(K):
            alpha1 = (K - k) / K
            alpha2 = k / K
            X[:, k] = self.x_init * alpha1 + self.x_final * alpha2
            X[0, k] = Xest[k]
            X[1, k] = Yest[k]
            X[2, k] = Psi_est[k]
            X[7, k] = delta_est[k]

        ## ESTIME THE TIME STATE
        X[-1, :-1] = np.cumsum(np.diff(X[3, :]) / X[6, :-1])
        X[-1, -1] = X[-1, -2] + X[-1, -2] - X[-1, -3]
        U[0, :-1] = np.diff(X[6, :])
        U[0, -1] = U[0, -2]

        U[1, :-1] = np.diff(delta_est)
        U[1, -1] = U[1, -2]

        return X, U

    def get_constraints(self, X_v, U_v, X_last_p, U_last_p, kappa):
        """
        Get model specific constraints.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A list of cvx constraints

        states = xw, yw, ψ,  s, ey, epsi, Vx, delta, ts
        """
        constraints = []
        # TIRE ADHESION COEFFICIENT CONSTRAINT
        # muge = cvx.Constant(self.mu * self.g)
        #
        # a_y = [kappa[k] * X_last_p[6, k] ** 2 + 2 * kappa[k] * X_last_p[6, k] * (X_v[6, k] - X_last_p[6, k])
        #        for k in range(self.K)]
        #
        # soc_constraints = [cvx.square(U_v[0, k]) + a_y[k] ** 2 <= muge ** 2 for k in range(self.K)]
        # constraints += soc_constraints

        # STATE BOUNDS and LOWER BOUNDS
        state_upper = upper_bounds_states
        state_lower = lower_bounds_states

        # constraints += [cvx.abs(X_v[7, :]) - self.s_prime_delta <= state_upper['delta']]  # constraint on the steering
        constraints += [cvx.abs(X_v[7, :]) <= state_upper['delta']]
        constraints += [X_v[6, :] >= state_lower['Vx']]
        constraints += [X_v[6, :] <= state_upper['Vx']]

        ## Error Bounds
        # constraints += [cvx.abs(X_v[4, :])<=0.1]
        # constraints += [cvx.abs(X_v[5, :]) <= np.deg2rad(5)]

        # UPPER LIMITS of CONTROLS
        control_upper = upper_bounds_controls
        constraints += [cvx.abs(U_v[0, :]) <= control_upper['acc_x']]

        # constraints += [cvx.abs(U_v[1, :]) <= control_upper['delta_dot']]
        Deltat = X_v[-1, 1:] - X_v[-1, :-1]
        DeltaSigma = X_v[7, 1:] - X_v[7, :-1]
        constraints += [cvx.abs(DeltaSigma) <= control_upper['delta_dot'] * Deltat]

        ## Window Constraints
        # constraints += [X_v[6, 20:25] == 10]

        ## OBSTACLE AVOIDANCE
        # constraints += [X_v[4, -2:] >= 0.5]
        # constraints += [X_v[4, 20:24] == 0.5]

        ## ZERO DIVISION
        # constraints += [1 - kappa[k] * X_v[4, k] >= 0.01 for k in range(self.K)]

        return constraints

    def get_objective(self, X_v, U_v, X_last=0, U_last=0, Vdes=0):
        """
        Get model specific objective to be minimized.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs

        :return: A cvx objective function.
        """

        # PATH TRACKING OBJECTIVE
        '''
            Minimize e_y and e_psi 
            self.states_var_names = ['xw', 'yw', 'psi',  's', 'ey', 'epsi', 'Vx', 'delta', 'ts']
        '''

        objective = cvx.Minimize(cvx.norm(X_v[4, :]) * w_ey)
        objective += cvx.Minimize(cvx.norm(X_v[5, :]) * w_epsi)

        # Speed Objective
        # objective = cvx.Minimize(cvx.norm(X_v[6, :] - Vdes * D_x[6, 6] - C_x[6, 6]) * w_speed)

        # CONTROL OBJECTIVE
        du = U_v[0, 1:] - U_v[0, :-1]
        objective += cvx.Minimize(cvx.norm(du) * w_jerk)
        objective += cvx.Minimize(cvx.norm(U_v[1, :]) * w_deltadot)

        # Terminal Value Objective D_x is the scaler defined in the parameters file
        # objective += cvx.Minimize(cvx.norm(X_v[6, -1] - (Vdes * D_x[6, 6] + C_x[6, 6])) * w_speed_terminal)

        # Slack Variable Minimization
        # objective += cvx.Minimize(cvx.norm(self.s_prime_delta) * w_soft)
        # objective += cvx.Minimize(cvx.norm(self.s_prime_acc) * w_soft)
        # objective += cvx.Minimize(cvx.norm(self.s_prime_speed) * w_soft)
        # objective += cvx.Minimize(cvx.norm(self.s_prime_dist) * w_soft)

        return objective

    def get_linear_cost_constraints(self):
        total_slack_cost = 0.0
        # total_slack_cost += np.sum(self.s_prime_delta.value)
        # total_slack_cost += np.sum(self.s_prime_speed.value)
        # total_slack_cost += np.sum(self.s_prime_dist.value)
        # total_slack_cost += np.sum(self.s_prime_acc.value)
        # total_slack_cost = 0

        return total_slack_cost

    def get_nonlinear_cost_constraints(self, Xnl, U, kappa):
        cost = 0
        # muge = self.mu * self.g
        #
        # acc_y = kappa * Xnl[6, :] ** 2
        # acc_x = U[0, :]
        # acc_c = np.hypot(acc_y, acc_x)
        #
        # # constraint violation
        # cost = - np.sum(np.minimum((muge - acc_c), 0))

        # STEERING VIOLATION
        # deltamax = upper_bounds_states['delta']
        # cost += np.linalg.norm(Xnl[7, :]-deltamax, 1)

        return cost

    def compute_initial_errors(self, Xw, Xcar):
        '''

        :param Xw: [Xw0, Yw0, Psiw0]
        :param Xcar: [Xcar, Ycar, Pcar]
        :return: ey, epsi
        '''

        Xw0, Yw0, Psiw0 = Xw  # extract the states
        Xc0, Yc0, Psic0 = Xcar

        epsi = wrapToPi(Psic0 - Psiw0)
        # ey = (Yc0 - Yw0) * np.cos(Psic0) - (Xc0 - Xw0) * np.sin(Psic0)
        ey = (Yc0 - Yw0) * np.cos(Psic0) - (Xc0 - Xw0) * np.sin(Psic0)

        return ey, epsi

    def get_path_start(self):
        X0 = self.rpath[0, 1]
        Y0 = self.rpath[0, 2]
        Psi0 = self.rpath[0, 3]

        return X0, Y0, Psi0

    def get_current_XYPsi(self, s):

        map = self.current_reference_map
        X = np.interp(s, map[:, 0], map[:, 1])
        Y = np.interp(s, map[:, 0], map[:, 2])
        Psi = np.interp(s, map[:, 0], map[:, 3])

        return X, Y, Psi
