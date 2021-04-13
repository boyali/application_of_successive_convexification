import numpy as np
from scipy.integrate import odeint, solve_ivp


# from scikits.odes import odeint as sc


class FirstOrderHold:
    def __init__(self, m, K, sigma):
        self.K = K
        self.m = m
        self.nx = m.nx
        self.nu = m.nu

        self.A_bar = np.zeros([m.nx * m.nx, K - 1])
        self.B_bar = np.zeros([m.nx * m.nu, K - 1])
        self.C_bar = np.zeros([m.nx * m.nu, K - 1])
        self.z_bar = np.zeros([m.nx, K - 1])

        # vector indices for flat matrices
        x_end = m.nx
        A_bar_end = m.nx * (1 + m.nx)
        B_bar_end = m.nx * (1 + m.nx + m.nu)
        C_bar_end = m.nx * (1 + m.nx + m.nu + m.nu)
        z_bar_end = m.nx * (1 + m.nx + m.nu + m.nu + 1)

        self.x_ind = slice(0, x_end)
        self.A_bar_ind = slice(x_end, A_bar_end)
        self.B_bar_ind = slice(A_bar_end, B_bar_end)
        self.C_bar_ind = slice(B_bar_end, C_bar_end)
        self.z_bar_ind = slice(C_bar_end, z_bar_end)

        self.f, self.A, self.B = m.get_equations()

        # integration initial condition
        self.V0 = np.zeros((m.nx * (1 + m.nx + m.nu + m.nu + 1),))
        self.V0[self.A_bar_ind] = np.eye(m.nx).reshape(-1)

        self.sigma = sigma
        self.dt = sigma * 1. / (K - 1)

        # For ODE integration for scikits.odes package

    def calculate_discretization(self, X, U, Kappa):
        """
        Calculate discretization for given states, inputs and total time.

        :param X: Matrix of states for all time points
        :param U: Matrix of inputs for all time points
        :return: The discretization matrices
        """
        for k in range(self.K - 1):
            self.V0[self.x_ind] = X[:, k]
            V = np.array(odeint(self._ode_dVdt, self.V0, (0, self.dt),
                                args=(U[:, k], U[:, k + 1], Kappa[k], Kappa[k + 1]))[1, :])

            # flatten matrices in column-major (Fortran) order for CVXPY
            Phi = V[self.A_bar_ind].reshape((self.nx, self.nx))
            self.A_bar[:, k] = Phi.flatten(order='F')
            self.B_bar[:, k] = np.matmul(Phi, V[self.B_bar_ind].reshape((self.nx, self.nu))).flatten(order='F')
            self.C_bar[:, k] = np.matmul(Phi, V[self.C_bar_ind].reshape((self.nx, self.nu))).flatten(order='F')
            self.z_bar[:, k] = np.matmul(Phi, V[self.z_bar_ind])

        return self.A_bar, self.B_bar, self.C_bar, self.z_bar

    def _ode_dVdt(self, V, t, u_t0, u_t1, p_t0, p_t1):
        """
        ODE function to compute dVdt.

        :param V: Evaluation state V = [x, Phi_A, B_bar, C_bar, z_bar]
        :param t: Evaluation time
        :param u_t0: Input at start of interval
        :param u_t1: Input at end of interval
        :return: Derivative at current time and state dVdt
        """
        alpha = (self.dt - t) / self.dt
        beta = t / self.dt
        x = V[self.x_ind]

        u = u_t0 + (t / self.dt) * (u_t1 - u_t0)
        p = p_t0 + (t / self.dt) * (p_t1 - p_t0)

        # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
        # and pre-multiplying with \Phi_A(\tau_{k+1},\tau_k) after integration
        Phi_A_xi = np.linalg.inv(V[self.A_bar_ind].reshape((self.nx, self.nx)))

        A_subs = self.A(x, u, p)
        B_subs = self.B(x, u, p)
        f_subs = self.f(x, u, p)

        dVdt = np.zeros_like(V)
        dVdt[self.x_ind] = f_subs.T
        dVdt[self.A_bar_ind] = np.matmul(A_subs, V[self.A_bar_ind].reshape((self.nx, self.nx))).reshape(-1)
        dVdt[self.B_bar_ind] = np.matmul(Phi_A_xi, B_subs).reshape(-1) * alpha
        dVdt[self.C_bar_ind] = np.matmul(Phi_A_xi, B_subs).reshape(-1) * beta
        z_t = np.squeeze(f_subs) - np.matmul(A_subs, x) - np.matmul(B_subs, u)

        dVdt[self.z_bar_ind] = np.matmul(Phi_A_xi, z_t)
        return dVdt

    def integrate_nonlinear_piecewise(self, X_l, U, Kappa):
        """
        Piecewise integration to verify accuracy of linearization.
        :param X_l: Linear state evolution
        :param U: Linear input evolution
        :return: The piecewise integrated dynamics
        """

        X_nl = np.zeros_like(X_l)
        X_nl[:, 0] = X_l[:, 0]

        for k in range(self.K - 1):
            # deltat = X_l[-1, k + 1] - X_l[-1, k]
            # X_nl[:, k + 1] = odeint(self._dx, X_nl[:, k], (0, deltat),
            #                         args=(U[:, k], U[:, k + 1], Kappa[k], Kappa[k + 1]))[1, :]

            '''
                ODEINT of SCIPY
            '''
            # X_nl[:, k + 1] = odeint(self._dx, X_nl[:, k], (0, self.dt),
            #                         args=(U[:, k], U[:, k + 1], Kappa[k], Kappa[k + 1]))[1, :]

            '''
                SOLVE_IVP of scipy
            '''
            sol = solve_ivp(fun=lambda t, y: self._dx(y, t, U[:, k], U[:, k + 1], Kappa[k], Kappa[k + 1]),
                            y0=X_nl[:, k], t_span=[0, self.dt], method='Radau')  # RK45, RK23, Radau, LSODA, BDF

            X_nl[:, k + 1] = sol.y[:, -1]

            '''
                SOLVE  of SCIKITS.ODES
                method = bdf, admo, rk5, rk8
            '''
            # sol = sc.odeint(rhsfun=lambda t, x, xdot:
            # self._dxsc(t, x, xdot, U[:, k], U[:, k + 1], Kappa[k], Kappa[k + 1]), tout=(0, self.dt), y0=X_nl[:, k],
            #                 method='rk8', old_api=False)
            #
            # X_nl[:, k + 1] = sol.values.y[-1, :]

            # print(k)

        return X_nl

    def integrate_nonlinear_full(self, x0, U, Kappa):
        """
        Simulate nonlinear behavior given an initial state and an input over time.
        :param x0: Initial state
        :param U: Linear input evolution
        :return: The full integrated dynamics
        """
        X_nl = np.zeros([x0.size, self.K])
        X_nl[:, 0] = x0

        for k in range(self.K - 1):
            # deltat = X_l[-1, k + 1] - X_l[-1, k]
            # X_nl[:, k + 1] = odeint(self._dx, X_nl[:, k], (0, deltat),
            #                         args=(U[:, k], U[:, k + 1], Kappa[k], Kappa[k + 1]))[1, :]

            '''
                ODEINT of SCIPY
            '''
            # X_nl[:, k + 1] = odeint(self._dx, X_nl[:, k], (0, self.dt),
            #                         args=(U[:, k], U[:, k + 1], Kappa[k], Kappa[k + 1]))[1, :]

            '''
                SOLVE_IVP of scipy
            '''
            sol = solve_ivp(fun=lambda t, y: self._dx(y, t, U[:, k], U[:, k + 1], Kappa[k], Kappa[k + 1]),
                            y0=X_nl[:, k], t_span=[0, self.dt], method='Radau')  # RK45, RK23, Radau, LSODA, BDF

            X_nl[:, k + 1] = sol.y[:, -1]

            '''
                SOLVE  of SCIKITS.ODES
                method = bdf, admo, rk5, rk8
            '''
            # sol = sc.odeint(rhsfun=lambda t, x, xdot:
            # self._dxsc(t, x, xdot, U[:, k], U[:, k + 1], Kappa[k], Kappa[k + 1]), tout=(0, self.dt), y0=X_nl[:, k],
            #                 method='rk8', old_api=False)
            #
            # X_nl[:, k + 1] = sol.values.y[-1, :]

            # print(k)

        return X_nl

    def _dx(self, x, t, u_t0, u_t1, p_t0, p_t1):
        u = u_t0 + (t / self.dt) * (u_t1 - u_t0)
        p = p_t0 + (t / self.dt) * (p_t1 - p_t0)

        return np.squeeze(self.f(x, u, p))

    def _dxsc(self, t, x, xdot, u_t0, u_t1, p_t0, p_t1):
        u = u_t0 + (t / self.dt) * (u_t1 - u_t0)
        p = p_t0 + (t / self.dt) * (p_t1 - p_t0)

        f = self.f(x, u, p)

        for k in range(self.nx):
            xdot[k] = f[k]
