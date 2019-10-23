import cvxpy as cvx


class MPCproblem:
    """
    Defines a standard Successive Convexification problem and adds the model specific constraints and objectives.

    :param m: The model object
    :param K: Number of discretization points
    """

    def __init__(self, m, K):
        # Variables:
        self.var = dict()
        self.K = K

        # Scaled Variables \in [1, 1]
        self.var['X_hat'] = cvx.Variable((m.nx, self.K))  # state trajectory
        self.var['U_hat'] = cvx.Variable((m.nu, self.K))  # until the end of trajectory because of FOH otherwise K-1
        self.var['nu_hat'] = cvx.Variable((m.nx, self.K - 1))

        # Scaling Variables
        self.par = dict()
        self.par['D_xhat'] = cvx.Parameter((m.nx, m.nx))
        self.par['D_x'] = cvx.Parameter((m.nx, m.nx))
        self.par['C_xhat'] = cvx.Parameter((m.nx, K))
        self.par['C_x'] = cvx.Parameter((m.nx, K))

        # Scaling cvx Parameters
        self.par['D_uhat'] = cvx.Parameter((m.nu, m.nu))
        self.par['D_u'] = cvx.Parameter((m.nu, m.nu))
        self.par['C_uhat'] = cvx.Parameter((m.nu, K))
        self.par['C_u'] = cvx.Parameter((m.nu, K))

        # Parameters:
        self.par['A_bar'] = cvx.Parameter((m.nx * m.nx, K - 1))
        self.par['B_bar'] = cvx.Parameter((m.nx * m.nu, K - 1))
        self.par['C_bar'] = cvx.Parameter((m.nx * m.nu, K - 1))
        self.par['z_bar'] = cvx.Parameter((m.nx, K - 1))

        self.par['X_init'] = cvx.Parameter((m.nx,))
        self.par['X_last'] = cvx.Parameter((m.nx, K))
        self.par['U_last'] = cvx.Parameter((m.nu, K))

        self.par['weight_nu'] = cvx.Parameter(nonneg=True)
        self.par['tr_radius'] = cvx.Parameter(nonneg=True)

        # Original - Unscaled Variables
        self.var['X'] = self.par['D_xhat'] * self.var['X_hat'] + self.par['C_xhat']
        self.var['U'] = self.par['D_uhat'] * self.var['U_hat'] + self.par['C_uhat']
        # self.var['nu'] = self.par['D_xhat'] * self.var['nu_hat'] + self.par['C_xhat'][:, :-1]
        self.var['nu'] = cvx.Variable((m.nx, K - 1))

        self.par['kappa'] = cvx.Parameter(self.K)  # first row is reserved for the distance

        # ADD Desired Velocity and Obstacle Avoidance Paprameters
        self.par['Vdes'] = cvx.Parameter()  # desired velocity is set by this parameter

        '''
            These parameters are set by default to higher margins so that they are passive when there is no obstacle
        '''
        self.par['eydes_min'] = cvx.Parameter(value=-1) # desired eydes_min is set by this parameter
        self.par['eydes_max'] = cvx.Parameter(value=1)  # desired eydes_max is set by this parameter
        self.par['obstacle_ind'] = cvx.Parameter(value=0, integer=True)  # desired eydes_max is set by this parameter

        # Constraints:
        constraints = []

        # Get Model Constraints
        constraints += m.get_constraints(self.var['X'], self.var['U'],
                                         self.par['X_last'], self.par['U_last'],
                                         self.par['kappa'])


        # Get Obstacle Constraints
        constraints += m.get_constraints_obstacles(self.var["X"], self.par)

        # Dynamics:
        constraints += [
            self.var['X'][:, k + 1] ==
            cvx.reshape(self.par['A_bar'][:, k], (m.nx, m.nx)) * self.var['X'][:, k]
            + cvx.reshape(self.par['B_bar'][:, k], (m.nx, m.nu)) * self.var['U'][:, k]
            + cvx.reshape(self.par['C_bar'][:, k], (m.nx, m.nu)) * self.var['U'][:, k + 1]
            + self.par['z_bar'][:, k]
            + self.var['nu'][:, k]
            for k in range(K - 1)
        ]

        # Add initial conditions constraints
        constraints += [self.var['X'][:, 0] == self.par['X_init']]

        # Trust region:
        # Trust Region Constraint
        # states are 'xw, yw, ψ, s, ey, epsi, Vx, delta, ts'
        # du = self.var['U_hat'] - (self.par['D_u'] * self.par['U_last'] + self.par['C_u'])
        # dx = self.var['X_hat'] - (self.par['D_x'] * self.par['X_last'] + self.par['C_x'])

        du = self.var['U_hat'] - (self.par['D_u'] * self.par['U_last'] + self.par['C_u'])
        dx = self.var['X_hat'][4:-1, :] - (self.par['D_x'] * self.par['X_last'] + self.par['C_x'])[4:-1, :]
        # dx = self.var['X_hat'] - (self.par['D_x'] * self.par['X_last'] + self.par['C_x'])
        # ddu = cvx.vstack([du, dx])

        # constraints += [cvx.norm(ddu[:, k], 1) <= self.par['tr_radius'] for k in range(self.K)]
        # constraints += [cvx.norm(ddu, 1) <= self.par['tr_radius']]
        constraints += [cvx.norm(dx, 1) + cvx.norm(du, 1) <= self.par['tr_radius']]

        # Objective:
        # model_objective = m.get_objective(self.var['X'], self.var['U'], self.par['X_last'], self.par['U_last'])
        model_objective = m.get_objective(self.var['X_hat'], self.var['U_hat'], self.par['D_x'] * self.par['X_last']
                                          + (self).par['C_x'], self.par['D_u'] * self.par['U_last'] + (self).par['C_u'],
                                          self.par['Vdes'])

        sc_objective = cvx.Minimize(self.par['weight_nu'] * cvx.norm(self.var['nu'], 1))
        # sc_objective += cvx.Minimize(cvx.sum_squares(dx) + cvx.sum_squares(du))

        objective = sc_objective + model_objective

        self.prob = cvx.Problem(objective, constraints)

    def set_parameters(self, **kwargs):
        """
        All parameters have to be filled before calling solve().
        """

        for key in kwargs:
            if key in self.par:
                # print(key)
                self.par[key].value = kwargs[key]
            else:
                print(f'Parameter \'{key}\' does not exist.')

    def print_available_parameters(self):
        print('Parameter names:')
        for key in self.par:
            print(f'\t {key}')
        print('\n')

    def print_available_variables(self):
        print('Variable names:')
        for key in self.var:
            print(f'\t {key}')
        print('\n')

    def get_variable(self, name):
        """
        :param name: Name of the variable.
        :return The value of the variable.
        """

        if name in self.var:
            return self.var[name].value
        else:
            print(f'Variable \'{name}\' does not exist.')
            return None

    def solve(self, **kwargs):
        '''
            Error is the problem.solution.status
        :param kwargs:
        :return:
        '''
        error = 'None'
        try:
            self.prob.solve(**kwargs)
            error = self.prob.solution.status
        except cvx.SolverError:
            error = self.prob.solution.status

        return error
