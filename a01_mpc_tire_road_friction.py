from time import time
import numpy as np
from Models.model_scoordinates import Model
from Optimization.discretization import FirstOrderHold
from Optimization.mpcproblem import SCProblem
from Models.parameters import *
from Models.utils import *
import pickle

# load path
rpath = np.loadtxt('test_path.txt')

# INITIALIZATION--------------------------------------------------------------------------------------------------------

target_distance, target_speed = 20, 12  # [m], [m/s]
sigma = target_distance

m = Model(rpath, target_distance, target_speed)


## INITIALIZE THE INITIAL STATES
m.x_init[6] = target_speed


# state and input
X = np.empty(shape=[m.nx, K])
U = np.empty(shape=[m.nu, K])

# INTEGRATOR --------------------------------------------------------------------------------------------------------
integrator = FirstOrderHold(m, K, sigma)
problem = SCProblem(m, K)

problem.set_parameters(D_xhat=D_xhat, D_x=D_x, D_uhat=D_uhat, D_u=D_u)

problem.set_parameters(D_xhat=D_xhat, D_x=D_x, C_xhat=C_xhat, C_x=C_x,
                       D_uhat=D_uhat, D_u=D_u, C_uhat=C_uhat, C_u=C_u)

last_nonlinear_cost = None
converged = False

'''
    Since our states are on the equi-distance grid, we know the curvature in advance
'''


X, U = m.initialize_trajectory(X, U)

# U = np.zeros((2, K))
# x0 = m.x_init
# X = integrator.integrate_nonlinear_full(x0, U, kappa_estimated)

# Keep State History and Save to Logs

'''
    Since we are optimizing single plan over iterations, we will keep the computed interval trajectory and controls 
    and pickle it to use with jupyter notebooks
'''

state_history_list = []
control_history_list = []
optimization_history = {key: [] for key in ['sc_cost', 'constraint_cost']}

# state_history_list.append(X[:, 0].copy())
# control_history_list.append(U[:, 0].copy())

## Save History to this file
logs_pickle = dict()
logs_pickle['Initial Estimate'] = X


# Keep the travelled distance
s0 = 0

for tk in range(120):
    t0_it = time()
    print('-' * 50)
    print('-' * 18 + f' Time Step {str(tk + 1).zfill(2)} ' + '-' * 18)
    print('-' * 50)

    t0_tm = time()

    ## Get Curvature Reference
    curvature_ref = m.get_curvature_ref()
    ds_grid = np.linspace(s0, s0 + target_distance, K)
    kappa_estimated = np.interp(ds_grid, curvature_ref[:, 0], curvature_ref[:, 1])

    A_bar, B_bar, C_bar, z_bar = integrator.calculate_discretization(X, U, kappa_estimated)
    print(format_line('Time for transition matrices', time() - t0_tm, 's'))

    problem.set_parameters(A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, z_bar=z_bar, X_init=X[:, 0],
                           X_last=X, U_last=U, kappa=kappa_estimated,
                           weight_nu=w_nu, tr_radius=tr_radius)

    ### ADD SOLUTION TO THE LIST
    state_history_list.append(X[:, 0].copy())
    state_history_list[-1][3] = s0 # put s0 back when saving
    control_history_list.append(U[:, 0].copy())

    ## Convergence
    converged = False

    for it in range(iterations):
        error = problem.solve(verbose=verbose_solver, solver=solver, max_iters=200)
        print(format_line('Solver Error', error))

        if error =='optimal':

            # get solution
            new_X = problem.get_variable('X')
            new_U = problem.get_variable('U')

            # X_nl = integrator.integrate_nonlinear_full(x0, new_U, kappa_estimated)
            X_nl = integrator.integrate_nonlinear_piecewise(new_X, new_U, kappa_estimated)

            linear_cost_dynamics = np.linalg.norm(problem.get_variable('nu'), 1)
            nonlinear_cost_dynamics = np.linalg.norm(new_X - X_nl, 1)

            linear_cost_constraints = m.get_linear_cost_constraints()  # m.get_linear_cost()
            nonlinear_cost_constraints = 0  # m.get_nonlinear_cost_constraints(X_nl, new_U, kappa_estimated)  #
            # m.get_nonlinear_cost(X=new_X, U=new_U)

            linear_cost = linear_cost_dynamics + linear_cost_constraints  # J
            nonlinear_cost = nonlinear_cost_dynamics + nonlinear_cost_constraints  # L

            if last_nonlinear_cost is None:
                last_nonlinear_cost = nonlinear_cost
                X[:, :-1] = X_nl[:, 1:]
                X[:, -1] = X_nl[:, -1]
                U[:, :-1] = new_U[:, 1:]
                U[:, -1] = new_U[:, -1]

                # subtract s0 from s all the time
                X[3, :] = X[3, :] - X[3, 0]

                s0 = s0 + X_nl[3, 1] - X_nl[3, 0]
                m.update_last_station_index(s0)
                break

            actual_change = last_nonlinear_cost - nonlinear_cost  # delta_J
            predicted_change = last_nonlinear_cost - linear_cost  # delta_L

            print('')
            print(format_line('Virtual Control Cost', linear_cost_dynamics))
            print(format_line('Constraint Cost', linear_cost_constraints))
            print('')
            print(format_line('Actual change', actual_change))
            print(format_line('Predicted change', predicted_change))
            print('')

            if abs(predicted_change) < 1e-2:
                last_nonlinear_cost = nonlinear_cost
                converged = True

                X[:, :-1] = X_nl[:, 1:]
                X[:, -1] = X_nl[:, -1]
                U[:, :-1] = new_U[:, 1:]
                U[:, -1] = new_U[:, -1]

                # subtract s0 from s all the time
                X[3, :] = X[3, :] - X[3, 0]

                s0 = s0 + X_nl[3, 1] - X_nl[3, 0]
                m.update_last_station_index(s0)
                break

            else:
                rho = actual_change / predicted_change
                if rho < rho_0:
                    # reject solution
                    tr_radius /= alpha
                    print(f'Trust region too large. Solving again with radius={tr_radius}')
                else:
                    # accept solution
                    last_nonlinear_cost = nonlinear_cost
                    converged = True

                    X[:, :-1] = X_nl[:, 1:]
                    X[:, -1] = X_nl[:, -1]
                    U[:, :-1] = new_U[:, 1:]
                    U[:, -1] = new_U[:, -1]

                    # subtract s0 from s all the time
                    X[3, :] = X[3, :] - X[3, 0]

                    s0 = s0 + X_nl[3, 1] - X_nl[3, 0]
                    m.update_last_station_index(s0)

                    print('Solution accepted.')

                    if rho < rho_1:
                        print('Decreasing radius.')
                        tr_radius /= alpha
                    elif rho >= rho_2:
                        print('Increasing radius.')
                        tr_radius *= beta

                    last_nonlinear_cost = nonlinear_cost
                    break

            problem.set_parameters(tr_radius=tr_radius)

            if converged:
                break

            print('-' * 50)

        else:
            print('It is not optimal, recomputing by the new trust region \n')
            print(f'Trust region too large. Solving again with radius={tr_radius}')

            tr_radius /= alpha

            problem.set_parameters(tr_radius=tr_radius)


    print('')
    print(format_line('Time for iteration', time() - t0_it, 's'))
    print('')



logs_pickle['States'] = np.vstack(state_history_list).transpose()
logs_pickle['Controls'] = np.vstack(control_history_list).transpose()
logs_pickle['Optimization_Params'] = optimization_history
logs_pickle['Current_Map'] = m.current_reference_map
# logs_pickle['kappa'] = kappa_estimated

file_name = './Logs/logs_pickle_tire.pickle'
with open(file_name, 'wb') as handle:
    pickle.dump(logs_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
