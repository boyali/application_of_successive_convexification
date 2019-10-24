from time import time
import numpy as np
from Models.model_scoordinates import Model
from Optimization.discretization import FirstOrderHold
from Optimization.mpcproblem import MPCproblem
from Models.parameters import *
from Models.utils import *
import pickle

# load path
rpath = np.loadtxt('test_path.txt')

# INITIALIZATION--------------------------------------------------------------------------------------------------------

target_distance, target_speed = 40, 12  # [m], [m/s]
sigma = target_distance

m = Model(rpath, target_distance, target_speed)

# INTEGRATOR --------------------------------------------------------------------------------------------------------
integrator = FirstOrderHold(m, K, sigma)
problem = MPCproblem(m, K)

problem.set_parameters(Vdes=target_speed)

problem.set_parameters(D_xhat=D_xhat, D_x=D_x, D_uhat=D_uhat, D_u=D_u)

problem.set_parameters(D_xhat=D_xhat, D_x=D_x, C_xhat=C_xhat, C_x=C_x,
                       D_uhat=D_uhat, D_u=D_u, C_uhat=C_uhat, C_u=C_u)

last_nonlinear_cost = None
converged = False

'''
    Since our states are on the equi-distance grid, we know the curvature in advance
'''

x0 = np.zeros((m.nx,))  # initial states
Xw0, Yw0, Psiw0 = m.get_path_start()
Xc0, Yc0, Psic0 = 100, 0, np.deg2rad(90)

ey, epsi = m.compute_initial_errors([Xw0, Yw0, Psiw0], [Xc0, Yc0, Psic0])

x0[0] = Xc0  # Xcar
x0[1] = Yc0  # Ycar
x0[2] = Psic0  # Psicar
x0[4] = ey  # assign initial lateral error - ey
x0[5] = epsi  # assign initial heading angle error -epsi
x0[6] = target_speed  # assign initial velocity Vx#
m.x_init = x0

# X = np.zeros((m.nx, K))
# U = np.zeros((m.nu, K))
# X, U = m.initialize_trajectory(X, U)

# Keep the travelled distance
s0 = 0

## Get Curvature Reference
curvature_ref = m.get_curvature_ref()
ds_grid = np.linspace(s0, s0 + target_distance, K)
kappa_estimated = np.interp(ds_grid, curvature_ref[:, 0], curvature_ref[:, 1])

U = np.zeros((m.nu, K))
X = integrator.integrate_nonlinear_full(x0, U, kappa_estimated)

# Keep State History and Save to Logs

'''
    Since we are optimizing single plan over iterations, we will keep the computed interval trajectory and controls 
    and pickle it to use with jupyter notebooks
'''

state_history_list = []
control_history_list = []
curvature_history_list = []
optimization_history = {key: [] for key in ['sc_cost', 'constraint_cost']}

# state_history_list.append(X[:, 0].copy())
# control_history_list.append(U[:, 0].copy())

## Save History to this file
logs_pickle = dict()
logs_pickle['Initial Estimate'] = X

# just to make sure the error variable is declarated
error = 'infeasible'

##
if len(m.obs_loc):

    file_name = './Logs/logs_pickle_tire_obstacle.pickle'

else:
    file_name = './Logs/logs_pickle_tire_no_obstacle.pickle'


for tk in range(250):

    print('-' * 50)
    print('-' * 18 + f' Time Step {str(tk + 1).zfill(2)} ' + '-' * 18)
    print('-' * 50)

    ## Get Curvature Reference
    curvature_ref = m.get_curvature_ref()
    ds_grid = np.linspace(s0, s0 + target_distance, K)
    kappa_estimated = np.interp(ds_grid, curvature_ref[:, 0], curvature_ref[:, 1])

    # ## SET ERROR
    # Xwc, Ywc, Psiwc = m.get_current_XYPsi(s=s0)
    # Xc0, Yc0, Psic0 = X[0, 0], X[1, 0], X[2, 0]
    #
    # ey, epsi = m.compute_initial_errors([Xw0, Yw0, Psiw0], [Xc0, Yc0, Psic0])
    # X[4, 0] = ey
    # X[5, 0] = epsi

    t0_tm = time()

    X = integrator.integrate_nonlinear_full(X[:, 0], U, kappa_estimated)
    A_bar, B_bar, C_bar, z_bar = integrator.calculate_discretization(X, U, kappa_estimated)

    print(format_line('Time for transition matrices', time() - t0_tm, 's'))

    problem.set_parameters(A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, z_bar=z_bar, X_init=X[:, 0],
                           X_last=X, U_last=U, kappa=kappa_estimated,
                           weight_nu=w_nu, tr_radius=tr_radius)

    ### ADD SOLUTION TO THE LIST
    state_history_list.append(X[:, 0].copy())
    state_history_list[-1][3] = s0  # put s0 back when saving
    control_history_list.append(U[:, 0].copy())
    curvature_history_list.append(kappa_estimated[0].copy())

    '''
        --------------- HANDLE CONSTRAINTS HERE -----------------
        * check current s0 --> how much we have proceeded on the current path
        * check distance to obstacle if it is in the planning range 
            * If it is in the planning range, define the constraint index ey[k] ><= 0.5
            * But obstacle avoidance constraints are unusual than the standard constraint (eymin < ey < eymax) 
            equations which requires formulating the constraint out of region (ey <=eymin or ey>= eymax)  
    '''

    if len(m.obs_loc):
        m.set_obstacle(target_distance=target_distance)

    ## Convergence
    converged = False
    t0_it = time()

    for it in range(iterations):
        error = problem.solve(verbose=verbose_solver, solver=solver, max_iters=200, warm_start= False)
        print(format_line('Solver Error', error))

        if error in ['optimal', 'optimal_inaccurate']:

            # get solution
            new_X = problem.get_variable('X')
            new_U = problem.get_variable('U')

            # X_nl = integrator.integrate_nonlinear_full(x0, new_U, kappa_estimated)
            X_nl = integrator.integrate_nonlinear_piecewise(X, new_U, kappa_estimated)

            ## Make the distance travelled relative wrt the start point
            new_X[3, :] = new_X[3, :] - new_X[3, 0]
            X_nl[3, :] = X_nl[3, :] - X_nl[3, 0]

            linear_cost_dynamics = np.linalg.norm(problem.get_variable('nu'), 1)
            nonlinear_cost_dynamics = np.linalg.norm(new_X - X_nl, 1)

            linear_cost_constraints = m.get_linear_cost_constraints()  # m.get_linear_cost()
            nonlinear_cost_constraints = m.get_nonlinear_cost_constraints(X_nl, new_U, kappa_estimated)  #
            # m.get_nonlinear_cost(X=new_X, U=new_U)

            linear_cost = linear_cost_dynamics + linear_cost_constraints  # J
            nonlinear_cost = nonlinear_cost_dynamics + nonlinear_cost_constraints  # L

            if last_nonlinear_cost is None:
                last_nonlinear_cost = nonlinear_cost
                X[:, :-1] = new_X[:, 1:]
                X[:, -1] = new_X[:, -1]

                U[:, :-1] = new_U[:, 1:]
                U[:, -1] = new_U[:, -1]

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

                s0 = s0 + new_X[3, 1] - new_X[3, 0]
                m.update_last_station_index(s0)
                break

            else:
                rho = actual_change / predicted_change
                if rho < rho_0:
                    # reject solution
                    tr_radius /= alpha
                    print(f'Trust region too large. Solving again with radius={tr_radius}')

                    if tr_radius < 1e-1:
                        a = 1

                else:
                    # accept solution
                    last_nonlinear_cost = nonlinear_cost
                    converged = True

                    X[:, :-1] = X_nl[:, 1:]
                    X[:, -1] = X_nl[:, -1]

                    U[:, :-1] = new_U[:, 1:]
                    U[:, -1] = new_U[:, -1]

                    s0 = s0 + new_X[3, 1] - new_X[3, 0]
                    m.update_last_station_index(s0)

                    print('Solution accepted.')

                    if rho < rho_1:
                        print('Decreasing radius.')
                        tr_radius /= alpha


                    elif rho >= rho_2:
                        print('Increasing radius.')
                        tr_radius *= beta

                    break

            problem.set_parameters(tr_radius=tr_radius)

            if converged:
                break

            print('-' * 50)

        else:
            break
            # print('It is not optimal, recomputing by the new trust region \n')
            # print(f'Trust region too large. Solving again with radius={tr_radius}')
            #
            # # tr_radius /= alpha
            # tr_radius *= beta
            #
            # if tr_radius < 1e-2:
            #     break

            # problem.set_parameters(tr_radius=tr_radius)

    if error == 'infeasible':
        print(' \n\n Infeasible Solution Halted')
        break

    print('')
    print(format_line('Time for iteration', time() - t0_it, 's'))
    print('')

logs_pickle['States'] = np.vstack(state_history_list).transpose()
logs_pickle['Controls'] = np.vstack(control_history_list).transpose()
logs_pickle['Optimization_Params'] = optimization_history
logs_pickle['Current_Map'] = m.current_reference_map
logs_pickle['kappa'] = np.vstack(curvature_history_list).transpose()

logs_pickle['obstacle_locs'] = m.get_obstacle_locs()

# file_name = './Logs/logs_pickle_tire_no_obstacle.pickle'
# file_name = './Logs/logs_pickle_tire_obstacle.pickle'
with open(file_name, 'wb') as handle:
    pickle.dump(logs_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
