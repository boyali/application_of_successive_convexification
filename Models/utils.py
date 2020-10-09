import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def format_line(name, value, unit=''):
    """
    Formats a line e.g.
    {Name:}           {value}{unit}
    """
    name += ':'
    if isinstance(value, (float, np.ndarray)):
        value = f'{value:{0}.{4}}'

    return f'{name.ljust(40)}{value}{unit}'


def plot_states(X):
    # xw, yw, ψ, ψdot, beta, s, ey, epsi, Vx, delta

    plt.plot(X[0, :], X[1, :])
    plt.title('Global Path')
    plt.xlabel('Xw')
    plt.ylabel('Yw')

    plt.tight_layout()
    plt.show()


def save(state_history_list, control_history_list, state_history, control_history):
    save_loc = './Logs/state_history.csv'
    dfstates = pd.DataFrame.from_records(state_history_list, columns=state_history._fields)
    dfstates.to_csv(save_loc)

    save_loc = './Logs/control_history.csv'
    dfcontrols = pd.DataFrame.from_records(control_history_list, columns=control_history._fields)
    dfcontrols.to_csv(save_loc)


def wrapToPi(angle):
    '''
        wraps to angle -pi< angle < pi
        :param angle: in radians
        :return: wrapped angle
    '''
    temp = np.exp(1j * angle)
    wrapped_angle = np.angle(temp)

    return wrapped_angle
