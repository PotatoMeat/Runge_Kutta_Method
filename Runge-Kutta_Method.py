import numpy as np

import matplotlib.pyplot as plt


def RungeKutta(X, Y, F, n, h):

    y_current = np.array(Y)
    x_current = np.array(X)

    Y_VAL = np.zeros((n, len(y_current)), dtype = float)
    X_VAL = np.zeros((n,), dtype = float)

    for i in range(0, n):

        k = np.zeros((len(y_current), 4), dtype=float)

        F_ANS = F(x_current, y_current)
        for j in range(0, len(y_current)):
            k[j][0] = F_ANS[j]

        F_ANS = F(x_current+h/3, y_current+h/3*k[:,0])
        for j in range(0, len(y_current)):
            k[j][1] = F_ANS[j]

        F_ANS = F(x_current+2*h/3, y_current-h/3*k[:,0]+h/3*k[:,1])
        for j in range(0, len(y_current)):
            k[j][2] = F_ANS[j]

        F_ANS = F(x_current+h, y_current+h*k[:,0]-h*k[:,1]+h*k[:,2])
        for j in range(0, len(y_current)):
            k[j][3] = F_ANS[j]

        Y_VAL[i] = y_current
        X_VAL[i] = x_current[0]

        for j in range(0, len(y_current)):
           y_current[j] = y_current[j] + h*sum(k[j])/8
        #x_current[:] = x_current[0]+h
        x_current[:]+=h


    return X_VAL, Y_VAL

def functions(x_values, y_values):
    functions_ansver = np.zeros((2, ), dtype = float)

    functions_ansver[0] = -y_values[1]
    functions_ansver[1] = y_values[0] - 0.1 *y_values[1]*(y_values[0]**2 - 1)

    return functions_ansver

def show_plot(x_values, y_values):

    for i in range(0, y_values.shape[1]):
        plt.plot(x_values, y_values[:,i])
    #plt.plot(x_values, y_values)
    plt.show()

x = np.array([0, 0], dtype=float)
y = np.array([0.5, 0.5], dtype=float)
h = 0.1
n = 100

x_val, y_val = RungeKutta(x, y, functions, n, h)

show_plot(x_val, y_val)
