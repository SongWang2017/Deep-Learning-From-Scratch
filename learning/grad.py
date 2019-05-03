import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        temp_val = x[idx]

        x[idx] = temp_val + h
        fxh1 = f(x)

        x[idx] = temp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = temp_val

    return grad

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    for step in range(step_num):
        grad = numerical_gradient(f, x)
        x -= grad * lr

    return x

