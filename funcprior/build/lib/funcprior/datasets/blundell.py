from ..imports import *

def true_blundell(xlim=(-1.5, 1.5), with_linear=True):
    xmin, xmax = xlim
    x = np.linspace(xmin, xmax, 1000)
    if with_linear:
        y = 0.3 * x + (0.3 * np.sin(2 * np.pi * x)) + (0.3 * np.sin(4 * np.pi * x))
    else:
        y = (0.3 * np.sin(2 * np.pi * x)) + (0.3 * np.sin(4 * np.pi * x))
    return x, y

def sample_blundell(to_keep=50, with_linear=True):
    mean = -0.8
    std = 0.15
    x_left = (np.random.randn(to_keep)*std + mean)
    
    mean = 0.8
    #std = 0.15
    std = 0.12
    x_right = (np.random.randn(to_keep)*std + mean)

    x = np.concatenate((x_left, x_right))
    if with_linear:
        y = 0.3 * x + (0.3 * np.sin(2 * np.pi * x)) + (0.3 * np.sin(4 * np.pi * x))
    else:
        y = (0.3 * np.sin(2 * np.pi * x)) + (0.3 * np.sin(4 * np.pi * x))
    noise = np.random.randn(y.shape[0]) * 0.05
    y = y + noise
    return x, y
