import torch
import numpy as np
import time
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
Y, X = np.mgrid[-1.3:1.3:0.0005, -2:1:0.0005]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y)

def lorenz(x, y, z, sigma=16.0, rho=40.0, beta=4.0):
    # lorenz equations
    x_dot = sigma * (y - x)
    y_dot = rho * x - y - x * z
    z_dot = x * y - beta * z

    return x_dot, y_dot, z_dot

# setting dt and number of steps for appropriate clarity
dt=0.01
num_steps = 10000

# initialise empty array and initial values for a clean spiral
xs, ys, zs = np.empty(num_steps + 1), np.empty(num_steps + 1), np.empty(num_steps + 1)
xs[0], ys[0], zs[0] = 0.0, 1.0, 1.05

# iterate over array calculating what the new value should be in each case based on previous values
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i+1] = xs[i] + x_dot * dt
    ys[i+1] = ys[i] + y_dot * dt
    zs[i+1] = zs[i] + z_dot * dt
    # each next point relies on the previous point of all 3 so if you were to do these actions in parallel 
    # they would just have to wait for each other anyway thus parallelism is pointless and introduces 
    # unnecassary convolution to the code making it harder to read


# do a 3d projection of the plot making it pretty with some colour choices
fig = plt.subplot(projection="3d")
fig.plot(xs, ys, zs, lw=0.5, color="blue")
fig.scatter(xs, ys, zs, lw=0.1, alpha=0.1, color="yellow")

plt.show()