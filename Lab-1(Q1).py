import torch
import numpy as np
import math
#print("PyTorch Version:", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

#load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

#transfer to the GPU device
x = x.to(device)
y = y.to(device)

xs = torch.tensor(X.astype(np.float32))
ys = torch.tensor(Y.astype(np.float32))

#Compute Gaussian
#z = torch.sin(x)+torch.sin(y)
z=xs*torch.cos(torch.tensor(1))+ys*torch.sin(torch.tensor(1))
z=torch.sin(2*np.pi*(z))

#z=torch.sin(x+y)

#z=torch.sin(torch.sqrt(x**2+y**2)+360)
#z = torch.exp(-(x**2+y**2)/2.0)*z

import matplotlib.pyplot as plt
plt.imshow(z.numpy())
plt.tight_layout()
plt.show()