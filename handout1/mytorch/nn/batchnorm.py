from mytorch.tensor import Tensor
import numpy as np
from mytorch.nn.module import Module
import functools

class BatchNorm1d(Module):
    """Batch Normalization Layer

    Args:
        num_features (int): # dims in input and output
        eps (float): value added to denominator for numerical stability
                     (not important for now)
        momentum (float): value used for running mean and var computation

    Inherits from:
        Module (mytorch.nn.module.Module)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features

        self.eps = Tensor(np.array([eps]))
        self.momentum = Tensor(np.array([momentum]))

        # To make the final output affine
        self.gamma = Tensor(np.ones((self.num_features,)), requires_grad=True, is_parameter=True)
        self.beta = Tensor(np.zeros((self.num_features,)), requires_grad=True, is_parameter=True)

        # Running mean and var
        self.running_mean = Tensor(np.zeros(self.num_features,), requires_grad=False, is_parameter=False)
        self.running_var = Tensor(np.ones(self.num_features,), requires_grad=False, is_parameter=False)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, num_features)
        Returns:
            Tensor: (batch_size, num_features)
        """
        if self.is_train:
            u_b = x.data.mean()
            x_sub_u_b = x.data - np.ones(x.data.shape) * u_b
            x_sub_u_b_square = np.square(x_sub_u_b)
            em_b_square = np.sum(x_sub_u_b_square) / (x_sub_u_b.size - 1)
            self.running_mean.data = (1 - self.momentum.data) * self.running_mean.data + self.momentum.data * u_b
            self.running_var.data = (1 - self.momentum.data) * self.running_var.data + self.momentum.data * em_b_square
            s_b_square = np.sum(x_sub_u_b_square) / x_sub_u_b.size
            x_norm = x_sub_u_b / (np.sqrt(s_b_square + self.eps.data))
            y_i = self.gamma.data * x_norm + self.beta.data
            # print('y_i:', y_i)
            return Tensor(y_i)
        else:
            x_sub_u_b = x.data - np.ones(x.data.shape) * self.running_mean.data
            x_norm = x_sub_u_b / (np.sqrt(self.running_var.data + self.eps.data))
            y_i = self.gamma.data * x_norm + self.beta.data
            # print('y_i:', y_i)
            return Tensor(y_i)
