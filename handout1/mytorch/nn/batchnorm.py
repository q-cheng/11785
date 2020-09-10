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
            u_b = x.batch_mean()
            x_sub_u_b = x - u_b
            x_sub_u_b_square = x_sub_u_b.square()
            em_b_square = x_sub_u_b_square.data / (self.num_features - 1)
            s_b_square = x_sub_u_b_square.batch_mean()
            self.running_mean.data = (1 - self.momentum.data) * self.running_mean.data + self.momentum.data * u_b.data
            self.running_var.data = (1 - self.momentum.data) * self.running_var.data + self.momentum.data * em_b_square
        else:
            u_b = self.running_mean
            s_b_square = self.running_var

        x_norm = (x - u_b) / (s_b_square + self.eps).sqrt()
        y_i = self.gamma * x_norm + self.beta
        return y_i

