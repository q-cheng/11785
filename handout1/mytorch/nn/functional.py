import numpy as np

import mytorch.tensor as tensor
from mytorch.autograd_engine import Function


class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.T)


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data)


"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
        if not check(a.data, b.data):
            raise Exception("Both args must have valid sizes: {}, {}".format(a.shape, b.shape))
        # if a.data.shape != b.data.shape:
        #     raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        if not check(a.data, b.data):
            raise Exception("Both args must have valid sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = -1 * np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check whether these two tensor could multiply.
        if not check(a.data, b.data):
            raise Exception("Both args must have valid sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data * b.data, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = b.data * grad_output.data * np.ones(a.shape)
        grad_b = a.data * grad_output.data * np.ones(b.shape)

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check whether these two tensor could divide.
        if not check(a.data, b.data):
            raise Exception("Both args must have valid sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data / b.data, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = np.ones(a.shape) * grad_output.data / b.data
        grad_b = -1 * np.ones(b.shape) * grad_output.data * a.data / (b.data * b.data)

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    return predicted.get_loss(target)
    # batch_size, num_classes = predicted.shape
    # x_mean = np.mean(predicted.data, axis=1, keepdims=True) * np.ones(predicted.data.shape)
    # softmax = np.exp(predicted.data) / np.sum(np.exp(predicted.data), axis=1, keepdims=True)
    # log_softmax = predicted.data - (x_mean + np.log(np.sum(np.exp(predicted.data - x_mean), axis=1, keepdims=True)))
    # loss_sum = 0
    # back_grad = np.ones(predicted.shape)
    # for i, j in enumerate(target.data):
    #     loss_sum += log_softmax[i][j]
    #     back_grad[i][j] = softmax[i][j] - 1
    # nll_loss = -1 * loss_sum / batch_size
    # # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    # #      However, if you'd prefer to implement a Function subclass you're free to.
    # #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.
    #
    # # Tip 2: Remember to divide the loss by batch_size; this is equivalent
    # #        to reduction='mean' in PyTorch's nn.CrossEntropyLoss
    # return_tensor = tensor.Tensor(nll_loss)
    # return_tensor.grad_fn = predicted.grad_fn
    # return return_tensor


class Loss(Function):
    @staticmethod
    def forward(ctx, predicted, target):
        if not (type(predicted).__name__ == 'Tensor' and type(target).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(predicted).__name__, type(target).__name__))
        batch_size, num_classes = predicted.shape
        x_max = np.max(predicted.data, axis=1, keepdims=True) * np.ones(predicted.data.shape)
        log_softmax = predicted.data - (x_max + np.log(np.sum(np.exp(predicted.data - x_max), axis=1, keepdims=True)))
        mask = to_one_hot(target, num_classes)
        back_grad = np.exp(log_softmax)
        loss_sum = np.sum(mask.data * log_softmax)
        back_grad = back_grad - mask.data
        back_grad = back_grad / batch_size
        back_grad_tensor = tensor.Tensor(back_grad)
        nll_loss = -1 * loss_sum / batch_size
        ctx.save_for_backward(back_grad_tensor,)
        requires_grad = predicted.requires_grad or target.grad
        c = tensor.Tensor(nll_loss, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        back_grad_tensor, = ctx.saved_tensors
        return back_grad_tensor,


def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    # >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]
     
    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad=True)


def check(a, b):
    """
    Check if two numpy arrays could be broadcast to same shape.
    Args:
        a: matrix a.
        b: matrix b.

    Returns: True if it could be done, otherwise False.

    """
    if a.shape == b.shape:
        return True
    else:
        if len(a.shape) > len(b.shape):
            new_b = [1 for _ in range(len(a.shape))]
            for idx in range(len(b.shape)):
                new_b[len(new_b) - 1 - idx] = b.shape[len(b.shape) - 1 - idx]
            new_a = a.shape
        else:
            new_a = [1 for _ in range(len(b.shape))]
            for idx in range(len(b.shape)):
                new_a[len(new_a) - 1 - idx] = b.shape[len(b.shape) - 1 - idx]
            new_b = b.shape
        for i in range(len(new_a)):
            if new_a[i] == new_b[i] or min(new_a[i], new_b[i]) == 1:
                continue
            else:
                return False
        return True


class ReLu(Function):
    @staticmethod
    def forward(ctx, x):
        if not type(x).__name__ == 'Tensor':
            raise Exception("args must be Tensors.")
        requires_grad = x.requires_grad
        ctx.save_for_backward(x, )
        c = tensor.Tensor(np.where(x.data < 0, 0, x.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        x, = ctx.saved_tensors
        grad_x = np.where(x.data >= 0, 1, 0) * grad_output.data
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_x),


class Matmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check whether these two tensor could divide.
        if not check(a.data, b.data):
            raise Exception("Both args must have valid sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.dot(a.data, b.data.T), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors
        # print('grad:', grad_output.data, '\na:', a, '\nb:', b)
        # calculate gradient of output w.r.t. each input
        grad_a = np.dot(grad_output.data, b.data)
        grad_b = np.dot(grad_output.data.T, a.data)

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class BatchMean(Function):
    @staticmethod
    def forward(ctx, a):
        if not (type(a).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}".format(type(a).__name__))
        ctx.save_for_backward(a, )
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.mean(a.data, axis=tuple(range(a.data.ndim - 1))), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        # Sum the grad_output to make sure the same batch has the same gradient, send it back.
        grad_a = np.ones(a.shape) * np.mean(grad_output.data, axis=tuple(range(grad_output.data.ndim - 1)))
        return tensor.Tensor(grad_a),


class Square(Function):
    @staticmethod
    def forward(ctx, a):
        if not (type(a).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}".format(type(a).__name__))
        ctx.save_for_backward(a, )
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.square(a.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        grad_a = np.ones(a.shape) * grad_output.data * 2 * a.data
        return tensor.Tensor(grad_a),


class SquareRoot(Function):
    @staticmethod
    def forward(ctx, a):
        if not (type(a).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}".format(type(a).__name__))
        ctx.save_for_backward(a, )
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.sqrt(a.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        # Divide back_grad by N, where n in batch_size.
        grad_a = 0.5 * grad_output.data / np.sqrt(a.data)
        return tensor.Tensor(grad_a),


# class BatchSum(Function):
#     @staticmethod
#     def forward(ctx, a):
#         if not (type(a).__name__ == 'Tensor'):
#             raise Exception("Both args must be Tensors: {}".format(type(a).__name__))
#         ctx.save_for_backward(a, )
#         requires_grad = a.requires_grad
#         c = tensor.Tensor(np.sum(a.data, axis=0), requires_grad=requires_grad,
#                           is_leaf=not requires_grad)
#         return c
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         a, = ctx.saved_tensors
#         # Divide back_grad by N, where n in batch_size.
#         grad_a = np.ones(a.shape) * grad_output.data
#         return tensor.Tensor(grad_a),


# class BatchNormTrain(Function):
#     @staticmethod
#     def forward(ctx, x):
#         if not (type(a).__name__ == 'Tensor'):
#             raise Exception("Both args must be Tensors: {}".format(type(a).__name__))
#         ctx.save_for_backward(x, )
#         requires_grad = x.requires_grad
#         u_b = np.mean(x.data, axis=tuple(range(x.data.ndim - 1)))
#         s_b_square = np.var(x.data, axis=tuple(range(x.data.ndim - 1)))
#         c = tensor.Tensor(np.sum(a.data, axis=0), requires_grad=requires_grad,
#                           is_leaf=not requires_grad)
#         return c
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         x, = ctx.saved_tensors
#         # Divide back_grad by N, where n in batch_size.
#         grad_x = np.ones(x.shape) * grad_output.data
#         return tensor.Tensor(grad_x),
#
#
# class BatchNormTrainNotTrain:
#     @staticmethod
#     def forward(ctx, x):
#         if not (type(a).__name__ == 'Tensor'):
#             raise Exception("Both args must be Tensors: {}".format(type(a).__name__))
#         ctx.save_for_backward(x, )
#         requires_grad = x.requires_grad
#         u_b = np.mean(x.data, axis=tuple(range(x.data.ndim - 1)))
#         s_b_square = np.var(x.data, axis=tuple(range(x.data.ndim - 1)))
#         c = tensor.Tensor(np.sum(a.data, axis=0), requires_grad=requires_grad,
#                           is_leaf=not requires_grad)
#         return c
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         x, = ctx.saved_tensors
#         # Divide back_grad by N, where n in batch_size.
#         grad_x = np.ones(x.shape) * grad_output.data
#         return tensor.Tensor(grad_x),


# def resize_grad(input_tensor, target_tensor):
#     index_list = []
#
#     for cur_idx in range(len(input_tensor.data.shape)):
#         if cur_idx > len(target_tensor.data.shape) - 1:
#             index_list.append((0, len(input_tensor.data.shape) - 1 - cur_idx))
#         elif target_tensor.data.shape[len(target_tensor.data.shape) - 1 - cur_idx] == 1:
#             index_list.append((1, len(input_tensor.data.shape) - 1 - cur_idx))
#
#     for flag, idx in index_list:
#         if flag == 0:
#             input_tensor.data = np.sum(input_tensor.data, axis=idx)
#         else:
#             input_tensor.data = np.sum(input_tensor.data, axis=idx, keepdims=True)
#     return
