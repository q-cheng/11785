import numpy as np


def backward(grad_fn, grad_of_outputs):
    """Recursive DFS that traverses comp graph, handing back gradients as it goes.
    Args:
        grad_fn (BackwardFunction or AccumulateGrad): Current node type from
                                                      parent's `.next_functions`
        grad_of_output (Tensor): Gradient of the final node w.r.t. current output
    Returns:
        No return statement needed.
    """

    # 1) Calculate gradients of final node w.r.t. to the current nodes parents
    if not grad_fn:
        return

    if grad_fn.function_name == "AccumulateGrad" and grad_of_outputs.data.shape != grad_fn.variable.shape:
        index_list = []

        for cur_idx in range(len(grad_of_outputs.data.shape)):
            if cur_idx > len(grad_fn.variable.shape) - 1:
                index_list.append((0, len(grad_of_outputs.data.shape) - 1 - cur_idx))
            elif grad_fn.variable.shape[len(grad_fn.variable.shape) - 1 - cur_idx] == 1:
                index_list.append((1, len(grad_of_outputs.data.shape) - 1 - cur_idx))

        for flag, idx in index_list:
            if flag == 0:
                grad_of_outputs.data = np.sum(grad_of_outputs.data, axis=idx)
            else:
                grad_of_outputs.data = np.sum(grad_of_outputs.data, axis=idx, keepdims=True)

    new_grads = grad_fn.apply(grad_of_outputs)

    # 2) Pass gradient onto current node's beloved parents (recursive DFS)
    for idx in range(len(grad_fn.next_functions)):
        if not grad_fn.next_functions[idx] or not new_grads[idx]:
            continue
        backward(grad_fn.next_functions[idx], new_grads[idx])


class Function:
    """Superclass for linking nodes to the computational graph.
    Operations in `functional.py` should inherit from this"""

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError("All subclasses must implement forward")

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("All subclasses must implement backward")

    @classmethod
    def apply(cls, *args):
        """Runs forward of subclass and links node to the comp graph.
        Args:
            cls (subclass of Function): (NOTE: Don't provide this;
                                               already provided by `@classmethod`)
                                        Current function, such as Add, Sub, etc.
            args (tuple): arguments for the subclass's `.forward()`.
                  (google "python asterisk arg")
        Returns:
            Tensor: Output tensor from operation that stores the current node.
        """
        # Creates BackwardFunction obj representing the current node
        backward_function = BackwardFunction(cls)

        # Run subclass's forward with context manager and operation input args
        output_tensor = cls.forward(backward_function.ctx, *args)

        # TODO: Complete code below
        # 1) For each parent tensor in args, add their node to `backward_function.next_functions`
        #    Note: Parents may/may not already have their own nodes. How do we handle this?
        #    Note: Parents may not need to be connected to the comp graph. How do we handle this?
        #    (see Appendix A.1 for hints)
        for tensor in args:
            if tensor.requires_grad and tensor.is_leaf:
                backward_function.next_functions.append(AccumulateGrad(tensor))
            elif tensor.is_leaf is False and tensor.requires_grad:
                backward_function.next_functions.append(tensor.grad_fn)
            else:
                # Make sure position is right.
                backward_function.next_functions.append(None)
            if cls.__name__ == 'Reshape':
                break

        # 2) Store current node in output tensor (see `tensor.py` for ideas)
        output_tensor.grad_fn = backward_function

        return output_tensor


class AccumulateGrad:
    """Wrapper around tensor, representing node where gradient must be accumulated
    Args:
        tensor (Tensor): The tensor where the gradients are accumulated in `.grad`
    """

    def __init__(self, tensor):
        self.variable = tensor  # tensor to wrap around
        self.next_functions = []  # node types of current node's children (generally empty)

        self.function_name = "AccumulateGrad"  # just for convenience lol

    def apply(self, arg):
        """Accumulates gradient provided.
        (Hint: Notice name of function is the same as BackwardFunction's `.apply()`)
        Args:
            arg (Tensor): Gradient to accumulate
        """
        # if no grad stored yet, initialize. otherwise +=
        if self.variable.grad is None:
            self.variable.grad = arg
        else:
            self.variable.grad += arg

        # Some tests to make sure valid grads were stored.
        shape = self.variable.shape
        grad_shape = self.variable.grad.shape
        # print('var:', self.variable.data, '\ngrad:', self.variable.grad)
        assert shape == grad_shape, (shape, grad_shape)


class ContextManager:
    """Used to pass variables between a function's `.forward()` and `.backward()`.
    (Argument "ctx" in these functions)

    To store a tensor:
    # >>> ctx.save_for_backward(<tensors>, <to>, <store>)

    To store other variables (like integers):
    # >>> ctx.<some_name> = <some_variable>
    """

    def __init__(self):
        self.saved_tensors = []  # list that TENSORS get stored in

    def save_for_backward(self, *args):
        """Saves TENSORS only
        See example above for storing other data types.
        Args:
            args (Tensor(s)): Tensors to store
        """
        for arg in args:
            # Raises error if arg is not tensor (i warned you)
            if type(arg).__name__ != "Tensor":
                raise Exception(
                    "Got type {} of object {}. \nOnly Tensors should be saved in save_for_backward. For saving constants, just save directly as a new attribute.".format(
                        type(arg), arg))

            self.saved_tensors.append(arg.copy())


class BackwardFunction:
    """Wrapper around tensor, representing node where gradient must be passed
    Args:
        cls (subclass of Function): Operation being run. Don't worry about this;
                                    already handled in Function.apply()
    """

    def __init__(self, cls):
        self.ctx = ContextManager()  # Just in case args need to be passed (see above)
        self._forward_cls = cls

        # Node types of children, populated in `Function.apply`
        self.next_functions = []

        # The name of the operation as a string (for convenience)
        self.function_name = cls.__name__

    def apply(self, *args):
        """Generates gradient by running the operation's `.backward()`.
        Args:
            args: Args for the operation's `.backward()`
        Returns:
            Tensor: gradient of parent's output w.r.t. current output
        """
        # Note that we've already provided the ContextManager
        return self._forward_cls.backward(self.ctx, *args)
