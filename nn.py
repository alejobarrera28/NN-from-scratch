import random


class Value:
    """Stores a scalar value and its gradient for autograd."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data  # stored numeric value
        self.grad = 0  # gradient initialized to zero
        self._backward = lambda: None  # function to compute local gradients
        self._prev = set(_children)  # set of parent Value nodes
        self._op = _op  # operation name (for debugging)

    def __add__(self, other):
        operand = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + operand.data, (self, operand), "+")

        def _backward():
            # gradient of sum: dL/dx = upstream_grad, same for both inputs
            self.grad += output.grad
            operand.grad += output.grad

        output._backward = _backward
        return output

    def __mul__(self, other):
        operand = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * operand.data, (self, operand), "*")

        def _backward():
            # product rule: dL/dx = other.data * upstream_grad
            self.grad += operand.data * output.grad
            operand.grad += self.data * output.grad

        output._backward = _backward
        return output

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only int/float powers supported"
        output = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward():
            # power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += exponent * (self.data ** (exponent - 1)) * output.grad

        output._backward = _backward
        return output

    def relu(self):
        # ReLU activation: max(0, x)
        output = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            # gradient flows only if input was positive
            self.grad += (output.data > 0) * output.grad

        output._backward = _backward
        return output

    def backward(self):
        """Compute gradients for all nodes in the computation graph."""
        topo_order = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topo(parent)
                topo_order.append(node)

        build_topo(self)

        # initialize gradient of output node
        self.grad = 1
        # traverse in reverse topological order
        for node in reversed(topo_order):
            node._backward()

    # operator overloads for convenience
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


class Module:
    """Base class for all neural network modules."""

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    """Single neuron performing a weighted sum and optional ReLU."""

    def __init__(self, input_size, use_nonlinearity=True):
        # initialize weights and bias
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(input_size)]
        self.bias = Value(0)
        self.use_nonlinearity = use_nonlinearity

    def __call__(self, inputs):
        # weighted sum: w1*x1 + w2*x2 + ... + bias
        linear_sum = sum((w * x for w, x in zip(self.weights, inputs)), self.bias)
        # apply ReLU if specified
        return linear_sum.relu() if self.use_nonlinearity else linear_sum

    def parameters(self):
        return self.weights + [self.bias]

    def __repr__(self):
        kind = "ReLU" if self.use_nonlinearity else "Linear"
        return f"{kind}Neuron(num_inputs={len(self.weights)})"


class Layer(Module):
    """Layer containing multiple neurons."""

    def __init__(self, input_size, output_size, **kwargs):
        self.neurons = [Neuron(input_size, **kwargs) for _ in range(output_size)]

    def __call__(self, inputs):
        outputs = [neuron(inputs) for neuron in self.neurons]
        # return single value if only one neuron
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __repr__(self):
        return f"Layer(neurons=[{', '.join(str(n) for n in self.neurons)}])"


class MLP(Module):
    """Multi-layer perceptron: sequence of layers."""

    def __init__(self, input_size, layer_sizes):
        sizes = [input_size] + layer_sizes
        # create layers: nonlinearity except last layer
        self.layers = [
            Layer(sizes[i], sizes[i + 1], use_nonlinearity=(i < len(layer_sizes) - 1))
            for i in range(len(layer_sizes))
        ]

    def __call__(self, inputs):
        # forward pass through all layers
        result = inputs
        for layer in self.layers:
            result = layer(result)
        return result

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

    def __repr__(self):
        return f"MLP(layers=[{', '.join(str(layer) for layer in self.layers)}])"
