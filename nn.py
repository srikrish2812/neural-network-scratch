"""
The main code for the back propagation assignment. See README.md for details.
"""
import math
from typing import List

import numpy as np
#### ADDITIONAL IMPORTS HERE, IF DESIRED ####
import scipy.special


# NOTE: In the docstrings, "UDL" refers to the book Pierce (2023),
#       "Understanding Deep learning".


class SimpleNetwork:
    """A simple feedforward network where all units have sigmoid activation,
    including at final layer (output) of model, and there are no bias term
    parameters, only layer weights. Input, output and weight matrices follow
    denominator layout format (same as UDL).
    """

    @classmethod
    def random(cls, *layer_units: int):
        """Creates a feedforward neural network with the given number of units
        for each layer (including input (first) and output (last) layers).

        Example: create a network, net, with input layer of 3 units, a first
        hidden layer with 4 hidden units, a second hidden layer with 5 hidden
        units, and an output layer with 2 units:
            net = SimpleNetwork.random(3, 4, 5, 2)

        :param layer_units: Number of units for each layer
        :return: the neural network
        """

        def uniform(n_in, n_out):
            epsilon = math.sqrt(6) / math.sqrt(n_in + n_out)
            return np.random.uniform(-epsilon, +epsilon, size=(n_out, n_in))

        pairs = zip(layer_units, layer_units[1:])

        return cls(*[uniform(i, o) for i, o in pairs])

    def __init__(self, *layer_weights: np.ndarray):
        """Creates a neural network from a list of weight matrices.
        The weights specify linear transformations from one layer to the next, so
        the number of layers is equal to one more than the number of layer_weights
        weight matrices.

        :param layer_weights: A list of weight matrices
        """
        #### YOUR CODE HERE ####
        self.layer_weights = list(layer_weights) # initialize the list of weights


    def predict(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix.

        Each unit's output should be calculated by taking a weighted sum of its
        inputs (using the appropriate weight matrix) and passing the result of
        that sum through a (logistic) sigmoid activation function. This includes
        at the final layer of the network.

        (This network does not include bias parameters.)

        :param input_matrix: The matrix of inputs to the network, where each
        column in the matrix represents an input instance for which the neural
        network should make a prediction.
        :return: A matrix of predictions, where each column is the predicted
        outputs - each in the value range (0, 1) - for the corresponding column
        in the input matrix.
        """
        #### YOUR CODE HERE ####
        activation_h = input_matrix
        for weight in self.layer_weights:
            # compute pre-activaions_f that is weighted sum of inputs
            pre_activation_f = np.matmul(weight, activation_h)
            # apply sigmoid function to compute activations_h
            activation_h = scipy.special.expit(pre_activation_f)
        return activation_h

    def predict_zero_one(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix, and converts the outputs to binary (0 or 1).

        Outputs will be converted to 0 if they are less than 0.5, and converted
        to 1 otherwise.

        :param input_matrix: The matrix of inputs to the network, where each
        column in the matrix represents an instance for which the neural network
        should make a prediction.
        :return: A matrix of predictions, where each column is the predicted
        outputs - each either 0 or 1 - for the corresponding column in the input
        matrix.
        """
        #### YOUR CODE HERE ####
        return np.where(self.predict(input_matrix) < 0.5, 0, 1)

    def gradients(self,
                  input_matrix: np.ndarray,
                  target_output_matrix: np.ndarray) -> List[np.ndarray]:
        """Performs backpropagation to calculate the gradients (derivatives of
        the loss with respect to the model weight parameters) for each of the
        weight matrices.

        This method first performs a pass of forward propagation through the
        network, where the pre-activations (f) and activations (h) of each
        layer are stored. (NOTE: this bookkeeping could be performed in
        self.predict(), if desired.)
        This method then applies the following procedure to calculate the
        gradients.

        In the following description, × is matrix multiplication, ⊙ is
        element-wise product, and ⊤ is matrix transpose. The acronym 'w.r.t.'
        is shorhand for "with respect to".

        First, calculate the derivative of the squared loss w.r.t. model's
        final layer, K, activations, Sig[f_K], and the target output matrix, y:

            dl_df[K] = (Sig[f_K] - y)^2

        Then for each layer k in the network, starting with the layer before
        the output layer and working back to the first layer (the input matrix),
        calculate the gradient for the corresponding weight matrix as follows.

        (1) Calculate the derivatives of the loss w.r.t. the weights at the
        layer, dl_dweights[layer] (i.e., the parameter gradients), using the
        derivative of the loss w.r.t. layer pre-activation, dl_df[layer], and
        the activation, h[layer].
        (UDL equation 7.22)
        NOTE: With multiple inputs, there will be one gradient per input
        instance, and these must be summed (element-wise across gradient per
        input) and the resulting summed gradient must be (element-wise) divided
        by the number of input instances. As discussed in class, the simultaneous
        outer product and sum across gradients can be achieved using numpy.matmul,
        leaving only the element-wise division by the number of input instances.
        NOTE: The gradient() method returns the list of gradients per layer,
        so you will need to store the computed gradient per layer in a List
        for return at the end. The order of the gradients should be in
        "forward" order (layer 0 first, layer 1 second, etc...).

        (2) Calculate the derivatives of the loss w.r.t. the activations,
        dl_dh[layer], from the transpose of the weights, weights[layer].⊤,
        and the derivatives of the next pre-activation, dl_df[layer].
        (the second part of the last line of UDL equation 7.24)

        (3) If the current layer is not the 0'th layer, then:
        Calculate the derivatives of the loss w.r.t. the pre-activation
        for the previous layer, dl_df[layer - 1]. This involves the derivative
        of the activation function (sigmoid), dh_df.
        (first part of the last line of UDL eq 7.24)

        :param input_matrix: The matrix of inputs to the network, where each
        column in the matrix represents an instance for which the neural network
        should make a prediction.
        :param target_output_matrix: A matrix of expected outputs, where each column
        is the expected outputs - each either 0 or 1 - for the corresponding column
        in the input matrix.
        :return: List of the gradient matrices, 1 for each weight matrix, in same
        order as the weight matrix layers of the network, from input to output.
        """
        #### YOUR CODE HERE ####
        # Forward pass and book-keeping
        activations_h = [input_matrix]
        pre_activations_f = []
        for weight in self.layer_weights:
            # compute pre-activations and store them
            pre_activation_f_each_layer = np.matmul(weight, activations_h[len(activations_h) - 1])
            pre_activations_f.append(pre_activation_f_each_layer)
            # compute activations and store them
            activation_h_each_layer = scipy.special.expit(pre_activation_f_each_layer)
            activations_h.append(activation_h_each_layer)
        # Backward pass
        num_layers = len(self.layer_weights)
        gradients = [] * num_layers
        num_input_instances = input_matrix.shape[1]
        # derivative of loss wrt final layer
        # dl_df[k] = 2 * (h[k] - y) * h[k] * (1-h[k])
        # h[k] = sigmoid(f[k-1]) => h'[k] = h[k] * (1-h[k])
        dl_df = (
            2 * (activations_h[-1] - target_output_matrix)
            * activations_h[-1]*(1-activations_h[-1])
        )
        for layer in range(num_layers-1,-1,-1):
            # 1. derivatives of loss wrt to weights
            # dl_dweights[k] = dl_df[k] * h[k].T
            dl_dweights = np.matmul(dl_df, activations_h[layer].T)/num_input_instances
            gradients.insert(0, dl_dweights) # since gradients must be in forward order
            # 2. derivative of loss wrt to activations
            # dl_dh[k] = dl_df[k] * omega[k].T
            dl_dh = np.matmul(self.layer_weights[layer].T, dl_df)
            if layer>0:
                # 3. compute derivative of loss wrt to pre-activation of previous layer
                # dl_df[k-1] = dl_dh[k] * h[k] * (1-h[k]) since h[k] = sigmoid(f[k-1])
                dl_df = dl_dh * activations_h[layer]*(1-activations_h[layer])
        return gradients

    def train(self,
              input_matrix: np.ndarray,
              target_output_matrix: np.ndarray,
              iterations: int = 10,
              learning_rate: float = 0.1) -> None:
        """Trains the neural network on an input matrix and an expected output
        matrix.

        Training should repeatedly (`iterations` times) calculate the gradients,
        and update the model by subtracting the learning rate times the
        gradients from the model weight matrices.

        :param input_matrix: The matrix of inputs to the network, where each
        column in the matrix represents an input instance for which the neural
        network should make a prediction
        :param target_output_matrix: A matrix of expected outputs, where each
        column is the expected outputs - each either 0 or 1 - for the corresponding row in
        the input instance in the input matrix.
        :param iterations: The number of gradient descent steps to take.
        :param learning_rate: The size of gradient descent steps to take, a
        number that the gradients should be multiplied by before updating the
        model weights.
        """
        #### YOUR CODE HERE ####
        for dummy in range(iterations):
            # compute gradients in each iteration
            gradients = self.gradients(input_matrix, target_output_matrix)
            for i, grad in enumerate(gradients):
                # apply gradient descent and update weights
                # omega[t+1] = omega[t] - alpha * dl_domega[t]
                self.layer_weights[i] = self.layer_weights[i] - learning_rate * grad
                