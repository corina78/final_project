import numpy as np
import copy

def initialize_parameters(units_in_layer):
    """
    Applies He initialization

    Args:
    units_in_layer -- python list containing the dimensions of each layer in the network

    Returns:
    parameters --
    Wl -- weight matrix of shape (units_in_layer[l], units_in_layer[l-1])
    bl -- bias vector of shape (units_in_layer[l], 1)
    """

    np.random.seed(45)
    parameters = {}
    L = len(units_in_layer)  # number of layers in the network

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(units_in_layer[l], units_in_layer[l - 1]) * np.sqrt(2. / units_in_layer[l - 1])
        parameters['b' + str(l)] = np.zeros((units_in_layer[l], 1))

        # Save the weights and biases to files
    for l in range(1, L):
        np.savetxt(f"W{l}.txt", parameters['W' + str(l)])
        np.savetxt(f"b{l}.txt", parameters['b' + str(l)])

    return parameters

def relu(Z):
    """
    Compute the ReLU activation for an input array Z.

    Arguments:
    Z -- Input array, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    """
    A = np.maximum(0, Z)
    return A


def softmax(Z):
    """
    Compute the softmax activation for an input array Z.

    Arguments:
    Z -- Input array, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    """
    # Ensure numerical stability by subtracting the maximum value in Z from each entry in Z
    shiftZ = Z - np.max(Z, axis=0, keepdims=True)
    exps = np.exp(shiftZ)
    A = exps / np.sum(exps, axis=0, keepdims=True)

    return A


def linear_forward(A, W, b):
    """
    Implement pre activations of forward part.

    Args:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- pre-activation parameter ready to by processed by the selected activation.
    a cached tuple (A, W, b) ready to compute backward propagation.
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer
    W -- weights matrix
    b -- bias vector
    activation --  whether it will be  softmax (last layer) or relu (hidden layers)

    Returns:
    A -- the output of the activation function
    a cached tuple that contains "linear_cache" and "activation_cache"
    """

    if activation == "softmax":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z), Z


    elif activation == "relu":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = np.maximum(0, Z), Z

    cache = (linear_cache, activation_cache)

    # To save the object
    with open('cache_linear_act_forward.pkl', 'wb') as f:
        pickle.dump(cache, f)

    return A, cache


def Model_forward(X, parameters):
    """
    Implement forward propagation for the all layers

    Arguments:
    X -- data, numpy array of shape (num_features, num_examples)
    parameters -- output of initialize_parameters()

    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A

        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters['b' + str(l)],
                                             activation="relu")
        caches.append(cache)


    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation="softmax")
    caches.append(cache)


    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cross entropy cost function for classification.

    Arguments:
    AL -- probability vector, shape (1, number of examples)
    Y -- true "label" vector shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    epsilon = 1e-15
    cost = -1 / m * np.sum(Y * np.log(AL + epsilon))

    cost = np.squeeze(cost)
    print("cost at current iteration is:", cost)

    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer l

    Parameters:
    dZ -- Derivative of the cost in relation to the linear outcome of the present layer l.
    cache -- tuple holding values (A_prev, W, b) derived from the forward propagation of the current layer.

    Outputs:
    dA_prev -- Derivative of the cost with respect to the activation (from the preceding layer l-1).
    dW -- Derivative of the cost concerning W (of the current layer l).
    db -- Derivative of the cost concerning b (of the current layer l).
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, Y=None):
   """Parameters:
    dA -- post-activation gradient for the existing layer l.
    cache -- tuple containing values (linear_cache, activation_cache).
    activation -- the type of activation utilized in this layer.

    Outputs:
    dA_prev -- Derivative of the cost related to the activation (from the prior layer l-1), has the same shape as A_prev.
    dW -- Derivative of the cost concerning W (of the current layer l), has the same shape as W.
    db -- Derivative of the cost concerning b (of the current layer l), has the same shape as b.
   """
   linear_cache, activation_cache = cache
   Z = activation_cache

   if activation == "relu":

        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

   elif activation == "softmax":
        if np.isnan(Z).any():
            raise ValueError("NaN detected in Z")

        Z_max = np.max(Z, axis=0, keepdims=True)
        if np.isnan(Z_max).any():
            raise ValueError("NaN detected in Z_max")

        shifted_Z = Z - Z_max
        if np.isnan(shifted_Z).any():
            raise ValueError("NaN introduced after subtraction")

        s = np.exp(shifted_Z) / np.sum(np.exp(shifted_Z), axis=0)
        if np.isnan(s).any():
            raise ValueError("NaN introduced in softmax")

        dZ = s - Y
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

   return dA_prev, dW, db

def Model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector
    caches -- list of caches containing linear activations forward.

    Returns:
    grads -- A dictionary with the gradients
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Setting up an epsilon value for numerical stability
    epsilon = 1e-15
    dAL = -(np.divide(Y, AL + epsilon))

    # Lth layer (SOFTMAX -> LINEAR) gradients.

    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation="softmax", Y=Y)
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp


    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent.

    Arguments:
    params -- python dictionary containing learnt parameters
    grads -- python dictionary containing gradients

    Returns:
    parameters -- python dictionary containing updated parameters
    """
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters
