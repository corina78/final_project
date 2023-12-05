import numpy as np
from os.path import join
import pickle
from loader import MnistDataloader
from preprocessing import prepare_data, one_hot_encode
from helper_functions import *


def davidson_quasi_newton_update(x_train_flattened, parameters, E, k, units_in_layer, epsilon=0.001):

    # Initialize variables only once
    J = parameters['J'] # initial parameters
    E0 = cost # initial cost before the first forward pass, just the cost related to the initial parameters
    grad_vector, structure_cache = flatten_gradients_for_jacobian(grads, units_in_layer) # grads after first backward pass
    # Check if sizes are compatible for matrix multiplication
    if grad_vector.shape[0] != 4015:
        raise ValueError(f"The gradient vector is expected to be of size 4015, but got size {grad_vector.shape[0]}")
    k0 = np.dot(J,grad_vector) # perform matrix multiplication to get k0, grad_vector after first backward pass
    k0 = k0.reshape(4015,1)
    omega = k0

    while True:
        s = -k0
        E_prime0 = np.dot(k0.T,s)
        lambda_factor = 2

        # Iterative update of s until the condition 4 * E0 > - E_prime0 is true
        if 4 * E0 < - E_prime0:
            s = -4 * s * E0 / E_prime0

        # Update x using the current Jacobian and search direction
        parameters = update_parameters_with_jacobian(parameters,structure_cache, s)

        # Break the loop if the update is smaller than epsilon to prevent infinite loops
        if -E_prime0 < epsilon:
            break  # Stopping condition met, break the loop

        # Calculate a new forward pass and then the cost
        AL, caches = Model_forward(x_train_flattened, parameters)
        # Compute the cost
        E = compute_cost(AL, one_hot_encoded_y_train.T) # new cost with new parameters after first update

        # Check if the update is sufficient, otherwise adjust
        while E > E0:
            s = s/2
            E_prime0 = E_prime0/2
            lambda_factor = 1/2
            parameters = update_parameters_with_jacobian(parameters, structure_cache,s)
            AL, caches = Model_forward(x_train_flattened, parameters)
            E = compute_cost(AL, one_hot_encoded_y_train.T)

        while True:
            # compute k and E_prime
            k = np.dot(J.T, s)
            E_prime = np.dot(k.T, s)

            # Compute b0 and m
            b0 = E_prime - E_prime0 # E_prime0 is the E_prime from the previous iteration
            m = s + k0 - k # k0 is the k from the previous iteration
            E_prime0 = E_prime

            if b0 >= epsilon:
                m_square = np.dot(m.T, m)
                # Check if the norm of u is sufficiently small
                if (np.linalg.norm(m) ** 2 < epsilon):
                    break # convergence criteria met, break the loop
                else:
                    v = np.dot(m.T, s)
                    mu = v - m_square

                    # compute u
                    u = omega - compute(m,omega)

                    # Check if m'u is sufficiently small compared to mu'u
                    if 1e6 * (np.dot(m.T, u) ** 2 >= mu ** 2 * np.dot(u.T, u)):
                        n = np.zeros_like(s)
                        n_squared = 0
                    else:
                        n = compute(u,s)
                        n_squared = (np.dot(u.T, s) ** 2) / np.dot(u.T, u)

                    b = n_squared - ((mu*v)/m_square)

                    # Check if b is sufficiently large
                    while True:
                        if b <= epsilon:

                            if mu*v < m_square * n_squared:
                                a = b - mu
                                c = b + v
                                gamma = np.sqrt(1 - (mu * v) / (m_square * np.dot(mu, mu))*n_squared/ a*b)
                                d = c/a

                                if c >= a:
                                # Update alpha, p, q, omega
                                    m_square = np.dot(m, m)

                                    alpha, p, q, omega = calculate_updates(v, mu, m, n, gamma, delta)

                                else: # c < a
                                    gamma = -gamma
                                    alpha, p, q, omega = calculate_updates(v, mu, m, n, gamma, delta)

                            else:
                                gamma = 0
                                delta = np.sqrt(v/mu)
                                alpha, p, q, omega = calculate_updates(v, mu, m, n, gamma, delta)
                        else:
                            n = s - (v / mu) * m
                            n_squared = b0 - (mu * v / m ** 2)
                            b = b0
                            continue # go back to the beginning of the loop

                        # Update k0 and J
                        k0 = k0 + p * np.dot(q.T, k0)
                        J = J + np.dot(p, q.T)
                        # update parameters with the new value of J
                        parameters["J"] = J

                        # Update current cost values for next iteration
                        E0 = E

                        if n_squared > a:
                            omega = k0  # go back to the beginning of the algorithm

        else:
            s = lambda_factor * s
            E_prime0 = lambda_factor * E_prime0
            continue # go back to the beginning of the loop

        if convergence_criteria_met:
            break # break the while loop


if __name__ == '__main__':

    # Input data:
    input_path = '/home/corina/Documents/Math_Machine_Learning/minst'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Preprocess datasets
    x_train_flattened, x_test_flattened, y_train_flattened, y_test_flattened = prepare_data(x_train, y_train, x_test,
                                                                                            y_test)
    # One hot encode Y ground true values
    one_hot_encoded_y_train = one_hot_encode(y_train_flattened)

    # Define the number of units in each layer of the network
    units_in_layer = [784, 5, 5, 10]

    # Initialize the parameters
    parameters = initialize_parameters(units_in_layer)

    # First forward pass of the neural network
    AL, caches = Model_forward(x_train_flattened, parameters)

    # Compute the cost
    cost = compute_cost(AL, one_hot_encoded_y_train.T)

    # Compute the gradients
    grads = Model_backward(AL, one_hot_encoded_y_train.T, caches)

    davidson_quasi_newton_update(x_train_flattened, parameters, cost, grads, units_in_layer, epsilon=0.001)
