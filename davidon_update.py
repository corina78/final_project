import numpy as np
from os.path import join
import pickle
from loader import MnistDataloader
from preprocessing import prepare_data, one_hot_encode
from helper_functions import *
from memory_profiler import profile

@profile
def davidson_quasi_newton_update(x_train_flattened, parameters, E, k, units_in_layer, epsilon=0.001, max_iterations=10):

    # Initialize variables only once
    J = parameters['J'] # initial parameters
    E0 = cost # initial cost before the first forward pass, just the cost related to the initial parameters
    grad_vector, structure_cache, size = flatten_gradients_for_jacobian(grads, units_in_layer) # grads after first backward pass
    # Check if sizes are compatible for matrix multiplication
    if grad_vector.shape[0] != size:
        raise ValueError(f"The gradient vector is expected to be of size 4015, but got size {grad_vector.shape[0]}")
    k0 = np.dot(J,grad_vector) # perform matrix multiplication to get k0, grad_vector after first backward pass
    k0 = k0.reshape(grad_vector.shape[0],1)
    omega = k0

    iteration_counter = 0

    while True:

        iteration_counter += 1

        if iteration_counter >= max_iterations:
            print("Maximum number of iterations reached, exiting the algorithm")
            return

        s = -k0
        E_prime0 = np.dot(k0.T,s)
        lambda_factor = 2

        # Iterative update of s until the condition 4 * E0 > - E_prime0 is true
        while 4 * E0 < - E_prime0:
            s = -4 * s * E0 / E_prime0

        # Update parameters using the current Jacobian and search direction
        parameters = update_parameters_with_jacobian(parameters,structure_cache, s)

        # Break the loop if the update is smaller than epsilon to prevent infinite loops
        if -E_prime0 < epsilon:
            print("Stopping criteria met, exiting the algorithm")
            return

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
            if -E_prime0 < epsilon:
                print("Stopping criteria met, exiting the algorithm")
                return
            AL, caches = Model_forward(x_train_flattened, parameters)
            E = compute_cost(AL, one_hot_encoded_y_train.T)

        while True:
            # compute k and E_prime
            E = compute_cost(AL, one_hot_encoded_y_train.T)
            k = np.dot(J.T, s)
            E_prime = np.dot(k.T, s)

            # Compute b0 and m
            b0 = E_prime - E_prime0 # E_prime0 is the E_prime from the previous iteration
            m = s + k0 - k # k0 is the k from the previous iteration
            E_prime0 = E_prime

            if b0 >= epsilon:
                m_square = np.dot(m.T, m)
                # Check if the norm of m is sufficiently small
                if (np.linalg.norm(m) ** 2 < epsilon):
                    print("Convergence criteria met, exit")
                    return
                else:
                    v = np.dot(m.T, s)
                    mu = v - m_square

                    # compute u
                    u = omega - compute(m,omega)

                    # Check if m'u is sufficiently small compared to u'u
                    if 1e6 * (np.dot(m.T, u) ** 2 >= m_square * np.dot(u.T, u)):
                        n = np.zeros_like(s) # assuming size of s
                        n_squared = 0
                    else:
                        n = compute(u,s)
                        n_squared = (np.dot(u.T, s) ** 2) / np.dot(u.T, u)

                    b = n_squared - ((mu*v)/m_square)

                    # Check if b is sufficiently large
                    alpha = p = omega = delta = None
                    if b <= epsilon:
                        if mu*v < m_square * n_squared:
                            a = b - mu
                            c = b + v
                            gamma = calculate_gamma(mu, v, m_square, n_squared, a, b)
                            d = c/a if a!=0 else None

                            if c < a:
                                gamma = -gamma
                                alpha, p, q, omega = calculate_alpha_p_q_omega(v, mu, m, n, gamma, delta)
                            else:
                                alpha, p, q, omega = calculate_alpha_p_q_omega(v, mu, m, n, gamma, delta)
                        else:
                            gamma = 0
                            delta = np.sqrt(v/mu)
                            alpha, p, q, omega = calculate_alpha_p_q_omega(v, mu, m, n, gamma, delta)
                    else:
                        n = s - (v * m)/m_square
                        n_squared = b0 - (mu*v)/m_square
                        b = b0
                        if mu*v < m_square * n_squared:
                            a = b - mu
                            c = b + v
                            gamma = calculate_gamma(mu, v, m_square, n_squared, a, b)
                            d = c/a if a!=0 else None

                            if c < a:
                                gamma = -gamma
                                alpha, p, q, omega = calculate_alpha_p_q_omega(v, mu, m, n, gamma, delta)
                            else:
                                alpha, p, q, omega = calculate_alpha_p_q_omega(v, mu, m, n, gamma, delta)
                        else:
                            gamma = 0
                            delta = np.sqrt(v/mu)
                            alpha, p, q, omega = calculate_alpha_p_q_omega(v, mu, m, n, gamma, delta)

                    # Update parameters
                    qTk0 = np.dot(q.T, k0)
                    k0 = k0 + p * qTk0
                    qpT = np.dot(q, p.T)
                    J = J +  J * qpT
                    # save cost and parameters for next iteration
                    J = parameters['J']
                    E0 = E
                    #print("cost: ", E0)

                    if n_squared > 0:
                        break
                    else:
                        omega = k0
                        break
            else:
                S = lambda_factor * s
                E_prime0 = lambda_factor* E_prime0
                continue

    return parameters

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

    # take a sample:
    x_train_flattened = x_train_flattened[:, :64]
    y_train_flattened = y_train_flattened[:, :64]
    x_test_flattened = x_test_flattened[:, :64]
    y_test_flattened = y_test_flattened[:, :64]

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

    parameters = davidson_quasi_newton_update(x_train_flattened, parameters, cost, grads, units_in_layer, epsilon=0, max_iterations=5)

    # Get predictions for the training and test sets
    #predictions_train = predict(x_train_flattened, parameters)
    #predictions_test = predict(x_test_flattened, parameters)

    # Compute the accuracy of the predictions
    #accuracy_train = compute_accuracy(predictions_train, y_train_flattened)
    #accuracy_test = compute_accuracy(predictions_test, y_test_flattened)

    #print(f"Accuracy on the training set: {accuracy_train}")
    #print(f"Accuracy on the test set: {accuracy_test}")