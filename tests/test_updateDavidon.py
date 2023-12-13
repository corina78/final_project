import unittest
import numpy as np
import os
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from helper_functions import load_pickle, flatten_gradients_for_jacobian, update_parameters_with_jacobian

class TestNeuralNetFunctions(unittest.TestCase):

    def test_flatten_gradients_for_jacobian(self):
        # Load Mock input
        grads = load_pickle('grads.pickle')

        units_in_layer = [2, 2]  # Mock layer units

        # Expected output
        expected_flattened = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        expected_structure = [('dW1', (2, 2)), ('db1', (2,))]

        # Run the function
        flattened_gradient, structure_cache, _ = flatten_gradients_for_jacobian(grads, units_in_layer)

        # Check if the output is as expected
        np.testing.assert_array_equal(flattened_gradient, expected_flattened)
        self.assertEqual(structure_cache, expected_structure)

    def test_update_parameters_with_jacobian(self):
        # Mock input
        params = {
            'W1': np.array([[1, 2], [3, 4]]),
            'b1': np.array([5, 6]),
            'J': np.eye(6)  # Identity matrix for simplicity
        }
        structure_cache = [('dW1', (2, 2)), ('db1', (2,))]
        s = np.array([1, 2, 3, 4, 5, 6])

        # Expected output
        expected_params = {
            'W1': np.array([[2, 4], [6, 8]]),
            'b1': np.array([10, 12])
        }

        # Run the function
        updated_params = update_parameters_with_jacobian(params, structure_cache, s)

        # Check if the parameters are updated as expected
        np.testing.assert_array_almost_equal(updated_params['W1'], expected_params['W1'])
        np.testing.assert_array_almost_equal(updated_params['b1'], expected_params['b1'])

if __name__ == '__main__':
    unittest.main()
