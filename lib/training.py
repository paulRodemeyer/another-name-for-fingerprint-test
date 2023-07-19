from lib.cost_funtions import resolve_cost_function
from lib.logging_setup import build_logger

import numpy as np


def gradient_descent(independents: np.ndarray, dependent: np.ndarray, learning_rate: float, num_iterations: int,
                     cost_function: str, decrease_rate: float = 0.5, stop_tolerance: float = 1e-4):
    resolved_cost_function = resolve_cost_function(cost_function)

    def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
        rss = np.sum((actual - predicted) ** 2)
        tss = np.sum((actual - np.mean(actual)) ** 2)
        return 1 - rss / tss

    logger = build_logger('training', f'mlr_training_{num_iterations}_{learning_rate}_'
                                      f'{resolved_cost_function.get_name()}.log', True)

    num_observations: int = len(dependent)

    # initialize weights and bias to zero
    weights = np.zeros(len(independents))
    bias = 0.0
    mse_values = []

    previous_mse = float('inf')
    for iteration in range(num_iterations):
        logger.info(f'Iteration: {iteration+1}/{num_iterations}')

        # compute predicted values
        predicted = independents.T @ weights + bias

        recorded_mse = resolved_cost_function.get_cost(dependent, predicted)
        logger.info(f'{resolved_cost_function.get_name().upper()}: {recorded_mse}')

        # compute gradients
        d_weights = resolved_cost_function.get_weight_partial_derivative(independents, dependent, predicted)
        d_bias = resolved_cost_function.get_bias_partial_derivative(dependent, predicted)
        logger.debug(f'Partial derivatives of weights: {d_weights}')
        logger.debug(f'Partial derivatives of bias: {d_bias}')

        # update weights and bias
        weights -= learning_rate * d_weights
        bias -= learning_rate * d_bias
        logger.debug(f'Updated weights: {weights}')
        logger.debug(f'Updated bias: {bias}')

        # check for convergence
        if abs(previous_mse - recorded_mse) < stop_tolerance:
            logger.info(f'Stopping early at iteration {iteration + 1} due to convergence.')
            break

        # check if error has increased and decrease learning rate if so
        if recorded_mse > previous_mse:
            logger.warning('Error has increased. Decreasing learning rate.')
            learning_rate *= decrease_rate
            logger.debug(f'New learning rate: {learning_rate}')

        mse_values.append(recorded_mse)
        previous_mse = recorded_mse

    return weights, bias, r_squared(dependent, predicted), mse_values