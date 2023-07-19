import matplotlib.pyplot as plt
import numpy as np


def plot_graphs(dependent: np.ndarray, predicted: np.ndarray,
                mse_over_iterations: np.ndarray, learning_rate: float, cost_function: str, norm_method: str):
    num_iters = len(mse_over_iterations)

    def plot_mse_over_iterations():
        plt.plot(range(len(mse_over_iterations)), mse_over_iterations)
        plt.ylabel(cost_function.upper())
        plt.xlabel('Iterations')
        plt.suptitle('Cost over iterations')
        plt.title(f'Normalization method: {norm_method} Cost function: {cost_function}')
        plt.savefig(f'mlr_training_{num_iters}_{learning_rate}_{cost_function}.png')
        plt.clf()

    def plot_prediction_error():
        index = np.arange(len(dependent)) + 1
        differences = np.abs(dependent - predicted)
        plt.scatter(index, dependent, label='Actual Values')
        plt.scatter(index, predicted, label='Predicted Values')
        for i in range(len(dependent)):
            plt.vlines(i + 1, dependent[i], predicted[i], colors='r', linewidth=2)
        plt.scatter(np.arange(len(differences)) + 1, dependent, color='b', label=None)
        # plt.errorbar(index, actual, yerr=differences, fmt='none', ecolor='r', capsize=5, label='Absolute Difference')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.suptitle('Actual vs Predicted Values')
        plt.title(f'Normalization method: {norm_method} Cost function: {cost_function}')
        plt.legend()
        plt.savefig(f'mlr_error_{num_iters}_{learning_rate}_{cost_function}.png')
        plt.clf()

    plot_mse_over_iterations()
    plot_prediction_error()