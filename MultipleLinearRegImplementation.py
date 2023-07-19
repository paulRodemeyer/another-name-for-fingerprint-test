# This program will implement Multiple Linear Regression using Gradient Descent,
# multiple cost functions (MSE, MAE, MAPE, RMSE), multiple normalization methods (Max, Min-Max, and Z-Score),
# and r-squared as a measure of good fit.

from lib.data_pipeline import read_data
from lib.graphing import plot_graphs
from lib.logging_setup import build_logger
from lib.scribe import record_run, RecordBuilder
from lib.training import gradient_descent

import argparse as ap
import time


def parse_args() -> ap.Namespace:
    parser = ap.ArgumentParser(
        prog='Multiple Linear Regression',
        description='Multiple Linear Regression algorithm ' +
                    'using gradient descent, mse (cost function), and r-squared (fit)'
    )

    def must_be_positive_float_le_one(value: str) -> float:
        value = float(value)
        if value <= 0.0 or value > 1.0:
            raise ap.ArgumentTypeError(f'Must be between (0, 1]')
        return value

    def must_be_positive_int(value: str) -> int:
        value = int(value)
        if value <= 0:
            raise ap.ArgumentTypeError(f'Must be between (0, infinity)')
        return value

    parser.add_argument('-l', '--learning-rate',
                        help='A value between [0, 1] that determines the sensitivity of training.',
                        required=True,
                        type=must_be_positive_float_le_one,
                        dest='learning_rate')
    parser.add_argument('-n', '--num-iters',
                        help='Number of iterations to train using gradient descent. [1, inf)',
                        required=True,
                        type=must_be_positive_int,
                        dest='num_iters')
    parser.add_argument('-f', '--file',
                        help='Path to input file used for training. Only CSV and XLSX supported.',
                        required=True,
                        type=str,
                        dest='path')
    parser.add_argument('-d', '--data-normalization',
                        default='max',
                        type=str,
                        dest='data_norm')
    parser.add_argument('-c', '--cost-function',
                        default='mse',
                        type=str,
                        dest='cost_function')

    return parser.parse_args()


def main():
    start = time.perf_counter()
    args = parse_args()

    response = read_data(args.path, args.data_norm)
    if response.is_nothing():
        exit(1)

    independents, dependent = response.value

    weights, bias, r_squared_val, mse_vals = \
        gradient_descent(independents, dependent, args.learning_rate, args.num_iters, args.cost_function)
    final_cost = mse_vals[-1]
    logger = build_logger('parameters', f'mlr_model_parameters.log', True)

    logger.info(f'Learning rate: {args.learning_rate}')
    logger.info(f'Number of iterations: {args.num_iters}')
    logger.info(f'x1 slope: {weights[0]}')
    logger.info(f'x2 slope: {weights[1]}')
    logger.info(f'Intercept: {bias}')
    logger.info(f'{args.cost_function.upper()}: {final_cost}')
    logger.info(f'R-squared: {r_squared_val}')
    logger.info(f'Normalization method: {args.data_norm}')
    logger.debug(f'Total execution time: {time.perf_counter() - start:0.04f} seconds')
    x1, x2 = independents

    predicted = weights[0] * x1 + weights[1] * x2 + bias

    plot_graphs(dependent, predicted, mse_vals, args.learning_rate, args.cost_function, args.data_norm)
    record_run(builder=RecordBuilder()
               .set_model_hyperparams(args.learning_rate, len(mse_vals), args.data_norm)
               .set_error_fit(final_cost, r_squared_val)
               .set_slopes_intercept(weights[0], weights[1], bias),
               cost_fcn=args.cost_function,
               norm_meth=args.data_norm)


if __name__ == '__main__':
    main()

