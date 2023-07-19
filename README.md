# Multiple Linear Regression Implementation
## Introduction
The contents of this document will cover the setup of the core python program `MultipleLinearRegImplementation.py` and instructions on how to use the program. This program is intended to expose students to the inner workings of Multiple Linear Regression by measuring the correlation between an independent variables x1 and x2, and a dependent variable y. Strategies such as gradient descent and cost functions and normalization methods are employed to illustrate the importance of proper hyperparameter selection.

In addition to the main `MultipleLinearRegImplementation.py` will be a `libs` folder containing modules for gradient descent, cost functions, logging, graphing, and trial tracking. The program _will not_ work without this folder and its contents.

## Setup
This program leverages the following third party packages (see `requirements.txt`):
- pandas~=1.5.3
- numpy~=1.24.2
- openpyxl~=3.0.10
- matplotlib~=3.6.3
- PyMonad~=2.4.0

In order to restore these dependencies, execute `python.exe -m pip install -r requirements.txt`.

## Usage
The program's usage is as follows:
`python.exe MultipleLinearRegImplementation.py -l 1e-05 -n 1000 -f myDataMLR.csv`
- `-l` or `--learning-rate`: A scalar value for the learning rate. Proper choice of this variable will influence the sensitivity of the training process. A value too low will require additional training epochs to reach convergence; a value too high may result in divergence.
- `-n` or `--num-iters`: an integer for the number of training iterations to run. A value too low will prevent convergence; a value too high will waste compute.
- `-f` or `--file`: the path to the tabular data file used for training. Columns for x and y are expected; both variables need to be real numbers. Only CSV and XLSX formats are supported.

By default, MSE will be used as a cost function and Max Scaling will be used as a normalization method. Optional switches are provided to override this behavior. To override the normalization method, `-d` or `--data-normalization` can be specified with the options: `max`, `minmax`, `zscore`, or `null`. To override the cost function, `-c` or `--cost-function` can be specified with the following options: `mse`, `mae`, `mape`, or `rmse`.

## Outputs
The program will log to the console events as the model is training, culminating wth the final model parameters:
```
2023-03-05 19:34:42,901 [INFO]: Iteration: 1/1000
2023-03-05 19:34:42,901 [INFO]: MSE: 131183096212.91489
2023-03-05 19:34:42,902 [INFO]: Iteration: 2/1000
2023-03-05 19:34:42,902 [INFO]: MSE: 157103893557.5128
2023-03-05 19:34:42,902 [WARNING]: Error has increased. Decreasing learning rate.
2023-03-05 19:34:42,903 [INFO]: Iteration: 3/1000
2023-03-05 19:34:42,903 [INFO]: MSE: 286672685850.25385
2023-03-05 19:34:42,903 [WARNING]: Error has increased. Decreasing learning rate.
2023-03-05 19:34:42,903 [INFO]: Iteration: 4/1000
2023-03-05 19:34:42,904 [INFO]: MSE: 56359233806.45162
2023-03-05 19:34:42,905 [INFO]: Iteration: 5/1000
2023-03-05 19:34:42,905 [INFO]: MSE: 6617042276.665947
2023-03-05 19:34:42,906 [INFO]: Iteration: 6/1000
2023-03-05 19:34:42,906 [INFO]: MSE: 4209119160.4515734
...
2023-03-05 19:34:42,958 [INFO]: MSE: 4086560101.2059293
2023-03-05 19:34:42,959 [INFO]: Iteration: 46/1000
2023-03-05 19:34:42,959 [INFO]: MSE: 4086560101.2058215
2023-03-05 19:34:42,959 [INFO]: Iteration: 47/1000
2023-03-05 19:34:42,960 [INFO]: MSE: 4086560101.205758
2023-03-05 19:34:42,960 [INFO]: Stopping early at iteration 47 due to convergence.
```
Some outputs are purely for monitor the program as it executes (DEBUG) and are only outputted to console, others are displayed to both console and file output (INFO and WARNING). If anything goes wrong in the execution of the program an ERROR and CRITICAL log will be displayed, detailing the nature of the failure, and will be displayed to console and written to an `errors.log` file. Training process will be logged to a file called `mlr_training_<num-iters>_<learning-rate>_<cost_function>.log` and the final model parameters will be logged to a finle called `mlr_model_parameters.log`. Another log file `data_in.log` will detail the data loading and normalization process.

Finally, two plots will be produced:
- `mlr_error_<num-iters>_<learning-iters>_<cost_function>.png` which will show a two-series scatterplot along with a along with error bars denoting the difference between actual and predicted value pairs.
- `mlr_training_<num-iters>_<learning-iters>_<cost_function>.png` which will show the optimization of MSE over iterations.