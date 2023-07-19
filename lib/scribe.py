import os.path
from csv import DictWriter

FIELD_NAMES = [
    'Learning Rate', 'Iterations', 'Normalization Method', 'Cost', 'R-squared',
    'x1 Slope', 'x2 Slope', 'Intercept']


class RecordBuilder:
    __row__ = {
        'Learning Rate': 0.0, 'Iterations': 0, 'Normalization Method': 'null', 'Cost': 0.0, 'R-squared': 0.0,
        'x1 Slope': 0.0, 'x2 Slope': 0.0, 'Intercept': 0.0}

    def set_model_hyperparams(self, learning_rate: float, iterations: int, norm_method: str):
        self.__row__['Learning Rate'] = learning_rate
        self.__row__['Iterations'] = iterations
        self.__row__['Normalization Method'] = norm_method
        return self

    def set_slopes_intercept(self, x1_slope: float, x2_slope: float, intercept: float):
        self.__row__['x1 Slope'] = round(x1_slope, 4)
        self.__row__['x2 Slope'] = round(x2_slope, 4)
        self.__row__['Intercept'] = round(intercept, 4)
        return self

    def set_error_fit(self, cost: float, r_squared: float):
        self.__row__['Cost'] = round(cost, 4)
        self.__row__['R-squared'] = round(r_squared, 4)
        return self

    def build(self):
        return self.__row__


def record_run(builder: RecordBuilder, cost_fcn: str, norm_meth: str) -> None:
    row = builder.build()
    file = 'trial_runs.csv'

    if not os.path.isfile(file):
        with open(file, 'w', newline='') as handle:
            writer = DictWriter(handle, fieldnames=FIELD_NAMES)
            writer.writeheader()
            handle.close()

    with open(file, 'a', newline='') as handle:
        writer = DictWriter(handle, fieldnames=FIELD_NAMES)
        writer.writerow(row)
        handle.close()
