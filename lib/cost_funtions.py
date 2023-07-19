import numpy as np
from abc import ABC, abstractmethod


class AbstractCostFunction(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_cost(self, dependent: np.ndarray, predicted: np.ndarray) -> float:
        pass

    @abstractmethod
    def get_weight_partial_derivative(self, independents: np.ndarray, dependent: np.ndarray,
                                      predicted: np.ndarray) -> float:
        pass

    @abstractmethod
    def get_bias_partial_derivative(self, dependent: np.ndarray, predicted: np.ndarray) -> float:
        pass


def resolve_cost_function(cost_function: str) -> AbstractCostFunction:
    mse = "mse"
    rmse = "rmse"
    mae = "mae"
    mape = "mape"

    class MseCostFunction(AbstractCostFunction):
        def get_name(self) -> str:
            return mse

        def get_cost(self, dependent: np.ndarray, predicted: np.ndarray) -> float:
            return np.mean((dependent - predicted) ** 2)

        def get_weight_partial_derivative(self, independents: np.ndarray, dependent: np.ndarray,
                                          predicted: np.ndarray) -> float:
            return (-2 / len(dependent)) * independents @ (dependent - predicted)

        def get_bias_partial_derivative(self, dependent: np.ndarray, predicted: np.ndarray) -> float:
            return (-2 / len(dependent)) * np.sum(dependent - predicted)

    class RmseCostFunction(AbstractCostFunction):
        def get_name(self) -> str:
            return rmse

        def get_cost(self, dependent: np.ndarray, predicted: np.ndarray) -> float:
            return np.sqrt(np.mean((dependent - predicted) ** 2))

        def get_weight_partial_derivative(self, independents: np.ndarray, dependent: np.ndarray,
                                          predicted: np.ndarray) -> float:
            return (-2 / len(dependent)) * independents @ (dependent - predicted) / \
                np.sqrt(np.mean((dependent - predicted) ** 2))

        def get_bias_partial_derivative(self, dependent: np.ndarray, predicted: np.ndarray) -> float:
            return (-2 / len(dependent)) * np.sum(dependent - predicted) / \
                np.sqrt(np.mean((dependent - predicted) ** 2))

    class MaeCostFunction(AbstractCostFunction):
        def get_name(self) -> str:
            return mae

        def get_cost(self, dependent: np.ndarray, predicted: np.ndarray) -> float:
            return np.mean(np.abs(dependent - predicted))

        def get_weight_partial_derivative(self, independents: np.ndarray, dependent: np.ndarray,
                                          predicted: np.ndarray) -> float:
            return (-1 / len(dependent)) * independents @ np.sign(dependent - predicted)

        def get_bias_partial_derivative(self, dependent: np.ndarray, predicted: np.ndarray) -> float:
            return (-1 / len(dependent)) * np.sum(np.sign(dependent - predicted))

    class MapeCostFunction(AbstractCostFunction):
        def get_name(self) -> str:
            return mape

        def get_cost(self, dependent: np.ndarray, predicted: np.ndarray) -> float:
            return np.mean(np.abs((dependent - predicted) / dependent)) * 100

        def get_weight_partial_derivative(self, independents: np.ndarray, dependent: np.ndarray,
                                          predicted: np.ndarray) -> float:
            abs_error = np.abs(dependent - predicted)
            return (-1 / len(dependent)) * (independents @ (dependent / abs_error - predicted / abs_error ** 2))

        def get_bias_partial_derivative(self, dependent: np.ndarray, predicted: np.ndarray) -> float:
            abs_error = np.abs(dependent - predicted)
            return (-1 / len(dependent)) * np.sum(dependent / abs_error - predicted / abs_error ** 2)

    if cost_function == mse:
        return MseCostFunction()
    elif cost_function == rmse:
        return RmseCostFunction()
    elif cost_function == mae:
        return MaeCostFunction()
    elif cost_function == mape:
        return MapeCostFunction()
    else:
        return None

