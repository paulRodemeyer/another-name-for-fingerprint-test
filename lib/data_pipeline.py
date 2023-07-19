from lib.logging_setup import build_logger

from enum import Enum, auto
import numpy as np
import os
import pandas as pd
from pymonad.maybe import Maybe, Just, Nothing
from typing import Tuple, Union


def read_data(file_path: str, normalization: str) -> Union[Maybe[Tuple[np.ndarray, np.ndarray]], Maybe]:
    logger = build_logger(read_data.__name__, 'data_in.log', True)

    class Normalization(Enum):
        MINMAX = auto(),
        MAX = auto(),
        ZSCORE = auto(),
        NULL = auto()

    requested_normalization = Normalization.__members__.get(normalization.upper())

    def ensure_file_path_exists(file_path: str) -> Union[Maybe[str], Maybe]:
        logger.info('Checking if data file exists...')
        response = Just(file_path) if os.path.exists(file_path) and os.path.isfile(file_path) and os.path.getsize(file_path) > 0 else Nothing
        if response.is_nothing():
            logger.error('File does not exist or is empty!')
        return response

    def read_contents(file_path: str) -> Union[Maybe[str], Maybe]:
        logger.info('Attempting to read contents of data file...')
        response = Just(pd.read_csv(file_path)) if file_path.endswith('.csv') else Just(pd.read_excel(file_path)) if file_path.endswith('xlsx') else Nothing
        if response.is_nothing():
            logger.error('Only CSV or XLSX supported. Please supply another file.')
        return response

    def extract_variables(data: pd.DataFrame) -> Union[Maybe[Tuple[np.ndarray, np.ndarray, np.ndarray]], Maybe]:
        logger.info('Searching for expected variables and extracting them...')
        try:
            value = (data['x1'].values, data['x2'].values, data['y'].values)
        except KeyError as error:
            logger.error('Expected three variables: x1, x2, y')
            logger.error(error)
            return Nothing

        response = Just(value)

        return response

    def normalize_variables(variables: Tuple[np.ndarray, np.ndarray]) \
            -> Union[Maybe[Tuple[np.ndarray, np.ndarray]], Maybe]:
        def minmax_scaling(before: np.ndarray) -> np.ndarray:
            return (before - np.min(before)) / (np.max(before) - np.min(before))

        def max_scaling(before: np.ndarray) -> np.ndarray:
            return before / np.max(before)

        def zscore(before: np.ndarray) -> np.ndarray:
            return (before - np.mean(before)) / np.std(before)

        def null(before: np.ndarray) -> np.ndarray:
            return before

        def unsupported(before: np.ndarray) -> None:
            return None

        logger.info('Normalizing variables...')
        method = minmax_scaling if requested_normalization == Normalization.MINMAX else \
            max_scaling if requested_normalization == Normalization.MAX else \
            zscore if requested_normalization == Normalization.ZSCORE else \
            null if requested_normalization == Normalization.NULL else \
            unsupported

        if method is unsupported:
            logger.error(f'Unsupported normalization method! Supported types: {list(Normalization)}')
            return Nothing

        (x1, x2, y) = variables
        x1 = method(x1)
        x2 = method(x2)

        return Just((np.array([x1, x2]), y))

    return ensure_file_path_exists(file_path)\
        .bind(lambda p: read_contents(p))\
        .bind(lambda c: extract_variables(c))\
        .bind(lambda v: normalize_variables(v))
