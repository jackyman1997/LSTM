from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from pandas import DataFrame, Series
import typing


class AutoRegModel(AutoReg):
    def __init__(self, **kwargs):
        super().__init__()


class ARIMAModel(ARIMA):
    def __init__(self, **kwargs):
        super().__init__()