from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from pandas import DataFrame, Series
import numpy as np
import typing


class AutoRegModel(AutoReg):
    def __init__(self, **kwargs):
        super().__init__()


class ARIMAModel():
    def __init__(self, data: typing.Union[DataFrame, Series], order: typing.Union[list, tuple]):
        self.data = data
        self.order = order

    def fit(self, seq):
        return self.model.fit(seq)

    def predict(self, lower_limit=None, upper_limit=None):
        if lower_limit is None:
            lower_limit = len(self.data)-10
        if upper_limit is None:
            upper_limit = len(self.data)
        input_seq = self.data.loc[0: lower_limit, :]
        result_seq = self.data.loc[lower_limit:, :]
        pred = np.array([])
        history = input_seq[:, :]
        for t in len(result_seq):
            self.model = ARIMA(history, order=self.order)
            self.trained_model = self.model.fit()
            value = self.trained_model.forecast()
            pred = np.append(pred, value)
            history = np.append(history, result_seq[t])
        return pred
