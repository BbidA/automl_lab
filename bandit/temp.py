import logging

import numpy as np
import pandas as pd
from scipy.stats import norm, gamma

import framework.sk_models as sk
from framework.random_search import random_search
from utils import data_loader
from bandit.model_selection import ModelSelection

ALL_DATA = data_loader.all_data(
    exclude=['adult', 'banknote', 'credit', 'egg', 'flag', 'seismic', 'wpbc', 'yeast', 'magic04'])

EVALUATION_CRITERIA = 'Accuracy'

model_generators = [
    sk.DecisionTree(),
    sk.AdaBoost(),
    sk.QuadraticDiscriminantAnalysis(),
    sk.GaussianNB(),
    sk.KNeighbors(),
    sk.BernoulliNB(),
    sk.ExtraTree(),
    sk.PassiveAggressive(),
    sk.RandomForest(),
    sk.SGD()
]


def _get_optimizations():
    return [HunterOptimization(generator, type(generator).__name__) for generator in model_generators]


class HunterOptimization:
    def __init__(self, model_generator, name=None):
        self.model_generator = model_generator
        self.name = name
        self.count = 0
        self.time_out_count = 0
        self.instances = pd.DataFrame(columns=['Raw Parameters', 'Actual Parameters', 'Accuracy', 'Time'])
        # new parameter
        self.p_value = 0
        self.p_value1 = 0
        self.alpha_em = 0
        self.c_em = 0
        self.b1 = 0
        self.b2 = 0
        self.B = 0
        self.K = 0
        self.E = 0

    @property
    def best_evaluation(self):
        if self.instances.empty:
            return pd.Series(data=[0], index=['Accuracy'])
        return self.instances.sort_values(by=EVALUATION_CRITERIA, ascending=False).iloc[0]

    @property
    def best_model(self):
        best_params = self.best_evaluation['Raw Parameters']
        return self.model_generator.generate_model(best_params)

    def _update_parameter(self, previous_count, new_eval_result, minbeta, minbeta1, delta, threshold):
        if new_eval_result > np.exp(threshold):
            self.p_value = (previous_count * self.p_value + 1) / (previous_count + 1)
            if new_eval_result > np.exp(threshold + 1):
                self.p_value1 = (previous_count * self.p_value1 + 1) / (previous_count + 1)
        self.alpha_em = np.log(self.p_value) - np.log(self.p_value1)
        if new_eval_result > (previous_count + 1) ** (1 / self.alpha_em / minbeta1):
            self.c_em = (self.c_em * (previous_count ** (-1 / minbeta1 + 1) + 1) / (
                    (previous_count + 1) ** (-1 / minbeta1 + 1)))
        self.B = (previous_count + 1) ** (1 / (self.alpha_em * minbeta1))
        self.K = np.floor(self.B)
        for k in range(int(self.K)):
            self.E += np.exp(-np.exp((self.K - k - 1) * self.alpha_em))
        self.b2 = self.E * np.sqrt(np.log((previous_count + 1) / delta)) * np.log(previous_count + 1) * (
                previous_count + 1) ** (-minbeta / minbeta1)
        self.b1 = (previous_count + 1) ** (-minbeta / minbeta1) * norm.ppf(1 - delta / 2) / self.alpha_em

    def run_one_step(self, train_x, train_y, minbeta, minbeta1, delta, threshold):

        evaluation_result = random_search(self.model_generator, train_x, train_y, search_times=1)

        while evaluation_result.empty:
            # The result is empty because some errors like timeout occurred
            self.time_out_count += 1
            evaluation_result = random_search(self.model_generator, train_x, train_y, search_times=1)

        self.instances = self.instances.append(evaluation_result, ignore_index=True)

        # update count
        previous_count = self.count
        self.count += 1

        eval_value = evaluation_result[EVALUATION_CRITERIA].values[0]

        self._update_parameter(previous_count, 1 / (1 - eval_value), minbeta, minbeta1, delta, threshold)


def _hunter_func(optimization):
    result = ((optimization.c_em + optimization.b2) * optimization.count) ** (
            1 / optimization.alpha_em + optimization.b1) * gamma(1 - 1 / optimization.alpha_em - optimization.b1)
    return result


class ExtremeHunter(ModelSelection):
    def __init__(self, optimizations, logging_level=logging.DEBUG):
        super().__init__(optimizations, logging_level)

    def fit(self, train_x, train_y, budget=200, minbeta=1, threshold=5):
        delta = np.exp(-(np.log(budget))) / (2 * budget * len(self.optimizations))
        delta_1 = 24 * np.log(2 / delta) / len(self.optimizations)
        threshold = threshold
        minbeta = minbeta
        minbeta1 = 2 * minbeta + 1

        self._logger.debug('Initializing')
        self._init_each_optimizations(train_x, train_y, minbeta, minbeta1, delta, threshold)

        for t in range(len(self.optimizations) + 1, budget + 1):
            self._logger.debug('Process: {} / budget'.format(t))
            next_model = self._next_selection()
            next_model.run_one_step(train_x, train_y, minbeta, minbeta1, delta, threshold)

        return self._best_selection()

    def _init_each_optimizations(self, train_x, train_y, minbeta, minbeta1, delta, threshold):
        for optimization in self.optimizations:
            optimization.run_one_step(train_x, train_y, minbeta=minbeta, minbeta1=minbeta1, delta=delta,
                                      threshold=threshold)

    def _next_selection(self):
        values = [_hunter_func(o) for o in self.optimizations]
        return self.optimizations[np.argmax(values)]


def run_model_selection():
    extreme_hunter = ExtremeHunter(_get_optimizations())

    train_x, train_y = ALL_DATA[0].train_data()

    extreme_hunter.fit(train_x, train_y, budget=20)


if __name__ == '__main__':
    run_model_selection()
