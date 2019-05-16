import logging
import time
from logging import DEBUG

import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.stats import norm

import framework.base as base
import framework.racos as ra
import framework.sk_models as sk
from bandit.model_selection import ModelSelection
from utils import data_loader
from utils.logging_ import get_logger

ALL_DATA = data_loader.all_data(
    exclude=['adult', 'banknote', 'credit', 'egg', 'flag', 'seismic', 'wpbc', 'yeast', 'magic04'])
PROPOSED_DATA = data_loader.data_for_proposed_method()

EVALUATION_CRITERIA = 'Accuracy'

model_generators = [
    sk.DecisionTree(),
    sk.AdaBoost(),
    sk.GaussianNB(),
    sk.KNeighbors(),
    sk.BernoulliNB(),
    sk.ExtraTrees(),
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
        self.p_value = 0.2
        self.p_value1 = 0.1
        self.alpha_em = 0
        self.c_em = 0
        self.b1 = 0
        self.b2 = 0
        self.B = 0
        self.K = 0
        self.E = 0

        self.racos = self._init_racos()

    @property
    def best_evaluation(self):
        if self.instances.empty:
            return pd.Series(data=[0], index=['Accuracy'])
        return self.instances.sort_values(by=EVALUATION_CRITERIA, ascending=False).iloc[0]

    @property
    def best_model(self):
        best_params = self.best_evaluation['Raw Parameters']
        return self.model_generator.generate_model(best_params)

    def _init_racos(self):
        dim = ra.Dimension()
        hp_space = self.model_generator.hp_space
        dim_size = len(hp_space)
        dim.set_dimension_size(dim_size)

        for i in range(dim_size):
            hp = hp_space[i]
            dim.set_region(i, hp.param_bound, hp.param_type)

        racos = ra.RacosOptimization(dim)

        # optimization hyper-parameters
        sample_size = 8  # the instance number of sampling in an iteration
        budget = 20000  # budget in online style
        positive_num = 2  # the set size of PosPop
        rand_probability = 0.99  # the probability of sample in model
        uncertain_bit = 2  # the dimension size that is sampled randomly

        racos.set_parameters(ss=sample_size, bud=budget, pn=positive_num, rp=rand_probability, ub=uncertain_bit)
        # clear optimization model
        racos.clear()

        return racos

    def _update_parameter(self, previous_count, new_eval_result, minbeta, minbeta1, delta, threshold):
        if new_eval_result > np.exp(threshold):
            self.p_value = (previous_count * self.p_value + 1) / (previous_count + 1)
            if new_eval_result > np.exp(threshold + 1):
                self.p_value1 = (previous_count * self.p_value1 + 1) / (previous_count + 1)
        if self.p_value == self.p_value1:
            self.p_value = 0.2
            self.p_value1 = 0.1
        self.alpha_em = np.log(self.p_value) - np.log(self.p_value1)
        if new_eval_result > (previous_count + 1) ** (1 / self.alpha_em / minbeta1):
            self.c_em = (self.c_em * (previous_count ** (-1 / minbeta1 + 1) + 1) / (
                    (previous_count + 1) ** (-1 / minbeta1 + 1)))
        self.B = (previous_count + 1) ** (1 / (self.alpha_em * minbeta1))
        self.K = np.floor(np.log(self.B))
        for k in range(int(self.K)):
            self.E += np.exp(-np.exp((self.K - k - 1) * self.alpha_em))
        self.b2 = self.E * np.sqrt(np.log((previous_count + 1) / delta)) * np.log(previous_count + 1) * (
                previous_count + 1) ** (-minbeta / minbeta1)
        self.b1 = (previous_count + 1) ** (-minbeta / minbeta1) * norm.ppf(1 - delta / 2) / self.alpha_em ** 2

    def run_one_step(self, train_x, train_y, minbeta, minbeta1, delta, threshold):

        start = time.time()

        raw_params = self.racos.sample()
        evaluator = base.ModelEvaluator(self.model_generator, train_x, train_y)
        eval_result = evaluator.evaluate(raw_params)
        self.racos.update_model(raw_params, eval_result)

        elapsed = time.time() - start

        # retrieve actual parameters
        actual_params = self.model_generator.retrieve_actual_params(raw_params)
        self.instances.loc[self.count] = [raw_params, actual_params, eval_result, elapsed]

        # update count
        previous_count = self.count
        self.count += 1

        self._update_parameter(previous_count, 1 / (1 - eval_result), minbeta, minbeta1, delta, threshold)


def _hunter_func(optimization):
    result = ((optimization.c_em + optimization.b2) * optimization.count) ** (
            1 / optimization.alpha_em + optimization.b1) * gamma(1 - 1 / optimization.alpha_em - optimization.b1)
    return result


class ExtremeHunter(ModelSelection):
    def __init__(self, optimizations, logging_level=logging.DEBUG):
        super().__init__(optimizations, logging_level)

    def fit(self, train_x, train_y, budget=200, minbeta=1, threshold=2):
        delta = np.exp(-(np.log(budget))) / (2 * budget * len(self.optimizations))
        # delta_1 = 24 * np.log(2 / delta) / len(self.optimizations)
        threshold = threshold
        minbeta = minbeta
        minbeta1 = 2 * minbeta + 1

        # print(delta_1, delta)
        self._logger.debug('Initializing')
        self._init_each_optimizations(train_x, train_y, minbeta, minbeta1, delta, threshold)

        for t in range(len(self.optimizations) + 1, budget + 1):
            self._logger.debug('Process: {} / {}'.format(t, budget))
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

    def statistics(self):
        data = [(o.name, o.count, o.best_evaluation[EVALUATION_CRITERIA])
                for o in self.optimizations]
        return pd.DataFrame(data=data, columns=['name', 'budget', 'best v'])


def run_model_selection():
    result = []
    for (data, _) in PROPOSED_DATA:
        result.append(run_extreme_bandit(data))
    df_result = pd.DataFrame(data=result, columns=['data set', 'best_v', 'best_model', 'test_v'])
    df_result.to_csv('log/exh/exh-total.csv')


def run_extreme_bandit(data):
    log = get_logger('extreme bandit', 'log/exb.log', level=DEBUG)

    optimizations = _get_optimizations()
    model_selection = ExtremeHunter(optimizations)

    log.info('Begin fit on {}'.format(data.name))
    train_x, train_y = data.train_data()

    best_optimization = model_selection.fit(train_x, train_y, budget=50)

    log.info('Fitting on {} is over'.format(data.name))

    csv_file = 'log/exh/exh_{}.csv'.format(data.name)

    return _get_test_result(best_optimization, data, model_selection.statistics(), csv_file, log)


def _get_test_result(best_optimization, data, statistics, csv_file, log):
    # save statistics to csv
    statistics.to_csv(csv_file)

    # return best_v, best_model, budget
    print(best_optimization.best_evaluation)
    best_v = best_optimization.best_evaluation['Accuracy']
    best_model = best_optimization.name
    test_v = _evaluate_test_v(data, best_optimization.best_model)

    log.info('\n===========================\n'
             'Result of fitting on {}\n'
             'best v: {}\n'
             'best model: {}\n'
             'test v: {}\n'
             '======================'
             .format(data.name, best_v, best_model, test_v))

    return data.name, best_v, best_model, test_v


def _evaluate_test_v(data, model):
    train_x, train_y = data.train_data()
    model.fit(train_x, train_y)

    test_x, test_y = data.test_data()
    return model.score(test_x, test_y)


if __name__ == '__main__':
    run_model_selection()
