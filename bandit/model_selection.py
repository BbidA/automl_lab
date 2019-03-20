import abc

import numpy as np
import pandas as pd

from framework.random_search import random_search
import random
import logging

EVALUATION_CRITERIA = 'Accuracy'


class RandomOptimization:

    def __init__(self, model_generator, name=None):
        self.model_generator = model_generator
        self.name = name
        self.count = 0
        self.time_out_count = 0

        # Evaluation results
        self.instances = pd.DataFrame(columns=['Raw Parameters', 'Actual Parameters', 'Accuracy', 'Time'])

        # Gaussian parameters
        self.mu = 0
        self.sigma = 0

        # 其实就是 X 的平方的均值，名字取得有点问题
        self.square_mean = 0

        # Parameter change record
        self.param_change_info = []

    def __str__(self):
        return 'Model {}\nBudget: {}\nTimeout count: {}\n======Best result======:\n {}\n============\n' \
               'Gaussian mu: {}\nGaussian sigma: {}\nmu_Y: {}'.format(self.name, self.count, self.time_out_count,
                                                                      self.best_evaluation, self.mu, self.sigma,
                                                                      self.square_mean)

    @property
    def best_evaluation(self):
        if self.instances.empty:
            return pd.Series(data=[0], index=['Accuracy'])
        return self.instances.sort_values(by=EVALUATION_CRITERIA, ascending=False).iloc[0]

    @property
    def best_model(self):
        best_params = self.best_evaluation['Raw Parameters']
        return self.model_generator.generate_model(best_params)

    def run_one_step(self, train_x, train_y, beta=0):
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
        self._update_parameter(previous_count, eval_value, beta)

    def clear(self):
        self.count = 0
        self.time_out_count = 0

        self.instances = pd.DataFrame()

        self.mu = 0
        self.sigma = 0

        self.square_mean = 0

        self.param_change_info = []

    def _update_parameter(self, previous_count, new_eval_result, beta):
        """Update parameters of this optimization after one step

        Parameters
        ----------

        beta : float
            bias minus from mu

        """
        new_eval_result = new_eval_result - beta

        self.mu = (previous_count * self.mu + new_eval_result) / (previous_count + 1)
        self.sigma = self.instances[EVALUATION_CRITERIA].std()
        self.square_mean = (previous_count * self.square_mean + new_eval_result ** 2) / (previous_count + 1)


def _new_func(optimization, t, theta=1, record=None, gamma=1):
    third_term = np.sqrt(2 * np.log(t) / optimization.count)
    forth_term = np.sqrt(1 / theta * third_term)
    second_term = np.sqrt(1 / theta * optimization.square_mean)
    result = gamma * (optimization.mu + second_term) + third_term + forth_term

    if record is not None:
        assert isinstance(record, list)
        record.append((optimization.name, optimization.mu, optimization.square_mean, second_term,
                       third_term, forth_term, third_term + forth_term, result))

    return result


def _ucb_func(optimization, t, record=None):
    second_term = np.sqrt(2 * np.log(t) / optimization.count)
    result = optimization.mu + second_term

    if record is not None:
        assert isinstance(record, list)
        record.append((optimization.name, optimization.mu, second_term, result))

    return result


class ModelSelection:
    def __init__(self, optimizations, logging_level=logging.DEBUG):
        self.optimizations = optimizations
        self._logger = self._init_logger(logging_level)

    def show_models(self):
        models_info = ''
        for optimization in self.optimizations:
            models_info += str(optimization)
            models_info += '\n\n'

        return models_info

    @abc.abstractmethod
    def fit(self, train_x, train_y):
        return

    def statistics(self):
        data = [(o.name, o.mu, o.sigma, o.count, o.best_evaluation[EVALUATION_CRITERIA])
                for o in self.optimizations]
        return pd.DataFrame(data=data, columns=['name', 'mu', 'sigma', 'budget', 'best v'])

    def _best_selection(self):
        best_results = [r.best_evaluation[EVALUATION_CRITERIA] for r in self.optimizations]
        best_index = np.argmax(best_results)

        return self.optimizations[best_index]

    @staticmethod
    def _init_logger(level):
        logger = logging.getLogger('model_selection')
        logger.setLevel(level)

        if logger.hasHandlers():
            return logger

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        return logger


class BanditModelSelection(ModelSelection):
    _update_functions = ['new', 'ucb', 'random']

    def __init__(self, optimizations, update_func='new', theta=1, gamma=1, beta=0):
        super().__init__(optimizations)
        self.param_change_info = []
        self.theta = theta
        self.gamma = gamma
        self.update_func = update_func
        self.beta = beta

    def fit(self, train_x, train_y, budget=200):
        """Fit on training data and select the best model

        Parameters
        ----------
        train_x: np.ndarray or list
            the features

        train_y: np.ndarray or list
            the label

        budget: int
            the number of samples

        Returns
        -------

        result: RandomOptimization
            best model

        """
        self._clean()  # clean history data
        self._logger.debug('Initializing')
        self._init_each_optimizations(train_x, train_y, beta=self.beta)

        for t in range(len(self.optimizations) + 1, budget + 1):
            self._logger.debug('Process: {} / budget'.format(t))
            next_model = self._next_selection(t)
            next_model.run_one_step(train_x, train_y, beta=self.beta)

        return self._best_selection()

    def statistics(self):
        if self.update_func == 'new':
            data = [(o.name, o.mu, o.sigma, o.square_mean, o.count, o.best_evaluation[EVALUATION_CRITERIA])
                    for o in self.optimizations]
            return pd.DataFrame(data=data, columns=['name', 'mu(-beta)', 'sigma', 'mu_Y', 'budget', 'best v'])
        else:
            # random or ucb method
            return super().statistics()

    def _wrap_selection_information(self, data):
        if self.update_func == 'new':
            return pd.DataFrame(data=data, columns=['name', 'mu', 'square_mean', 'sqrt(mu_Y)', 'third term',
                                                    'forth term', 'sum of last two', 'sum all'])
        elif self.update_func == 'ucb':
            return pd.DataFrame(data=data, columns=['name', 'mu', 'second_term', 'sum all'])

    def _init_each_optimizations(self, train_x, train_y, beta):
        for optimization in self.optimizations:
            optimization.clear()  # clear history data
            optimization.run_one_step(train_x, train_y, beta=beta)

    def _next_selection(self, current_count):
        selection_record = []  # used to record values of the terms of the equation for each models
        if self.update_func == 'new':
            values = [_new_func(o, current_count, theta=self.theta, record=selection_record, gamma=self.gamma)
                      for o in self.optimizations]
        elif self.update_func == 'ucb':
            values = [_ucb_func(o, current_count, selection_record) for o in self.optimizations]
        else:
            # return random result
            return random.choice(self.optimizations)

        self.param_change_info.append(self._wrap_selection_information(selection_record))
        return self.optimizations[np.argmax(values)]

    def _clean(self):
        self.param_change_info = []


class EpsilonGreedySelection(ModelSelection):

    def __init__(self, optimizations):
        super().__init__(optimizations)

    def fit(self, train_x, train_y, epsilon=0.1, budget=200):
        for i in range(1, budget + 1):
            self._logger.debug('Process: {} / {}'.format(i, budget))
            point = random.uniform(0, 1)
            if point < epsilon:
                # do exploration
                selection = random.choice(self.optimizations)
            else:
                # do exploitation
                values = np.array([o.mu for o in self.optimizations])
                max_items = np.argwhere(values == values.max())
                max_item = random.choice(max_items.reshape(max_items.shape[0]))
                selection = self.optimizations[max_item]
            assert isinstance(selection, RandomOptimization)

            selection.run_one_step(train_x, train_y)

        return self._best_selection()


class SoftMaxSelection(ModelSelection):

    def fit(self, train_x, train_y, temperature=0.1, budget=200):
        for i in range(budget):
            self._logger.debug('Process: {} / {}'.format(i + 1, budget))
            model = self._next_selection(temperature)
            model.run_one_step(train_x, train_y)

        return self._best_selection()

    def _next_selection(self, temperature):
        # construct select range
        select_range = [0]
        for o in self.optimizations:
            select_range.append(select_range[-1] + np.power(np.e, o.mu / temperature))

        # choose a model according the select range
        point = random.uniform(0, select_range[-1])
        for i in range(1, len(select_range)):
            if point < select_range[i]:
                return self.optimizations[i - 1]

        assert False
