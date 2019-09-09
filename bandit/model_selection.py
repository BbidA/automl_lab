import abc
import logging
import random

import numpy as np
import pandas as pd

EVALUATION_CRITERIA = 'Accuracy'


def _new_func(optimization, t, theta=1.0, record=None, gamma=1):
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


class ERUCB(ModelSelection):
    def __init__(self, optimizations, theta=0.01):
        super().__init__(optimizations)
        self.theta = theta
        self.param_change_info = []
        self.columns = ['name', 'count', 'beta1', 'beta0', 'e_beta0', 'e_beta1', 'e_variance',
                        'sigma', 'func_m', 'func_n', '1/t^alpha', 'e_beta0_inside_ppf',
                        'e_beta1_inside_ppf', 'e_beta0_item3', 'muY_kt', 'delta_t', 'sqrt(1/theta*variance)',
                        'last_item', 'selection_value']

    def fit(self, train_x, train_y, budget=200):
        # Initializing models
        self._logger.info("Initialization")
        consumption = self._init_models(train_x, train_y) - 1
        self._logger.info("Initialization Done")

        # do model selection
        while consumption < budget:
            self._logger.info("Process: {}/{}".format(consumption + 1, budget))
            selection_values = [o.selection_value(consumption + 1, self.theta) for o in self.optimizations]

            if np.isnan(selection_values).any():
                for (value, o) in zip(selection_values, self.optimizations):
                    if np.isnan(value):
                        while consumption < budget and np.isnan(value):
                            self._logger.info('Selection value of {} is nan, rerunning: {}/{}'
                                              .format(o.name, consumption + 1, budget))
                            self._update_param_info(consumption + 1, prefix='rerun {} '.format(o.name))

                            o.run_one_step(train_x, train_y)
                            consumption += 1
                            value = o.selection_value(consumption + 1, self.theta)

                        if consumption >= budget:
                            return self._best_selection()

                selection_values = [o.selection_value(consumption + 1, self.theta) for o in self.optimizations]
                assert not np.isnan(selection_values).any()

            next_model = self.optimizations[np.argmax(selection_values)]
            self._update_param_info(consumption + 1, prefix='Select {} '.format(next_model.name))
            next_model.run_one_step(train_x, train_y)
            consumption += 1

        return self._best_selection()

    def _update_param_info(self, t, prefix=''):
        param_info = [o.param_info(t, self.theta) for o in self.optimizations]
        self.param_change_info.append(('{}t={}'.format(prefix, t),
                                       pd.DataFrame(data=param_info, columns=self.columns)))

    def statistics(self):
        data = [(o.name, o.beta0, o.beta1, o.variance, o.count, o.best_evaluation[EVALUATION_CRITERIA]) for o in
                self.optimizations]
        return pd.DataFrame(data=data, columns=['name', 'beta0', 'beta1', 'variance', 'budget', 'best v'])

    def _init_models(self, train_x, train_y, init_times=3):
        count = 1
        total_count = len(self.optimizations) * init_times
        for o in self.optimizations:
            self._logger.info('Initializing {}'.format(o.name))
            for _ in range(init_times):
                self._logger.info('Init {}/{}'.format(count, total_count))
                count += 1
                o.run_one_step(train_x, train_y)
        return count


class BanditModelSelection(ModelSelection):
    _update_functions = ['new', 'ucb', 'random']

    def __init__(self, optimizations, update_func='new', theta=0.01, gamma=20, beta=0):
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

        result: bandit.model_optimization.RandomOptimization
            best model

        """
        self._clean()  # clean history data
        self._logger.debug('Initializing')
        self._init_each_optimizations(train_x, train_y, beta=self.beta)

        for t in range(len(self.optimizations) + 1, budget + 1):
            self._logger.debug('Process: {} / {}'.format(t, budget))
            next_model = self._next_selection(t)
            if self.update_func == 'new':
                next_model.run_one_step(train_x, train_y, beta=self.beta)
            else:
                next_model.run_one_step(train_x, train_y)

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
            if self.update_func == 'new':
                optimization.run_one_step(train_x, train_y, beta=beta)
            else:
                optimization.run_one_step(train_x, train_y)

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


class SingleArm(ModelSelection):

    def fit(self, train_x, train_y, budget=200):
        model = self.optimizations[0]
        for i in range(budget):
            self._logger.debug(f'Process: {i + 1}/{budget}')
            model.run_one_step(train_x, train_y)
        return model.instances
