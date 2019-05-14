import time
import signal

import numpy as np
import pandas as pd
import scipy.stats as stats

import framework.base as base
import framework.racos as ra
from framework.param_search import random_search

EVALUATION_CRITERIA = 'Accuracy'


def timeout_handler(signum, frame):
    raise TimeoutError("Timeout!")


signal.signal(signal.SIGALRM, timeout_handler)


class Optimization:
    def __init__(self, model_generator, name=None):
        self.model_generator = model_generator
        self.name = name
        self.count = 0
        self.time_out_count = 0

        # Evaluation results
        self.instances = pd.DataFrame(columns=['Raw Parameters', 'Actual Parameters', 'Accuracy', 'Time'])

    @property
    def best_evaluation(self):
        if self.instances.empty:
            return pd.Series(data=[0], index=['Accuracy'])
        return self.instances.sort_values(by=EVALUATION_CRITERIA, ascending=False).iloc[0]

    @property
    def best_model(self):
        best_params = self.best_evaluation['Raw Parameters']
        return self.model_generator.generate_model(best_params)


def sigmoid(x):
    return 1 / (1 + 1 / (np.e ** x))


def _func_a(n):
    return (n ** 3 - n) / 12


def inverse_sigmoid(y):
    return np.log(y / (1 - y))


class RacosOptimization(Optimization):

    def __init__(self, model_generator, name=None, b1=1, b2=1, alpha=1.0 / 4,
                 function_g=sigmoid, inverse_func_g=inverse_sigmoid):
        super().__init__(model_generator, name)
        self.beta0 = 0
        self.beta1 = 0
        self.variance = 0
        self.e_beta1 = 0
        self.e_beta0 = 0
        self.e_variance = 0
        self.racos = self._init_racos()
        self.inverse_eval = []
        self.mu = 0

        self.b1 = b1
        self.b2 = b2
        self.alpha = alpha
        self.function_g = function_g
        self.inverse_func_g = inverse_func_g

        self.param_change_info = []
        self._m_value = 0
        self._n_value = 0
        self._e_beta0_item3 = 0

    def __repr__(self) -> str:
        return self.name

    @property
    def sigma(self):
        return self.instances['Accuracy'].std()

    def clear(self):
        return

    def run_one_step(self, train_x, train_y):
        start = time.time()
        signal.alarm(120)

        raw_params = self.racos.sample()
        evaluator = base.ModelEvaluator(self.model_generator, train_x, train_y)
        eval_result = evaluator.evaluate(raw_params)
        self.racos.update_model(raw_params, eval_result)

        signal.alarm(0)
        elapsed = time.time() - start

        self.inverse_eval.append(self.inverse_func_g(eval_result))
        # calculate mu for methods like ucb, epsilon-greedy, and softmax
        self.mu = (self.mu * self.count + eval_result) / (self.count + 1)

        # retrieve actual parameters
        actual_params = self.model_generator.retrieve_actual_params(raw_params)
        self.instances.loc[self.count] = [raw_params, actual_params, eval_result, elapsed]

        # update parameters
        self.count += 1
        self._update_parameters()

    def selection_value(self, t, theta):
        self._update_e_beta0(t)
        self._update_e_beta1(t)
        self._update_e_variance(t)

        return self.beta1 * t + self.beta0 + self.e_beta1 * t + self.e_beta0 + np.sqrt(
            1 / theta * self.variance) + np.sqrt(self.e_variance)

    def param_info(self, t, theta):
        s1 = self.beta1 * t + self.beta0
        s2 = self.e_beta1 * t + self.e_beta0
        s3 = np.sqrt(1 / theta * self.variance)
        s4 = np.sqrt(self.e_variance)
        t_alpha = 1 / t ** self.alpha
        selection_value = s1 + s2 + s3 + s4
        return [self.name, self.count, self.beta1, self.beta0, self.e_beta0, self.e_beta1, self.e_variance,
                np.sqrt(self.variance), self._m_value, self._n_value, t_alpha, t_alpha - self._m_value,
                t_alpha - self._n_value, self._e_beta0_item3, s1, s2, s3, s4, selection_value]

    def _update_parameters(self):
        # update beta0 and beta1
        if len(self.inverse_eval) == 1:
            return
        self.beta1, self.beta0, *_ = stats.linregress(range(1, self.count + 1), self.inverse_eval)

        # update variance
        y_hat = self.function_g(self.beta1 * self.count + self.beta0)
        self._update_variance(y_hat)

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

    def _update_variance(self, y_hat):
        y = self.instances['Accuracy'].values
        variance = ((y - y_hat) ** 2).sum() / self.count
        self.variance = variance if variance != 0 else 1e-5

    def _update_e_beta1(self, t):
        m_value = self._func_m(self.count)
        # print('e_beta1 first {}\nm_value is {}\n'.format(1 / t ** self.alpha, m_value))
        numerator1 = stats.norm.ppf(1 / t ** self.alpha - m_value)
        self.e_beta1 = - numerator1 * np.sqrt(self.variance) / np.sqrt(_func_a(self.count))
        # record m_value for debugging
        self._m_value = m_value

    def _update_e_beta0(self, t):
        x_hat = self._x_hat()
        n_value = self._func_n(self.count)
        # print('e_beta0 first {}\nn_value is {}\n'.format(1 / t ** self.alpha, n_value))
        item1 = stats.norm.ppf(1 / t ** self.alpha - n_value)
        item3 = np.sqrt(1 / self.count + x_hat ** 2 / _func_a(self.count))
        self.e_beta0 = - item1 * np.sqrt(self.variance) * item3

        # record n_value and item3 for debugging
        self._n_value = n_value
        self._e_beta0_item3 = item3

    def _update_e_variance(self, t):
        self.e_variance = (self.count * self.e_beta1 + self.e_beta0 + 1) * np.sqrt(2 * np.log(t) / self.count)

    def _func_m(self, n):
        assert len(self.inverse_eval) == n

        item1 = self.b1 / ((self.variance * _func_a(n)) ** 1.5)
        x_hat = self._x_hat()
        seq_i = np.arange(1, n + 1)
        item2 = (abs(seq_i - x_hat) ** 3) * (
                abs(np.asarray(self.inverse_eval) - (seq_i * self.beta1 + self.beta0)) ** 3)
        return item1 * item2.sum()

    def _func_n(self, n):
        assert len(self.inverse_eval) == n

        x_hat = self._x_hat()
        item1_denominator = ((1 / n + x_hat ** 2 / _func_a(n)) ** 1.5) * (self.variance ** 1.5) * (_func_a(n) ** 3)
        item1 = self.b2 / item1_denominator
        seq_i = np.arange(1, n + 1)
        item2_1 = abs((_func_a(n) / n) - x_hat * (seq_i - x_hat)) ** 3
        item2_2 = abs(np.asarray(self.inverse_eval) - (seq_i * self.beta1 + self.beta0)) ** 3
        item2 = item2_1 * item2_2
        return item1 * item2.sum()

    def _x_hat(self):
        return (self.count + 1) / 2


class RandomOptimization(Optimization):

    def __init__(self, model_generator, name=None):
        super().__init__(model_generator, name)

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
