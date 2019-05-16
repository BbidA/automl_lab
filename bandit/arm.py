import numpy as np
import scipy.stats as stats
from bandit.model_optimization import Optimization
from bandit.model_selection import ERUCB, SoftMaxSelection, EpsilonGreedySelection, BanditModelSelection
import pandas as pd


def _func_a(n):
    return (n ** 3 - n) / 12


class Arm:
    def __init__(self, beta1, beta0, mu=0, sigma=1):
        self.beta1 = beta1
        self.beta0 = beta0
        self.mu = mu
        self.sigma = sigma

        self.count = 0

        self.name = 'Arm{}{}{}{}'.format(beta1, beta0, mu, sigma)

    def linerf(self):
        return self.beta1 * self.count + self.beta0

    def noise(self):
        return np.random.normal(self.mu, self.sigma, 1)[0]

    def sample(self):
        self.count += 1
        return self.linerf() + self.noise()


class ProposedMethodArm(Optimization):

    def __init__(self, arm, name, b1=1.0, b2=1.0, alpha=0.25):
        super().__init__(None, name)

        self.arm = arm
        self.name = name

        self.count = 0

        self.beta0 = 0
        self.beta1 = 0
        self.variance = 0
        self.e_beta1 = 0
        self.e_beta0 = 0
        self.e_variance = 0
        self.eval_results = []
        self.mu = 0

        self.b1 = b1
        self.b2 = b2
        self.alpha = alpha

        self.param_change_info = []
        self._m_value = 0
        self._n_value = 0
        self._e_beta0_item3 = 0

    @property
    def sigma(self):
        return self.instances['Accuracy'].std()

    def clear(self):
        return

    def run_one_step(self, train_x=None, train_y=None):
        eval_result = self.arm.sample()
        self.eval_results.append(eval_result)

        self.mu = (self.mu * self.count + eval_result) / (self.count + 1)
        self.instances.loc[self.count] = [None, None, eval_result, None]

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
        if len(self.eval_results) == 1:
            return
        self.beta1, self.beta0, *_ = stats.linregress(range(1, self.count + 1), self.eval_results)

        # update variance
        y_hat = [self.beta1 * i + self.beta0 for i in range(1, self.count + 1)]
        self._update_variance(y_hat)

    def _update_variance(self, y_hat):
        y = np.asarray(self.eval_results)
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
        item1 = self.b1 / ((self.variance * _func_a(n)) ** 1.5)
        x_hat = self._x_hat()
        seq_i = np.arange(1, n + 1)
        item2 = (abs(seq_i - x_hat) ** 3) * (
                abs(np.asarray(self.eval_results) - (seq_i * self.beta1 + self.beta0)) ** 3)
        return item1 * item2.sum()

    def _func_n(self, n):
        x_hat = self._x_hat()
        item1_denominator = ((1 / n + x_hat ** 2 / _func_a(n)) ** 1.5) * (self.variance ** 1.5) * (_func_a(n) ** 3)
        item1 = self.b2 / item1_denominator
        seq_i = np.arange(1, n + 1)
        item2_1 = abs((_func_a(n) / n) - x_hat * (seq_i - x_hat)) ** 3
        item2_2 = abs(np.asarray(self.eval_results) - (seq_i * self.beta1 + self.beta0)) ** 3
        item2 = item2_1 * item2_2
        return item1 * item2.sum()

    def _x_hat(self):
        return (self.count + 1) / 2


def artificial_test(method):
    arms = [
        Arm(1, 10, 0, 1),
        Arm(2, 1, 0, 1),
        Arm(0.5, 1, 0, 1)
    ]

    optimizations = [ProposedMethodArm(a, a.name, 0.01, 0.01) for a in arms]

    if method == 'new_er':
        model_selection = ERUCB(optimizations)
    elif method == 'sf':
        model_selection = SoftMaxSelection(optimizations)
    elif method == 'eg':
        model_selection = EpsilonGreedySelection(optimizations)
    elif method == 'ucb':
        model_selection = BanditModelSelection(optimizations, 'ucb')
    else:
        raise ValueError('Wrong method')

    best_optimization = model_selection.fit(None, None, 2000)

    for (prefix, param_info) in model_selection.param_change_info:
        assert isinstance(param_info, pd.DataFrame)
        with open('log/arm-process-newer.csv', mode='a') as f:
            f.write(prefix)
            param_info.to_csv(f, mode='a')
    model_selection.statistics().to_csv('log/arm-{}.csv'.format(method))


if __name__ == '__main__':
    artificial_test('new_er')
