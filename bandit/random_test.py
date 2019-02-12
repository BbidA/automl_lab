import time
import utils.logging_ as log

import framework.sk_models as sk
import utils.data_loader as data_loader
from bandit.model_selection import BanditModelSelection, RandomOptimization
from utils.data_loader import DataSet

logger = log.LoggerManager.get_logger('random')


def random_test():
    models = [
        sk.DecisionTree(),
        sk.AdaBoost(),
        sk.QuadraticDiscriminantAnalysis(),
        sk.GaussianNB(),
        sk.LinearSVC(),
        sk.KNeighbors(),
        sk.BernoulliNB(),
        sk.ExtraTree(),
        sk.MultinomialNB(),
        sk.PassiveAggressive(),
        sk.RandomForest(),
        sk.SGD()
    ]

    optimizations = [RandomOptimization(generator, type(generator).__name__) for generator in models]

    # get data sets
    # data_sets = [
    #     ('adult', data_loader.adult_dataset()),
    #     ('cmc', data_loader.cmc_dataset()),
    #     ('car', data_loader.car_dataset()),
    #     ('banknote', data_loader.banknote_dataset())
    # ]

    data_sets = [
        DataSet('adult'),
        DataSet('cmc'),
        DataSet('car'),
        DataSet('banknote')
    ]

    logger.info("Begin random search")
    random_selection = BanditModelSelection(optimizations, update_func='random')

    for data_set in data_sets:
        logger.info("Begin fitting {}".format(data_set.name))
        start = time.time()

        train_x, train_y = data_set.train_data()

        best_result = random_selection.fit(train_x, train_y, budget=1000)

        elapsed = time.time() - start
        logger.info("Fitting {} done, spend {}s".format(data_set.name, elapsed))
        logger.info("Selection result:\n{}".format(best_result))

        random_selection.statistics().to_csv('log/random.csv')


if __name__ == '__main__':
    random_test()
