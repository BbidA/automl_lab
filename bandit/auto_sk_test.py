import time
import sys

import autosklearn.classification
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import utils.data_loader as data_loader
from utils.logging_ import get_logger

# other parameters
exclude_estimators = [
    'gradient_boosting',
    'lda',
    'libsvm_svc',
    'xgradient_boosting',
    'multinomial_nb',
    'liblinear_svc'
]


def auto_sk_lab(start, end):
    logger = get_logger('auto-sklearn', 'log/auto_sk/auto-sk.log')

    result = []
    data_sets = data_loader.data_for_auto_sklearn()[start:end]
    for (data, time_left) in data_sets:
        logger.info('Start fitting {}'.format(data.name))
        start = time.time()

        method_result = auto_sk_method(data, time_left)
        result.append(method_result)

        logger.info('Fitting on {} is over, spend {}s\n'
                    'result:\n'
                    '{}'.format(data.name, time.time() - start, method_result))

    df_result = pd.DataFrame(data=result, columns=['data set', 'best v', 'test v'])
    df_result.to_csv('log/auto_sk/auto-sk-{}to{}.csv'.format(start, end))
    df_result.to_pickle('log/auto_sk/auto-sk-{}tp{}.pkl'.format(start, end))


def auto_sk_method(data, time_left):
    classifier = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=time_left,
        per_run_time_limit=300,
        exclude_estimators=exclude_estimators)

    logger = get_logger('log/auto_sk/auto_sk_{}'.format(data.name), 'log/auto_sk/auto_sk_{}.log'.format(data.name))

    train_x, train_y = data.train_data()

    logger.info('Start fitting on {}'.format(data.name))
    start = time.time()

    classifier.fit(train_x, train_y)

    # get best validation score
    idx_best_run = np.argmax(classifier.cv_results_['mean_test_score'])
    best_score = classifier.cv_results_['mean_test_score'][idx_best_run]

    # calculate test v
    test_x, test_y = data.test_data()
    y_hat = classifier.predict(test_x)
    test_v = accuracy_score(test_y, y_hat)

    # show result information
    logger.info('Fitting on {} is done, spend {}s'.format(data.name, time.time() - start))
    logger.info('Sprint statistics\n{}'.format(classifier.sprint_statistics()))
    logger.info('Test V is {}'.format(test_v))
    # logger.info('Show model:\n{}'.format(classifier.show_models()))

    # save cv results
    cv_result = pd.DataFrame.from_dict(classifier.cv_results_)
    cv_result.to_csv('log/auto_sk/auto_sk_cv_result_on_{}.csv'.format(data.name))
    cv_result.to_pickle('log/auto_sk/auto_sk_cv_result_on_{}.pkl'.format(data.name))

    return data.name, best_score, test_v


if __name__ == '__main__':
    # arg_start = int(sys.argv[1])
    # arg_end = int(sys.argv[2])
    auto_sk_lab(0, 4)

# save whole model
# joblib.dump(classifier, 'log/auto_sk/auto_sk_{}.joblib'.format(data.name))
