import multiprocessing as mp
import pickle
import sys
import time
from logging import INFO, DEBUG

import pandas as pd

import framework.sk_models as sk
import utils.data_loader as data_loader
from bandit.model_optimization import RacosOptimization
from bandit.model_selection import BanditModelSelection, EpsilonGreedySelection, SoftMaxSelection, ERUCB
from framework.param_search import random_search
from utils.logging_ import get_logger

ALL_DATA = data_loader.all_data(
    exclude=['adult', 'banknote', 'credit', 'egg', 'flag', 'seismic', 'wpbc', 'yeast', 'magic04'])
PROPOSED_DATA = data_loader.data_for_proposed_method()
BETA_FINDING_DATA = data_loader.data_for_beta_finding()

BUDGET = 100
B = 0.001
GROUND_TRUTH_PKL = 'log/ground_truth.pkl'
CORES = 1  # one thread see function one_thread_lab(method)

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


def parameter_test():
    result = []
    theta = float(sys.argv[1])
    gamma = float(sys.argv[2])
    begin = int(sys.argv[3])
    end = int(sys.argv[4])
    for (data, beta_array) in BETA_FINDING_DATA[begin:end]:
        for beta in beta_array:
            result.append(proposed_method(data, theta, gamma, beta))
        csv_file = 'log/proposed/find-beta--{}-{}-{}-total.csv'.format(data.name, theta, gamma)
        df_result = pd.DataFrame(data=result, columns=['data set', 'best_v', 'best_model', 'test_v'])
        df_result.to_csv(csv_file)
        result.clear()


def one_thread_lab(method):
    result = []
    if method == 'proposed':
        theta = 0.01
        gamma = 20.0
        for (data, beta) in PROPOSED_DATA:
            result.append(proposed_method(data, theta, gamma, beta))
        csv_file = 'log/proposed-new/proposed-{}-{}-total.csv'.format(theta, gamma)
    elif method == 'new_er':
        for (data, _) in PROPOSED_DATA:
            result.append(new_erucb_method(data))
        csv_file = 'log/proposed-new/new-er-total.csv'
    elif method == 'ground':
        start = int(sys.argv[2])
        end = int(sys.argv[3])
        for data in ALL_DATA[start:end]:
            result.append(ground_truth_method(data))
        csv_file = 'log/ground/ground-total-statistics-{}to{}.csv'.format(start, end)
    elif method == 'full':
        for (data, _) in PROPOSED_DATA:
            for model in model_generators:
                single_arm_method(data, model)
        return
    else:
        for (data, _) in PROPOSED_DATA:
            if method == 'ucb' or method == 'random':
                result.append(ucb_or_random_method(data, method))
            elif method == 'eg':
                result.append(eg_method(data))
            elif method == 'sf':
                result.append(softmax_method(data))
        csv_file = 'log/{}/{}-total-statistics.csv'.format(method, method)

    df_result = pd.DataFrame(data=result, columns=['data set', 'best_v', 'best_model', 'test_v'])
    df_result.to_csv(csv_file)


def ground_truth_method(data):
    logger = get_logger('gt', 'log/ground/ground_truth.log', level=INFO)

    result = []
    budget_for_single_model = int(BUDGET / len(model_generators))
    logger.info('Begin fitting on {}'.format(data.name))
    start = time.time()

    for model_generator in model_generators:
        result.append(find_ground_truth(data, model_generator, budget_for_single_model))

    logger.info('Fitting on {} is over, spend {}s'.format(data.name, time.time() - start))

    df_result = pd.DataFrame(data=result, columns=['model', 'best v', 'mean', 'std', 'best model', 'time'])
    df_result.to_csv('log/ground/ground_{}.csv'.format(data.name))

    # get test v
    best_model_index = df_result['best v'].idxmax()
    best_model = df_result['best model'][best_model_index]
    test_v = _evaluate_test_v(data, best_model)
    logger.info('Test v of {} is {}'.format(data.name, test_v))

    return data.name, df_result['best v'].max(), type(best_model).__name__, test_v


def single_arm_method(data, model_gen, budget=BUDGET):
    model_name = type(model_gen).__name__
    optimization = RacosOptimization(model_gen, model_name)
    train_x, train_y = data.train_data()
    logger = get_logger('single_arm', 'log/single/single_arm.log')
    logger.info(f'Begin to fit {data.name} using {model_name}')
    for i in range(budget):
        logger.info(f'Process: {i + 1}/{budget}')
        optimization.run_one_step(train_x, train_y)
    logger.info(f'Fitting on {data.name} using model {model_name} is over')

    optimization.instances.to_csv(f'log/single/single_{data.name}_{model_name}.csv')


def find_ground_truth(data, model_generator, budget=BUDGET):
    """Find the ground truth model for each dataset

    Parameters
    ----------

    data: utils.data_loader.DataSet
        training data

    model_generator: framework.base.ModelGenerator
        generator for the target model

    budget: int
        number of samples

    Returns
    -------

    evaluation_result: (float, float, float)
        best evaluation result, mean and standard deviation

    """
    train_x, train_y = data.train_data()
    model_name = type(model_generator).__name__
    start = time.time()
    log = get_logger('gt.model', '', level=INFO)
    log.info('{} --- {} start fitting'.format(data.name, model_name))

    # begin sampling
    result = random_search(model_generator, train_x, train_y, search_times=budget)

    best_result_index = result['Accuracy'].idxmax()
    best_result_params = result['Raw Parameters'][best_result_index]
    best_model = model_generator.generate_model(best_result_params)

    elapsed = time.time() - start
    log.info('{} --- {} end running, spend {}s'.format(data.name, model_name, elapsed))
    acc_column = result['Accuracy']
    return model_name, acc_column.max(), acc_column.mean(), acc_column.std(), best_model, elapsed


def ground_truth_lab():
    log = get_logger('gt', 'log/gt.log', level=INFO)
    for data in ALL_DATA:
        start = time.time()
        log.info('Start finding ground truth model for data set {}'.format(data.name))

        result = []
        for generator in model_generators:
            result.append(find_ground_truth(data, generator))
        df_result = pd.DataFrame(data=result, columns=['name', 'max', 'mean', 'std', 'best_model', 'time'])
        df_result = df_result.set_index(df_result['name'])
        best_model = df_result['max'].idxmax()

        # save to csv
        with open('log/gt_{}.csv'.format(data.name), 'a') as f:
            f.write('best is {}\n'.format(best_model))
            df_result.to_csv(f, mode='a')

        elapsed = time.time() - start
        log.info('g-test --- Fitting on {} is over, spend {}s'.format(data.name, elapsed))


def ucb_lab(method):
    with mp.Pool(processes=CORES) as pool:
        result = pool.starmap(ucb_or_random_method, [(data, method) for data in ALL_DATA])
        df_result = pd.DataFrame(data=result, columns=['data set', 'best_v', 'best_model', 'test_v'])
        df_result.to_csv('log/{}_lab.csv'.format(method))
        df_result.to_pickle('log/{}_lab.pkl'.format(method))


def eg_or_sf_lab(method, record_file):
    all_data = ALL_DATA
    with mp.Pool(processes=CORES) as pool:
        result = pool.map(method, all_data)
        df_result = pd.DataFrame(data=result, columns=['data set', 'best_v', 'best_model', 'test_v'])
        df_result.to_csv('log/{}_lab.csv'.format(record_file))
        df_result.to_pickle('log/{}_lab.pkl'.format(record_file))


def ucb_or_random_method(data, method):
    """Do model selection by traditional ucb method

    Parameters
    ----------

    data: utils.data_loader.DataSet
        training data

    method: str
        model selection method (only ucb or random can be chosen)

    """
    log = get_logger(method, 'log/{}/{}.log'.format(method, method), level=DEBUG)

    optimizations = _get_optimizations()
    model_selection = BanditModelSelection(optimizations, method)

    log.info('Begin fit on {}'.format(data.name))
    train_x, train_y = data.train_data()

    start = time.time()

    best_optimization = model_selection.fit(train_x, train_y, budget=BUDGET)

    log.info('Fitting on {} is done! Spend {}s'.format(data.name, time.time() - start))

    csv_file = 'log/{}/{}_{}.csv'.format(method, method, data.name)
    pkl_file = 'log/{}/{}_{}.pkl'.format(method, method, data.name)
    return _get_test_result(best_optimization, data, model_selection.statistics(), csv_file, pkl_file, log)


def proposed_lab():
    theta = float(sys.argv[2])
    gamma = float(sys.argv[3])

    all_data = PROPOSED_DATA
    with mp.Pool(processes=CORES) as pool:
        result = pool.starmap(proposed_method, [(data, theta, gamma, beta) for (data, beta) in all_data])
        df_result = pd.DataFrame(data=result, columns=['data set', 'best_v', 'best_model', 'test_v'])
        df_result.to_csv('log/proposed/proposed_{}_{}.csv'.format(theta, gamma))
        df_result.to_pickle('log/proposed/proposed_{}_{}.pkl'.format(theta, gamma))


def new_erucb_method(data, b=B):
    log_name = 'new-erucb'
    log = get_logger(log_name, 'log/proposed-new/' + log_name + '.log', level=DEBUG)

    model_selection = _get_model_selection(b)

    log.info('Begin fit on {}'.format(data.name))
    train_x, train_y = data.train_data()
    start = time.time()

    best_optimization = model_selection.fit(train_x, train_y, budget=BUDGET)

    elapsed = time.time() - start
    log.info('Fitting on {} ends, spend {}s'.format(data.name, elapsed))
    for (prefix, param_info) in model_selection.param_change_info:
        assert isinstance(param_info, pd.DataFrame)
        with open('log/proposed-new/erucb-process-{}-{}.csv'.format(data.name, b), mode='a') as f:
            f.write(prefix)
            param_info.to_csv(f, mode='a')

    csv = 'log/proposed-new/new_erucb_{}_{}.csv'.format(data.name, b)
    return _get_test_result(best_optimization, data, model_selection.statistics(), csv, '', log)


def proposed_method(data, theta, gamma, beta, show_selection_detail=False):
    """Do model selection with proposed method

    Parameters
    ----------
    data: utils.data_loader.DataSet
        training data

    theta: float

    gamma: float

    beta: float

    show_selection_detail: bool
    """
    log_name = 'proposed-{}-{}'.format(theta, gamma)
    log = get_logger(log_name, 'log/proposed-new/' + log_name + '.log', level=DEBUG)

    optimizations = _get_optimizations()
    model_selection = BanditModelSelection(optimizations, 'new', theta=theta, gamma=gamma, beta=beta)

    log.info('Begin fit on {}'.format(data.name))
    train_x, train_y = data.train_data()

    start = time.time()
    best_optimization = model_selection.fit(train_x, train_y, budget=BUDGET)

    # write parameter change information
    if show_selection_detail:
        with open('log/ps_{}_{}_{}.csv'.format(theta, gamma, data.name), 'a') as f:
            count = len(model_generators)
            for record in model_selection.param_change_info:
                f.write('t = {}'.format(count))
                record.to_csv(f, mode='a')
                f.write('\n\n')

                count += 1

    log.info('Fitting on {} is over, spend {}s'.format(data.name, time.time() - start))

    csv_file = 'log/proposed-new/proposed_{}_{}_{}_{}.csv'.format(theta, gamma, beta, data.name)
    pkl_file = 'log/proposed-new/proposed_{}_{}_{}_{}.pkl'.format(theta, gamma, beta, data.name)

    return _get_test_result(best_optimization, data, model_selection.statistics(), csv_file, pkl_file, log)


def eg_method(data):
    """Do model selection with epsilon-greedy method

    Parameters
    ----------
    data: utils.data_loader.DataSet
        training data

    """

    log = get_logger('epsilon-greedy', 'log/eg/epsilon-greedy.log', level=DEBUG)

    optimizations = _get_optimizations()
    model_selection = EpsilonGreedySelection(optimizations)

    log.info('Begin fitting on {}'.format(data.name))
    train_x, train_y = data.train_data()

    start = time.time()
    best_optimization = model_selection.fit(train_x, train_y, budget=BUDGET)
    elapsed = time.time() - start

    log.info('Fitting on {} is over, spend {}s'.format(data.name, elapsed))

    csv_file = 'log/eg/eg_{}.csv'.format(data.name)
    pkl_file = 'log/eg/eg_{}.pkl'.format(data.name)

    return _get_test_result(best_optimization, data, model_selection.statistics(), csv_file, pkl_file, log)


def softmax_method(data):
    """Do model selection with softmax method

    Parameters
    ----------
    data: utils.data_loader.DataSet
        training data

    """

    log = get_logger('softmax', 'log/sf/softmax.log', level=DEBUG)

    optimizations = _get_optimizations()
    model_selection = SoftMaxSelection(optimizations)

    log.info('Begin fitting on {}'.format(data.name))
    train_x, train_y = data.train_data()

    start = time.time()
    best_optimization = model_selection.fit(train_x, train_y, temperature=0.5, budget=BUDGET)
    elapsed = time.time() - start

    log.info('Fitting on {} is over, spend {}s'.format(data.name, elapsed))

    csv_file = 'log/sf/sf_{}.csv'.format(data.name)
    pkl_file = 'log/sf/sf_{}.pkl'.format(data.name)

    return _get_test_result(best_optimization, data, model_selection.statistics(), csv_file, pkl_file, log)


def calculate_exploitation_rate(data, budget_statistics):
    """Calculate exploitation rate

    Parameters
    ----------

    budget_statistics: pandas.DataFrame
        statistics of budget

    data: utils.data_loader.DataSet
        target data set

    Returns
    -------

    exploitation_rate: float
        exploitation rate of this method

    """
    # read ground truth model information
    with open(GROUND_TRUTH_PKL, 'rb') as f:
        ground_truth_models = pickle.load(f)

    assert isinstance(ground_truth_models, dict)
    gt_model = ground_truth_models[data.name]  # get ground truth model's name
    assert isinstance(gt_model, str)

    # get budget and calculate exploitation rate
    budget = budget_statistics['budget'][gt_model]

    return budget / BUDGET


def _get_test_result(best_optimization, data, statistics, csv_file, pkl_file, log):
    # save statistics to csv
    statistics.to_csv(csv_file)
    # save to pickle file for calculating exploitation rate
    # statistics.to_pickle(pkl_file)

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


def _get_optimizations(b=B):
    return [RacosOptimization(generator, type(generator).__name__, b1=b, b2=b) for generator in
            model_generators]


def _get_model_selection(b=B):
    optimizations = _get_optimizations(b)
    return ERUCB(optimizations)


def _evaluate_test_v(data, model):
    """Use selected model to fit on the data

    Parameters
    ----------

    data: utils.data_loader.DataSet
        training and test data

    model: classifier
        selected model

    Returns
    -------

    test_v: float
        accuracy on test data
    """
    train_x, train_y = data.train_data()
    model.fit(train_x, train_y)

    test_x, test_y = data.test_data()
    return model.score(test_x, test_y)


if __name__ == '__main__':
    # method_choice = sys.argv[1]
    # if method_choice == 'ground':
    #     ground_truth_lab()
    # elif method_choice == 'ucb':
    #     ucb_lab('ucb')
    # elif method_choice == 'sf':
    #     eg_or_sf_lab(softmax_method, 'sf')
    # elif method_choice == 'eg':
    #     eg_or_sf_lab(eg_method, 'eg')
    # elif method_choice == 'random':
    #     ucb_lab('random')
    # elif method_choice == 'proposed':
    #     proposed_lab()
    # proposed_method(data_loader.DataSet('messidor'), 0.01, 20, 0.65)
    # parameter_test()
    # nursery_data_set = data_loader.DataSet('nursery')
    # t_x, t_y = nursery_data_set.train_data()
    #
    # print(t_x[0:10])
    # print(t_y[0:10])
    # qda = sk.QuadraticDiscriminantAnalysis()
    # random_search(qda, t_x, t_y, search_times=10)
    # ground_truth_lab()
    m_opt = sys.argv[1]
    one_thread_lab(m_opt)
