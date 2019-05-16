import ast
import os

import numpy as np
import pandas as pd
import pylab
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import framework.sk_models as sk
from framework.base import data_collector
from utils.data_loader import DataSet

ROOT_PATH = '/Users/jundaliao/Desktop/AutoML Log/proposed0420/'


def _get_betas(start, end, steps=50):
    return [start + (end - start) * i / steps for i in range(steps)]


_data_betas = {
    # 'balanceScale': _get_betas(0.35, 0.75),
    # 'ecoli': _get_betas(0.4, 0.6),
    # 'cylinder': _get_betas(0.4, 0.6),
    # 'glass': _get_betas(0.4, 0.6),
    # 'messidor': _get_betas(0.35, 0.7),
    # 'car': _get_betas(0.  5, 0.7),
    # 'chess': _get_betas(0.5, 0.7),
    # 'spambase': _get_betas(0.65, 0.85),
    # 'statlogSegment': _get_betas(0.5, 0.7),
    # 'wdbc': _get_betas(0.6, 0.8),
    # 'wilt': _get_betas(0.7, 0.95),
    'nursery': _get_betas(0.45, 0.55)
    # 'nursery': _get_betas(0.45, 0.55)
}

models = {
    'adaboost': sk.AdaBoost(),
    'extratrees': sk.ExtraTree(),
    'randomforest': sk.RandomForest(),
    'decisiontree': sk.DecisionTree(),
    'gaussiannb': sk.GaussianNB(),
    'bernoullinb': sk.BernoulliNB(),
    'sgd': sk.SGD(),
    'knearestneighbors': sk.KNeighbors(),
    'passiveaggressive': sk.PassiveAggressive()
}

_ground_truth_models = {
    'ecoli': ('RandomForest', 'GaussianNB'),
    'cylinder': ('BernoulliNB',),
    'glass': ('RandomForest',),
    'messidor': ('PassiveAggressive', 'AdaBoost'),
    'balanceScale': ('PassiveAggressive',),
    'car': ('RandomForest',),
    'chess': ('AdaBoost', 'PassiveAggressive'),
    'spambase': ('AdaBoost', 'RandomForest'),
    'statlogSegment': ('RandomForest',),
    'wdbc': ('AdaBoost', 'RandomForest'),
    'wilt': ('AdaBoost', 'RandomForest', 'DecisionTree'),
    'nursery': ('AdaBoost', 'PassiveAggressive')
}

_betas = {
    'balanceScale': 0.366,
    'ecoli': 0.41200000000000003,
    'cylinder': 0.492,
    'glass': 0.432,
    'messidor': 0.45499999999999996,
    'car': 0.584,
    'chess': 0.536,
    'wilt': 0.7949999999999999,
    'nursery': 0.47200000000000003,
    'spambase': 0.6900000000000001,
    'statlogSegment': 0.6599999999999999,
    'wdbc': 0.604
}


def underscore_to_camelcase(value: str):
    components = value.split('_')
    return components[0].title() + ''.join(x.title() for x in components[1:])


def read_lab_data(data_name) -> pd.DataFrame:
    df_result = pd.read_csv(ROOT_PATH + 'find-beta--{}-0.01-20.0-total.csv'.format(data_name))
    df_result['betas'] = _data_betas[data_name]
    return _calculate_exploitation_rate(df_result, data_name)


def _calculate_exploitation_rate(df_result: pd.DataFrame, data_name: str):
    gt_models = _ground_truth_models[data_name]
    ex_rates = {'total': []}
    for m in gt_models:
        ex_rates[m] = []

    for beta in df_result['betas']:
        budget_file = pd.read_csv(ROOT_PATH + 'proposed_0.01_20.0_{}_{}.csv'.format(beta, data_name))
        budget_file = budget_file.set_index(budget_file['name'])
        # calculate exploitation rate
        exploitation_count = 0
        for gt_model in _ground_truth_models[data_name]:
            model_count = budget_file['budget'][gt_model]
            exploitation_count += model_count
            ex_rates[gt_model].append(model_count / 1000)
        ex_rates['total'].append(exploitation_count / 1000)
    for (model, rates) in ex_rates.items():
        df_result['ex_{}'.format(model)] = rates
    return df_result


def draw_chart(df_result: pd.DataFrame):
    pylab.plot(df_result['beta'], df_result['test_v'])
    pylab.savefig('log/proposed/')


def process_data():
    for data in _data_betas:
        lab_data = read_lab_data(data)
        pylab.title('test v & best v')
        pylab.xlabel('beta')
        pylab.ylabel('accuracy')
        pylab.plot(lab_data['betas'], lab_data['test_v'], '.-')
        pylab.plot(lab_data['betas'], lab_data['best_v'], '.:')
        pylab.legend(['test accuracy', 'best validation'])
        pylab.savefig('log/proposed/{}_vt.svg'.format(data))
        pylab.show()

        pylab.title('exploitation rate on {}'.format(_ground_truth_models[data]))
        pylab.xlabel('beta')
        pylab.ylabel('exploitation rate')
        legend = []
        for column in lab_data.columns.values:
            if 'ex' in column:
                line_style = '.-' if 'total' in column else '.:'
                pylab.plot(lab_data['betas'], lab_data[column], line_style)
                legend.append(column[column.index('ex_'):])
        pylab.legend(legend)
        pylab.savefig('log/proposed/{}_ex.svg'.format(data))
        pylab.show()

        generate_markdown(data, list(lab_data['best_model']))


def generate_markdown(data, best_models):
    print('## {}'.format(data))
    print('### best validation & test accuracy')
    print('Models from left to right are: \n```\n{}\n```'.format(best_models))
    print('![](img/{}_vt.svg)'.format(data))
    print('### exploitation rate')
    print('![](img/{}_ex.svg)'.format(data))


def get_best_betas():
    betas = {}
    for data in _data_betas:
        lab_data = read_lab_data(data)[['betas', 'ex_total', 'best_v']]
        lab_data = lab_data.set_index(lab_data['betas'])
        betas[data] = lab_data['ex_total'].idxmax()
    print(betas)
    return betas


class ERUCBDataProcessor:

    def __init__(self, root, betas=None, ground_truth_model=None):
        self.root = root
        self.betas = betas if betas else _betas
        self.gt_models = ground_truth_model if ground_truth_model else _ground_truth_models

    def process_result(self, dirs):
        df_results = []
        for directory in dirs:
            path = '{}/{}'.format(self.root, directory)
            df_result = pd.read_csv('{}/proposed-0.01-20.0-total.csv'.format(path))
            df_result = df_result.set_index(df_result['data set'])
            df_result['betas'] = pd.Series(self.betas)
            df_result = df_result.drop(['Unnamed: 0', 'data set'], axis=1)

            # calculate exploitation rate
            ex_rates = {}
            ex_rate_info = {}
            for (data, beta) in self.betas.items():
                budget_file = pd.read_csv('{}/proposed_0.01_20.0_{}_{}.csv'.format(path, beta, data))
                budget_file = budget_file.set_index(budget_file['name'])
                # calculate exploitation rate
                exploitation_count = 0
                separate_info = ''
                for gt_model in self.gt_models[data]:
                    model_count = budget_file['budget'][gt_model]
                    exploitation_count += model_count
                    separate_info += '({}: {}) '.format(gt_model, model_count / 1000)
                ex_rates[data] = exploitation_count / 1000
                ex_rate_info[data] = separate_info
            df_result['ex_rate'] = pd.Series(ex_rates)
            df_result['ex_rate_detail'] = pd.Series(ex_rate_info)
            df_results.append(df_result)
        return df_results

    def aggregate_result(self, dirs):
        processed_results = self.process_result(dirs)
        res = pd.concat(self.process_result(dirs)).groupby('data set').mean()
        d1 = processed_results[0]
        for df in processed_results[1:]:
            d1['best_model'] = d1['best_model'] + ', ' + df['best_model']
        res['best_model'] = d1['best_model']
        return res


class ProposedMethodDataProcessor:

    def __init__(self, root):
        self.root = root

    def process(self, prefix):
        df_results = []
        for root, dirs, files in os.walk(self.root):
            dir_name = os.path.basename(root)
            if dir_name[:dir_name.rfind('-')] == prefix:
                df_results.append(self._process_total_statistics(root, files))

        # aggregate results
        return pd.concat(df_results, ignore_index=True).sort_values('data set').reset_index(drop=True)

    @staticmethod
    def _process_total_statistics(root, files: list):
        # find the total statistics file
        df_details = {}
        df_total = None
        for file in files:
            file_path = os.path.join(root, file)
            if 'total' in file:
                df_total = pd.read_csv(file_path, index_col=0)
            elif file.endswith('csv') and 'process' not in file:
                data_set_name = file[file.rindex('_') + 1: file.rindex('.')]
                if not data_set_name:
                    raise ValueError('Dataset name is not found')

                df_details[data_set_name] = pd.read_csv(file_path, index_col=0)

        if df_total is None:
            raise ValueError('Total statistics file not exists')

        # assign budget information to df_total
        for index, row in df_total.iterrows():
            detail = df_details[row['data set']]
            budget = detail.loc[detail['name'] == row['best_model']]['budget'].values[0]
            df_total.at[index, 'best_model'] = '{}: {}'.format(row['best_model'], budget)

        return df_total


class AutoSkParser:

    def __init__(self, root):
        self.root = root
        self.data_name_prefix = 'Start fitting on '
        self.test_v_prefix = 'Test v is '
        self.best_v_prefix = 'Best validation score: '
        self.model_info_prefix = 'SimpleClassificationPipeline('
        self.model_name_key = 'classifier:__choice__'

        self.validation_kf = StratifiedKFold(n_splits=5, shuffle=False)

    def process(self, prefix):
        results = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.startswith(prefix) and file.endswith('.log'):
                    file_path = os.path.join(root, file)
                    results.append(self._process_log_file(file_path))

        result = pd.concat(results).sort_values('data set').reset_index(drop=True)
        result['test_v'] = result['test_v'].astype(float)
        result['best_v'] = result['best_v'].astype(float)

        return result

    def _process_log_file(self, file_path):
        result = []
        with open(file_path, mode='r') as f:
            start_search = False
            curr = {}
            for line in f.readlines():
                if line.strip('\n').endswith('============'):
                    if start_search:
                        result.append(curr)
                        curr = {}
                    start_search = not start_search

                if start_search:
                    if self.data_name_prefix in line:
                        curr['data set'] = line.split(self.data_name_prefix)[1].strip('\n')
                    elif self.test_v_prefix in line:
                        curr['test_v'] = line.split(self.test_v_prefix)[1].strip('\n')
                    elif self.best_v_prefix in line:
                        curr['origin_best_v'] = line.split(self.best_v_prefix)[1].strip('\n')
                    elif self.model_info_prefix in line:
                        start = line.find(self.model_info_prefix)
                        start += len(self.model_info_prefix)

                        model_info = ast.literal_eval(line.strip('\n')[start: -1])
                        curr['best_model'] = underscore_to_camelcase(model_info[self.model_name_key])
                        curr['best_v'] = self._eval_best_v(model_info, curr['data set'])

        columns = ['data set', 'best_v', 'origin_best_v', 'best_model', 'test_v']
        return pd.DataFrame(data=result, columns=columns)

    def _eval_best_v(self, model_info: dict, data_name: str):
        model_name = underscore_to_camelcase(model_info[self.model_name_key])
        model_generator = models[model_name.lower()]
        param_prefix = 'classifier:{}:'.format(model_name)
        actual_params = []
        for key in model_info:
            if param_prefix in key:
                param = key.split(':')[2]
                if model_name == 'AdaBoost'.lower() and param == 'max_depth':
                    actual_params.append(('base_estimator', DecisionTreeClassifier(max_depth=model_info[key])))
                else:
                    actual_params.append((param, model_info[key]))

        model = model_generator.generate_model(None, actual_params=actual_params)
        print(model)

        data = DataSet(data_name)
        train_x, train_y = data.train_data()

        eval_values = []
        for train_index, valid_index in self.validation_kf.split(train_x, train_y):
            x, y = data_collector(train_index, train_x, train_y)
            valid_x, valid_y = data_collector(valid_index, train_x, train_y)

            try:
                model = model.fit(x, y)
            except ValueError as e:  # temporally just catch ValueError
                print("Parameter wrong, return 0, error message: {}".format(e))
                return 0

            predictions = model.predict(valid_x)

            eval_value = accuracy_score(valid_y, predictions)
            eval_values.append(eval_value)

        return np.mean(np.array(eval_values))


if __name__ == '__main__':
    w_dir = '/Users/jundaliao/Desktop/AutoML Log/New/0515/autosk0515'
    processor = AutoSkParser(w_dir)
    a = processor.process('autosk')
    a.to_csv('{}/final-autosk.csv'.format(w_dir))
    prefix_list = ['eg', 'ucb', 'sf', 'proposed-new', 'erucb']
    # for p in prefix_list:
    #     a = processor.process(p)
    #     a.to_csv('{}/final-{}.csv'.format(w_dir, p))
