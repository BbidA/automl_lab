import pandas as pd
import pylab

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


class ProposedDataProcessor:

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


if __name__ == '__main__':
    # process_data()
    p = ProposedDataProcessor('/Users/jundaliao/Desktop/AutoML Log/proposed-new')
    proposed_results = ['proposed-new4', 'proposed-new5', 'proposed-new6']
    r = p.process_result(proposed_results)
    a = p.aggregate_result(proposed_results)
    assert isinstance(a, pd.DataFrame)
    a.to_csv('log/final_result_aggregation.csv')
    for rf in r:
        rf.to_csv('log/final_result.csv', mode='a')
    # a = get_best_betas()
    # for d in a:
    #     print('(DataSet(\'{}\'), {}),'.format(d, a[d]))
