import pandas as pd
import os
import pylab

ROOT_PATH = '/Users/jundaliao/Downloads/log 2/proposed/'


def _get_betas(start, end, steps=50):
    return [start + (end - start) * i / steps for i in range(steps)]


_data_betas = {
    'balanceScale': _get_betas(0.4, 0.6),
    'ecoli': _get_betas(0.4, 0.6),
    'cylinder': _get_betas(0.4, 0.6),
    'glass': _get_betas(0.4, 0.6),
    'messidor': _get_betas(0.4, 0.6),
    'car': _get_betas(0.5, 0.7),
    'chess': _get_betas(0.5, 0.7),
    'spambase': _get_betas(0.65, 0.85),
    'statlogSegment': _get_betas(0.5, 0.7),
    'wdbc': _get_betas(0.6, 0.8),
    'wilt': _get_betas(0.7, 0.9),
    'nursery': _get_betas(0.5, 0.7)
}

_ground_truth_models = {
    'ecoli': 'RandomForest',
    'cylinder': 'ExtraTree',
    'glass': 'DecisionTree',
    'messidor': 'SGD',
    'balanceScale': 'SGD',
    'car': 'ExtraTree',
    'chess': 'AdaBoost',
    'spambase': 'AdaBoost',
    'statlogSegment': 'RandomForest',
    'wdbc': 'AdaBoost',
    'wilt': 'AdaBoost',
    'nursery': 'AdaBoost'
}


def read_lab_data(data_name):
    df_result = pd.read_csv(ROOT_PATH + 'find-beta--{}-0.01-20.0-total.csv'.format(data_name))
    df_result['betas'] = _data_betas[data_name]
    return _calculate_exploitation_rate(df_result, data_name)


def _calculate_exploitation_rate(df_result: pd.DataFrame, data_name: str):
    exploitation_rates = []
    for beta in df_result['betas']:
        budget_file = pd.read_csv(ROOT_PATH + 'proposed_0.01_20.0_{}_{}.csv'.format(beta, data_name))
        budget_file = budget_file.set_index(budget_file['name'])
        exploitation_rates.append(budget_file['budget'][_ground_truth_models[data_name]] / 1000)
    df_result['ex_r'] = exploitation_rates
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
        pylab.plot(lab_data['betas'], lab_data['ex_r'], '.-')
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


if __name__ == '__main__':
    process_data()
