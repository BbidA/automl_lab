import logging
import random
import signal
import time

import pandas as pd

import framework.base as base
from utils.logging_ import get_logger

# --------------------------------------------------------
# define a logger


log = get_logger('random_search', 'random_search.log', level=logging.INFO)


def timeout_handler(signum, frame):
    raise TimeoutError("Timeout!")


signal.signal(signal.SIGALRM, timeout_handler)


def random_search(model_generator, train_x, train_y, search_times=100):
    evaluator = base.ModelEvaluator(model_generator, train_x, train_y)
    model_name = type(model_generator).__name__
    raw_parameter_list = []
    actual_parameter_list = []
    accuracy_list = []
    time_list = []

    for i in range(search_times):
        # sample a set of parameters and evaluate
        raw_params, actual_params = random_sample_parameters(model_generator.hp_space)
        log.debug('[{}]: parameters: {}'.format(model_name, actual_params))
        try:
            start = time.time()
            signal.alarm(300)

            accuracy = evaluator.evaluate(raw_params)

            signal.alarm(0)
            elapsed = time.time() - start

            log.debug('accuracy: {}, spend {}s'.format(float(accuracy), elapsed))

            # add parameters and accuracy information to four lists
            raw_parameter_list.append(raw_params)
            accuracy_list.append(accuracy)
            actual_parameter_list.append(actual_params)
            time_list.append(elapsed)
        except TimeoutError:
            log.error('{} {} Timeout!'.format(model_name, actual_params))
        except Exception as e:
            log.error('Error message: {}'.format(e))
        except RuntimeWarning as w:
            log.warning('Warning message: {}'.format(w))

    # convert four lists to DataFrame
    result_data = list(zip(raw_parameter_list, actual_parameter_list, accuracy_list, time_list))
    return pd.DataFrame(data=result_data, columns=['Raw Parameters', 'Actual Parameters', 'Accuracy', 'Time'])


def random_sample_parameters(hp_space):
    assert len(hp_space) >= 0

    result = []
    actual_params = []

    for param in hp_space:
        assert isinstance(param, base.HyperParameter)

        if param.param_type == base.INTEGER or param.param_type == base.CATEGORICAL:
            curr_param = random.randint(*param.param_bound)
            result.append(curr_param)
        elif param.param_type == base.FLOAT:
            curr_param = random.uniform(*param.param_bound)
            result.append(random.uniform(*param.param_bound))
        else:
            assert False

        actual_params.append(param.convert_raw_param(curr_param))

    assert len(result) == len(hp_space)

    # result is used in evaluation(raw parameters), actual_params is used in logging
    return result, actual_params
