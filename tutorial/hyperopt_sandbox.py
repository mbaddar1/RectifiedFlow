# Tutorial : https://github.com/hyperopt/hyperopt/wiki/FMin
import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def objective(x):
    return {
        'loss': x ** 2,
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time(),
        'other_stuff': {'type': None, 'value': [0, 1, 2]},
        # -- attachments are handled differently
        'attachments':
            {'time_module': pickle.dumps(time.time)}
    }


if __name__ == '__main__':
    trials = Trials()
    best = fmin(objective,
                space=hp.uniform('x', -10, 10),
                algo=tpe.suggest,
                max_evals=100,
                trials=trials)
    min_loss = trials.best_trial['result']['loss']
    print(f'min_loss = {min_loss}')
    print(best)
