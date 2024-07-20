import pickle
import pandas as pd


def convert_trial_object_to_row(trial_obj: dict):
    row = dict()
    row['loss'] = trial_obj['result']['loss']
    r = trial_obj['misc']['vals']['r'][0]
    d = trial_obj['misc']['vals']['d'][0]
    row['r'] = r
    row['d'] = d
    return row


if __name__ == '__main__':
    with open("hopt_trials_2024-07-14T22:16:30.703369.hopt", "rb") as f:
        obj = pickle.load(f)
        row_list = list(map(lambda x: convert_trial_object_to_row(x), obj.trials))
        print(pd.DataFrame.from_records(data=row_list))
