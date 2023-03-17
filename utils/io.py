import datetime
import os
import random
import hashlib
import pandas as pd

from utils.paths import dataset_config_path


def get_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Get dataset information from the csv file.
    Args:
        dataset_name:   Filename of the dataset file under "dataset/config/" (ext name excluded).
    Returns:
        A DataFrame whose columns: [instance_name, metric, change_start, change_end, label, case_id, case_label]
            instance_name:      Name of the instance where the KPI is collected from.
            metric:             Name of the KPI.
            change_start:       t_0 in our paper. 10-digit timestamp.
            change_end:         t_1 in our paper. 10-digit timestamp.
            label:              Label of this KPI. 0 (normal) / 1 (erroneous).
            case_id:            Id of the software change case to which this KPI attach.
            case_label:         Label of this software change case. 0 (normal) / 1 (erroneous).
    """

    path = os.path.join(dataset_config_path, f'{dataset_name}.csv')
    df = pd.read_csv(path)
    return df


def get_bluegreen_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Get blue-green dataset information from the csv file.
    Args:
        dataset_name:   Filename of the dataset file under "dataset/config/" (ext name excluded).
    Returns:
        A DataFrame whose columns: [service, version, old_hash, new_hash, start, end, change]
            service:            Name of the instance where the KPI is collected from.
            version:            Anomaly version of the service.
            old_hash:           The normal version hash before blue-green deployment.
            new_hash:           The new version hash after blue-green deployment.
            start:              The start datetime of the csv data.
            end:                The end datetime of the csv data.
            change:             The timepoint when blue-green deployment went on.
    """

    path = os.path.join(dataset_config_path, f'{dataset_name}.csv')
    df = pd.read_csv(path)
    df['start'] = df['start'].apply(lambda x: int(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp()))
    df['end'] = df['end'].apply(lambda x: int(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp()))
    df['change'] = df['change'].apply(lambda x: int(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp()))
    return df


def exp_name_stamp() -> str:
    """
    Get a unique experiment name based on current timestamp for saving checkpoints.
    Returns:
        str
    """

    now_time = datetime.datetime.now()
    st = now_time.strftime(f'%m%d_%H%M%S_{random.randint(0, 9999):04d}')
    return st

def get_dataset_md5(dataset_name: str) -> str:
    """
    Get the md5 of a dataset file for saving caches.
    Args:
        dataset_name:   Filename of the dataset file under "dataset/config/" (ext name excluded).
    Returns:
        str
    """

    m = hashlib.md5()
    path = os.path.join(dataset_config_path, f'{dataset_name}.csv')
    with open(path, 'rb') as fin:
        while True:
            data = fin.read(4096)
            if not data:
                break
            m.update(data)

    return m.hexdigest()
