import datetime
from datetime import timedelta as td
import os
import pandas as pd
import torch
import yaml
import numpy as np
import tqdm
from dtw import accelerated_dtw

from kontrast.config import ModelConfig
from kontrast.model import Model
from kontrast.dataset.config import DatasetConfig
from kontrast.dataset.dataset import Dataset, BlueGreenDataset
from models.time_span import TimeSpan
from utils import io
from utils.paths import exp_info_save_path

def get_time_span(origin: int, deltas: list) -> TimeSpan:
    """
    Apply a time span offset to the origin.
    Args:
        origin:     The origin.
        deltas:     Offsets: [start_delta, end_delta].
    Returns:
        TimeSpan:   Shifted time span.
    """

    origin = datetime.datetime.fromtimestamp(origin)
    return TimeSpan(int((origin + deltas[0]).timestamp()), int((origin + deltas[1]).timestamp()))

class BaseExperiment:
    def __init__(self):
        self.name_stamp = io.exp_name_stamp()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}_{self.name_stamp}'

    def save_yaml(self):
        pass

    def train(self):
        pass

    def test(self) -> pd.DataFrame:
        pass


class Experiment(BaseExperiment):
    def __init__(self,
                 omega: td,
                 period: td,
                 dataset_name: str,
                 mode: str,
                 K: int,
                 epoch: int,
                 lstm_hidden_dim: int,
                 lstm_output_dim: int,
                 lr: float,
                 from_ckpt: str=None,
                 batch_size: int=5000,
                 dataset_size: int=10000,
                 rate: int=60,
                 **kwargs):
        """
        Configs of an experiment.
        Args:
            omega:              Inspection window size. omega in our paper. NOTE here we use TimeDelta.
            period:             The period of KPI time series in the dataset. T in our paper. NOTE here we use TimeDelta.
            dataset_name:       Filename of the dataset file under "dataset/config/" (ext name excluded).
            mode:               "LS" or "P", indicating this is an LS model or a P model.
            K:                  Number of noise intensity levels. K in our paper.
            epoch:              Epochs to train.
            lstm_hidden_dim:    The hidden dimension of LSTM.
            lstm_output_dim:    The output dimension of LSTM.
            lr:                 Learning rate.
            from_ckpt:          Whether to start from a checkpoint (name_stamp of another experiment). If so, we only test the model.
            batch_size:         Batch size of the dataset.
            dataset_size:       Number of KPI time series segment pairs in the dataset to generate per label (positive/negative).
            rate:               Sampling interval of the dataset (unit: second). For example, if the monitor service collects data once per minute, rate=60.
            ignore:             Whether to ignore the ongoing period in LS model, default True.
            device:             Which device should this experiment be conducted on.
        """
        super().__init__()

        # The list of distance names.

        self.omega = omega
        self.period = period
        assert mode in ['LS', 'P']
        self.mode = mode
        self.dataset_name = dataset_name
        self.K = K
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_output_dim = lstm_output_dim
        self.epoch = epoch
        self.lr = lr
        self.ignore = kwargs['ignore'] if mode == 'LS' else True
        device = kwargs['device'] if 'device' in kwargs else 'cuda:0'
        self.dataset_size = dataset_size
        self.from_ckpt = from_ckpt
        self.save_yaml()

        self.batch_size = batch_size
        self.dataset = None     # imported later when train() is called

        self.models = []
        for i in range(self.K):
            self.models.append(Model(
                config=ModelConfig(
                    input_dim=1,
                    epoch=self.epoch,
                    lstm_hidden_dim=self.lstm_hidden_dim,
                    lstm_output_dim=self.lstm_output_dim,
                    lr=self.lr,
                    device=torch.device(device),
                    input_len=int(self.omega.total_seconds() // rate)
                )))


    def save_yaml(self):
        """
        Save experiment setup to an yaml file.
        """

        os.makedirs(exp_info_save_path, exist_ok=True)
        filename = os.path.join(exp_info_save_path, f'{self.__repr__()}.yaml')

        attr_dict = vars(self).copy()
        attr_dict.pop('aft_span', None)
        attr_dict.pop('bef_spans', None)
        attr_dict.pop('period', None)
        attr_dict['omega'] = str(attr_dict['omega'])
        attr_dict['type'] = type(self).__name__

        with open(filename, 'w') as fout:
            yaml.dump(attr_dict, fout)

    def train(self) -> None:
        """
        Train the model.
        """

        self.dataset = Dataset(dataset_name=self.dataset_name,
                               omega=self.omega,
                               period=self.period,
                               config=DatasetConfig(K=self.K, mode=self.mode, ignore=self.ignore,
                                                    batch_size=self.batch_size, dataset_size=self.dataset_size))
        if self.from_ckpt is None:
            print('Start training..')
            for i, m in enumerate(self.models):
                m.train(self.dataset.get_train_data_loader(i), f'{self.name_stamp}_{i}')
            print('Training complete.')
        else:
            print('Start loading..')
            for i, m in enumerate(self.models):
                m.load(f'{self.from_ckpt}_{i}')
            print('Loading complete.')

    def test(self) -> pd.DataFrame:
        """
        Test the model.
        Returns:
            DataFrame
        """

        result = None
        for i, m in enumerate(self.models):
            cate_result = m.test(self.dataset.get_test_data_loader(i))
            if not cate_result is None:
                if result is None:
                    result = cate_result
                else:
                    result = pd.concat([result, cate_result])
        return result


class BlueGreenExperiment(BaseExperiment):
    def __init__(self,
                 omega: td,
                 dataset_name: str,
                 epoch: int,
                 lstm_hidden_dim: int,
                 lstm_output_dim: int,
                 lr: float,
                 from_ckpt: str=None,
                 batch_size: int=5000,
                 dataset_size: int=10000,
                 intensity: float=0.1,
                 rate: int=15,
                 **kwargs):
        super().__init__()

        self.omega = omega
        self.dataset_name = dataset_name
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_output_dim = lstm_output_dim
        self.epoch = epoch
        self.lr = lr
        self.intensity = intensity
        device = kwargs['device'] if 'device' in kwargs else 'cuda:0'
        self.dataset_size = dataset_size
        self.from_ckpt = from_ckpt
        self.save_yaml()

        self.batch_size = batch_size
        self.dataset = BlueGreenDataset(
            dataset_name=self.dataset_name,
            omega=self.omega,
            config=DatasetConfig(
                K=1, mode='BG',
                batch_size=self.batch_size,
                dataset_size=self.dataset_size
            ),
            intensity=self.intensity
        )

        self.model = Model(
            config=ModelConfig(
                input_dim=1,
                epoch=self.epoch,
                lstm_hidden_dim=self.lstm_hidden_dim,
                lstm_output_dim=self.lstm_output_dim,
                lr=self.lr,
                device=torch.device(device),
                input_len=int(self.omega.total_seconds() // rate)
            ))

    def __repr__(self):
        return f'{self.__class__.__name__}_{self.name_stamp}'

    def train(self):
        """
        Train the model.
        """

        if self.from_ckpt is None:
            print('Start training..')
            self.model.train(self.dataset.get_train_data_loader(), f'{self.name_stamp}')
            print('Training complete.')
        else:
            print('Start loading..')
            self.model.load(f'{self.from_ckpt}')
            print('Loading complete.')

    def test(self) -> pd.DataFrame:
        """
        Test the model.
        Returns:
            DataFrame
        """

        return self.model.test(self.dataset.get_test_data_loader())

    def save_yaml(self):
        """
        Save experiment setup to an yaml file.
        """

        os.makedirs(exp_info_save_path, exist_ok=True)
        filename = os.path.join(exp_info_save_path, f'{self.__repr__()}.yaml')

        attr_dict = vars(self).copy()
        attr_dict['omega'] = str(attr_dict['omega'])
        attr_dict['type'] = type(self).__name__

        with open(filename, 'w') as fout:
            yaml.dump(attr_dict, fout)


class DTWExperiment(BaseExperiment):
    def __init__(self,
                 omega: td,
                 dataset_name: str,
                 batch_size: int=5000,
                 dataset_size: int=10000,
                 ):
        super().__init__()
        self.dataset_name = dataset_name
        self.omega = omega
        self.dataset_size = dataset_size
        self.save_yaml()

        self.batch_size = batch_size
        self.dataset = BlueGreenDataset(
            dataset_name=self.dataset_name,
            omega=self.omega,
            config=DatasetConfig(
                K=1, mode='BG',
                batch_size=self.batch_size,
                dataset_size=self.dataset_size
            ),
            intensity=0
        )

    def train(self):
        pass

    @staticmethod
    def _dtw_dist(a: np.ndarray, b: np.ndarray) -> float:
        manhattan_distance = lambda x, y: np.abs(x - y)
        d, _, __, ___ = accelerated_dtw(a, b, dist=manhattan_distance)
        l = min(len(a), len(b))
        return d / l

    def compare(self,
                cur_values: list,
                base_values: list) -> float:
        assert len(cur_values) == len(base_values)

        return self._dtw_dist(np.array(cur_values), np.array(base_values))

    def test(self) -> pd.DataFrame:
        dataloader = self.dataset.get_test_data_loader()

        result = []
        for batch_x1, batch_x2, batch_y, batch_id, batch_case_id, batch_case_label in dataloader:
            batch_x1 = batch_x1.squeeze().numpy()
            batch_x2 = batch_x2.squeeze().numpy()
            losses = []
            for i in tqdm.tqdm(range(len(batch_x1))):
                losses.append(self._dtw_dist(batch_x1[i], batch_x2[i]))
            batch_y = batch_y.numpy()
            batch_case_id = batch_case_id.numpy()
            batch_case_label = batch_case_label.numpy()
            batch_id = batch_id.numpy()
            losses = np.array(losses)[:, np.newaxis]

            batch_result = np.concatenate([batch_id, batch_y, batch_case_id, batch_case_label, losses], axis=-1)
            result.append(batch_result)

        if len(result) > 0:
            result = np.concatenate(result, axis=0)
            result_df = pd.DataFrame(result, columns=['id', 'label', 'case_id', 'case_label', 'dist'])
            return result_df
        else:
            return None
