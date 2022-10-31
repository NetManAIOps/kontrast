import os
import datetime
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta as td
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import tqdm
import pickle
import torch

from kontrast.dataset.config import DatasetConfig
from models.time_span import TimeSpan
from utils import io
from utils.paths import cache_path, dataset_path


class Dataset:
    def __init__(self,
                 dataset_name: str,
                 omega: td,
                 period: td,
                 config: DatasetConfig):
        """
        Dataset preprocessing and data generating.
        Args:
            dataset_name:       Filename of the dataset file under "dataset/config/" (ext name excluded).
            omega:              Inspection window size. omega in our paper. NOTE here we use TimeDelta.
            period:             The period of KPI time series in the dataset. T in our paper. NOTE here we use TimeDelta.
            config:             Dataset config.
        """

        self.config = config
        self.period = period
        self.omega = omega
        self.mode = config.mode
        self.ignore = config.ignore
        self.dataset_name = dataset_name

        # Read the meta data of the dataset.
        dataset_df = io.get_dataset(self.dataset_name)
        instance = list(dataset_df['instance_name'])
        metric = list(dataset_df['metric'])
        size = len(dataset_df)
        self.filepath = [os.path.join(dataset_path, self.dataset_name, f'{instance[idx]}${metric[idx]}.csv')
                         for idx in range(size)]
        self.kpi_name = [f'{instance[idx].split("@")[-1]}${metric[idx]}' for idx in range(size)]
        change_start = list(dataset_df['change_start'])
        change_end = list(dataset_df['change_end'])
        self.change_span = [TimeSpan(change_start[idx], change_end[idx]) for idx in range(size)]
        self.label = list(dataset_df['label'])
        self.case_id = list(dataset_df['case_id'])
        self.case_label = list(dataset_df['case_label'])

        # Read the raw KPIs
        data_md5 = io.get_dataset_md5(dataset_name)
        data_cache_path = os.path.join(cache_path, f'dataset/{data_md5}.pkl')
        os.makedirs(os.path.join(cache_path, 'dataset'), exist_ok=True)
        if not os.path.exists(data_cache_path):
            print('Building dataset..')
            self.data = [None] * size
            def read_csv(id: int, filename: str) -> tuple:
                if id % 100 == 0:
                    print(id)

                df = pd.read_csv(filename)
                values = list(df['value'])
                q = np.percentile(values, 95)
                if q > 0:
                    values = list(np.array(values) / q)
                values = list((np.array(values) - 0.5) * 2)
                df['value'] = values
                return (id, df)
            executor = ThreadPoolExecutor(20)
            tasks = [executor.submit(read_csv, id, f) for id, f in enumerate(self.filepath)]
            for future in as_completed(tasks):
                id, df = future.result()
                self.data[id] = df
            with open(data_cache_path, 'wb') as fout:
                pickle.dump(self.data, fout)
            print('Finish')

        else:
            print('Loading data from cache..')
            with open(data_cache_path, 'rb') as fin:
                self.data = pickle.load(fin)
            print('Loaded.')

        # Noise intensity categories, saving the ids of KPIs
        self.category = [[] for _ in range(self.config.K)]
        # Standard deviation of each KPI
        self.std = [0] * size
        kpi_std_cache = {}
        for i in tqdm.tqdm(range(size)):
            df = self.data[i]
            kpi_name = self.kpi_name[i]

            if kpi_name in kpi_std_cache:
                std_avg = kpi_std_cache[kpi_name]
            else:
                # Pre-change period only
                df = df[df['timestamp'] < self.change_span[i].start]
                df['timestamp'] = df['timestamp'] % int(self.period.total_seconds())

                df.sort_values(by='timestamp', inplace=True)
                arr = np.array(df)

                start_idx = 0
                std_list = []
                while start_idx < len(arr):
                    end_idx = start_idx
                    while end_idx + 1 < len(arr) and arr[end_idx + 1, 0] == arr[start_idx, 0]:
                        end_idx += 1
                    std = np.nanstd(arr[start_idx:end_idx+1, 1])
                    std_list.append(std)

                    start_idx = end_idx + 1
                std_avg = float(np.mean(std_list))
                kpi_std_cache[kpi_name] = std_avg

            self.std[i] = std_avg
            cate = self.config.get_category(float(std_avg))

            # This KPI belongs to category "cate", and we should add it to "cate"
            # and the categories with a higher noise intensity level for training
            # data generation
            for j in range(cate, self.config.K):
                self.category[j].append(i)

        # Start generating data
        self.generate_train_data()
        self.generate_test_data()

    def rate(self):
        """
        Interval of two successive samples
        Returns:
            int
        """

        return self.data[0].loc[1, 'timestamp'] - self.data[0].loc[0, 'timestamp']

    def _generate_random_span_P(self, idx: int) -> tuple:
        """
        Randomly select a time span in the KPI for P model.
        Args:
            idx:    The index of the KPI.
        Returns:
            (ndarray, TimeSpan):    Selection and the corresponding time span.
        """

        df = self.data[idx]
        ts_start = df.loc[0, 'timestamp']
        ts_end = int(self.change_span[idx].start - self.omega.total_seconds())
        if ts_start > ts_end:
            return None, None

        start_point = random.randint(ts_start, ts_end)
        end_point = int(start_point + self.omega.total_seconds())

        result = np.array(df[(df['timestamp'] >= start_point) & (df['timestamp'] < end_point)]['value'])
        span = TimeSpan(start_point, end_point)
        expected_len = int(self.omega.total_seconds() // self.rate())
        while len(result) < expected_len:
            result = np.append(result, result[-1])

        return result, span

    def _generate_random_span_LS(self, idx: int) -> tuple:
        """
        Randomly select a KPI time series segment pair (pre and post-change) for LS model.
        There is a randomly-lengthed ongoing period between them.
        Args:
            idx:    The index of the KPI.
        Returns:
            (ndarray, ndarray):    Selections.
        """

        change_duration = min(0.1, float(np.random.exponential(0.03))) * self.period.total_seconds()
        df = self.data[idx]
        ts_start = df.loc[0, 'timestamp']
        rate = df.loc[1, 'timestamp'] - df.loc[0, 'timestamp']
        span_len = self.omega.total_seconds() * 2 + change_duration + rate
        ts_end = int(self.change_span[idx].start - span_len)
        if ts_start > ts_end:
            return None, None

        start_point = random.randint(ts_start, ts_end)
        change_start = start_point + self.omega.total_seconds()
        end_point = int(start_point + span_len)
        change_end = end_point - self.omega.total_seconds()
        expected_len = int(self.omega.total_seconds() // self.rate())

        v1 = np.array(df[(df['timestamp'] >= start_point) & (df['timestamp'] < change_start)]['value'])
        v2 = np.array(df[(df['timestamp'] >= change_end) & (df['timestamp'] < end_point)]['value'])
        while len(v1) < expected_len:
            v1 = np.append(v1, v1[-1])
        while len(v2) < expected_len:
            v2 = np.append(v2, v2[-1])
        return v1, v2

    def _generate_random_sametime_span_P(self, idx: int, span: TimeSpan) -> tuple:
        """
        Randomly select a contemporaneous historical time series segment of a given time span for P model.
        Args:
            idx:    The index of the KPI.
            span:   The given time span.
        Returns:
            (ndarray, TimeSpan):    Selection and the corresponding time span.
        """

        df = self.data[idx]
        ts_start = df.loc[0, 'timestamp']
        ts_end = self.change_span[idx].start - self.omega.total_seconds()

        negative_count = int((span.start - ts_start) // self.period.total_seconds())
        positive_count = int((ts_end - span.start) // self.period.total_seconds())

        candidates = [_ for _ in np.arange(-negative_count, positive_count + 1, 1)]
        if len(candidates) == 0:
            return None, None

        t = random.choice(candidates)
        start_point = span.start + t * self.period.total_seconds()
        end_point = span.end + t * self.period.total_seconds()

        result = np.array(df[(df['timestamp'] >= start_point) & (df['timestamp'] < end_point)]['value'])
        span = TimeSpan(start_point, end_point)
        expected_len = int(self.omega.total_seconds() // self.rate())
        while len(result) < expected_len:
            result = np.append(result, result[-1])

        return result, span

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Euclidean-based distance between two time series segments.
        Used to filter and make sure that generated pairs are dissimilar.
        Args:
            a, b:       Time series segments.
        Returns:
            float
        """

        return float(np.percentile(np.abs(np.array(a) - np.array(b)), 80))

    @staticmethod
    def _add_noises(d: np.ndarray, intensity: float, normal: bool=True) -> np.ndarray:
        """
        Noise pattern injection.
        Args:
            d:              The original time series.
            intensity:      The noise intensity.
            normal:         Normal or abnormal. If abnormal, inject intense noises.
        Returns:
            ndarray
        """

        def level_shift(d: np.ndarray, intensity: float) -> np.ndarray:
            t = np.random.uniform(intensity / 2, intensity)
            if np.random.rand() < 0.5:
                t *= -1
            return d + t
        def relative_level_shift(d: np.ndarray, intensity: float) -> np.ndarray:
            t = np.random.uniform(intensity / 2, intensity)
            if np.random.rand() < 0.5:
                t *= -1
            t = 1 - t
            return (d + 1) * t - 1
        def gaussian_noise(d: np.ndarray, intensity: float) -> np.ndarray:
            noise = np.random.normal(0, intensity, d.shape)
            return d + noise
        def transient_noise(d: np.ndarray, intensity: float) -> np.ndarray:
            pos = np.random.randint(0, len(d))
            t = np.random.uniform(-intensity * 10, intensity * 10)
            d_ = d.copy()
            d_[pos] += t
            return d_
        def ramp(d: np.ndarray, intensity: float) -> np.ndarray:
            pos = np.random.randint(0, len(d))
            t = np.random.uniform(intensity / 2, intensity)
            if np.random.rand() < 0.5:
                t *= -1
            d_ = d.copy()

            mode = np.random.randint(0, 10)
            if mode == 0:
                d_[:pos] += t
            elif mode == 1:
                d_[pos:] += t
            else:
                pos_ = np.random.randint(0, len(d))
                if pos > pos_:
                    pos, pos_ = pos_, pos
                d_[pos:pos_] += t
            return d_
        def steady_change(d: np.ndarray, intensity: float) -> np.ndarray:
            sgn = np.random.choice((-1, 1))
            pos = np.random.randint(1, len(d) // 4 * 3)
            t = np.random.uniform(intensity / (len(d)-pos), intensity / (len(d)-pos) * 2)

            noise = np.zeros(d.shape)
            for i in range(pos, len(d)):
                noise[i] = noise[i-1] + np.random.normal(t / 2, t)
            noise *= sgn
            return d + noise

        # Modules are not exclusive. They are selected by probability.
        funcs = []
        if np.random.rand() < 0.4:
            funcs.append([level_shift, intensity])
        if np.random.rand() < 0.4:
            funcs.append([relative_level_shift, intensity])
        if np.random.rand() < 0.2:
            funcs.append([transient_noise, intensity])
        if np.random.rand() < 0.8 or len(funcs) == 0:
            funcs.append([gaussian_noise, intensity])
        if not normal:
            if np.random.rand() < 0.3:
                funcs.append([level_shift, intensity * np.random.uniform(3, 10)])
            if np.random.rand() < 0.3:
                funcs.append([relative_level_shift, intensity * np.random.uniform(3, 10)])
            if np.random.rand() < 0.3:
                funcs.append([gaussian_noise, intensity * np.random.uniform(3, 10)])
            if np.random.rand() < 0.2:
                funcs.append([ramp, intensity * np.random.uniform(3, 10)])
            if np.random.rand() < 0.2:
                funcs.append([steady_change, intensity * np.random.uniform(3, 10)])
            if not funcs:
                funcs.append([steady_change, intensity * np.random.uniform(3, 10)])
        for f, intensity in funcs:
            d = f(d, intensity)
        return d

    def generate_train_data(self):
        """
        Generate sufficient train data pairs with pseudo labels.
        """

        train_data_md5 = io.get_dataset_md5(self.dataset_name)
        train_data_cache_path = os.path.join(cache_path, f'dataset/{train_data_md5}_train_{self.mode}_size_{self.config.dataset_size}_cate_{self.config.K}_omega_{int(self.omega.total_seconds())}.pkl')
        os.makedirs(os.path.join(cache_path, 'dataset'), exist_ok=True)

        if os.path.exists(train_data_cache_path):
            print('Loading train data from cache..')
            with open(train_data_cache_path, 'rb') as fin:
                self.train_data = pickle.load(fin)
            print('Loaded.')
        else:
            print('Building train data..')
            self.train_data = [[] for _ in range(self.config.K)]

            if self.mode == 'P':
                for i in range(self.config.K):
                    cands = self.category[i]
                    if len(cands) == 0:
                        continue
                    # Negative cases
                    for _ in tqdm.tqdm(range(self.config.dataset_size)):
                        idx = random.choice(cands)
                        v1, span = self._generate_random_span_P(idx)
                        if v1 is None:
                            continue
                        if np.random.rand() < 0.7:  # The same segment with mild noises
                            intensity = self.config.std_thres[self.config.get_category(self.std[idx])]
                            v2 = self._add_noises(v1, intensity)
                        else:   # Contemporaneous historical data
                            v2, span_ = self._generate_random_sametime_span_P(idx, span)
                        self.train_data[i].append((v1, v2, 0))
                    # Positive cases
                    for _ in tqdm.tqdm(range(self.config.dataset_size)):
                        idx = random.choice(cands)
                        v1, span = self._generate_random_span_P(idx)
                        if v1 is None:
                            continue
                        intensity = self.config.std_thres[self.config.get_category(self.std[idx])]
                        if np.random.rand() < 0.7:
                            v2 = self._add_noises(v1, intensity, False)
                        else:
                            v2 = None
                            # Make sure v1 is dissimilar enough to v2
                            while v2 is None or self._distance(v1, v2) < min(intensity * 2, 0.5):
                                v2, span_ = self._generate_random_span_P(random.choice(cands))
                        self.train_data[i].append((v1, v2, 1))
            elif self.mode == 'LS':
                for i in range(self.config.K):
                    cands = self.category[i]
                    if len(cands) == 0:
                        continue
                    # Negative cases
                    for _ in tqdm.tqdm(range(self.config.dataset_size)):
                        idx = random.choice(cands)
                        v1, v2 = self._generate_random_span_LS(idx)
                        if v1 is None:
                            continue
                        intensity = self.config.std_thres[self.config.get_category(self.std[idx])]
                        if np.random.rand() < 0.4:
                            v1 = self._add_noises(v1, intensity)
                        if np.random.rand() < 0.4:
                            v2 = self._add_noises(v2, intensity)
                        self.train_data[i].append((v1, v2, 0))
                    # Positive cases
                    for _ in tqdm.tqdm(range(self.config.dataset_size)):
                        idx = random.choice(cands)
                        v1, span = self._generate_random_span_P(idx)
                        if v1 is None:
                            continue
                        intensity = self.config.std_thres[self.config.get_category(self.std[idx])]
                        if np.random.rand() < 0.7:
                            v2 = self._add_noises(v1, intensity, False)
                        else:
                            v2 = None
                            # Make sure v1 is dissimilar enough to v2
                            while v2 is None or self._distance(v1, v2) < min(intensity * 2, 0.5):
                                v2, span_ = self._generate_random_span_P(random.choice(cands))
                        self.train_data[i].append((v1, v2, 1))

            with open(train_data_cache_path, 'wb') as fout:
                pickle.dump(self.train_data, fout)
            print('Finish.')

    def generate_test_data(self):
        """
        Generate test data based on the data extraction specifications.
        """

        test_data_md5 = io.get_dataset_md5(self.dataset_name)
        test_data_cache_path = os.path.join(cache_path, f'dataset/{test_data_md5}_test_{self.mode}_{self.ignore}_cate_{self.config.K}_omega_{int(self.omega.total_seconds())}.pkl')
        os.makedirs(os.path.join(cache_path, 'dataset'), exist_ok=True)

        expected_len = int(self.omega.total_seconds() // self.rate())
        if os.path.exists(test_data_cache_path):
            print('Loading test data from cache..')
            with open(test_data_cache_path, 'rb') as fin:
                self.test_data = pickle.load(fin)
            print('Loaded.')
        else:
            print('Building test data..')
            self.test_data = [[] for _ in range(self.config.K)]
            size = len(self.data)
            for i in tqdm.tqdm(range(size)):
                data = self.data[i]
                cate = self.config.get_category(self.std[i])
                change_start = self.change_span[i].start
                change_end = self.change_span[i].end

                # Post-change time span
                aft_span = TimeSpan(change_end, int(change_end + self.omega.total_seconds()))
                aft_data = np.array(data[(data['timestamp'] >= aft_span.start) & (data['timestamp'] < aft_span.end)]['value'])
                while len(aft_data) < expected_len:
                    aft_data = np.append(aft_data, aft_data[-1])

                # Pre-change time spans
                bef_spans = []
                if self.mode == 'P':
                    # 1T, 2T, 3T, 7T, 14T, 21T
                    for _ in range(1, 3):
                        bef_spans.append([-self.period * _, -self.period * _ + self.omega])
                    for _ in range(1, 3):
                        bef_spans.append([-self.period * 7 * _, -self.period * 7 * _ + self.omega])
                elif self.mode == 'LS':
                    ignore_td = td(seconds=change_end - change_start) if self.ignore else td()
                    bef_spans.append([-self.omega - ignore_td, td() - ignore_td])

                bef_data = []
                for span in bef_spans:
                    change_end_time = datetime.datetime.fromtimestamp(change_end)
                    st = int((change_end_time + span[0]).timestamp())
                    ed = int((change_end_time + span[1]).timestamp())
                    d = np.array(data[(data['timestamp'] >= st) & (data['timestamp'] < ed)]['value'])
                    if len(d) == 0:
                        continue
                    while len(d) < expected_len:
                        d = np.append(d, d[-1])
                    bef_data.append(d)

                case_id = self.case_id[i]
                case_label = self.case_label[i]
                # Add pairs into dataset
                for d in bef_data:
                    self.test_data[cate].append((aft_data, d, self.label[i], i, case_id, case_label))

            with open(test_data_cache_path, 'wb') as fout:
                pickle.dump(self.test_data, fout)
            print('Finish.')

    def get_train_data_loader(self, cate: int):
        return DataLoader(self.train_data[cate], self.config.batch_size, 'train')

    def get_test_data_loader(self, cate: int):
        return DataLoader(self.test_data[cate], self.config.batch_size, 'test')

class DataLoader:
    def __init__(self,
                 data: list,
                 batch_size: int,
                 subset: str):
        """
        Data loader of the dataset.
        Args:
            data:
                Training set:       [(v1, v2, label)]
                Test set:           [(v1, v2, label, id, case_id, case_label)]
            batch_size:             Batch size of the data loader.
            subset:                 "train" or "test"
        """

        self.data = np.random.permutation(np.array(data, dtype=object))
        for i, d in enumerate(data):
            if not d[0].shape == d[1].shape:
                print(i)
                print(d[0].shape, d[1].shape)
                print(d)
                exit(0)
        assert subset in ['train', 'test']
        self.subset = subset
        self.batch_size = batch_size
        self.cache = []
        self.cnt = 0

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def sample_count(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        batch_x1 = []
        batch_x2 = []
        batch_y = []
        batch_id = []
        batch_case_id = []
        batch_case_label = []
        remaining = self.sample_count() - self.cnt
        idx = self.cnt // self.batch_size

        if remaining > 0:
            # Batch size of the current batch
            size = min(self.batch_size, remaining)
            if len(self.cache) > idx:
                result = self.cache[idx]
            else:
                for i in range(self.cnt, min(self.cnt + self.batch_size, self.sample_count())):
                    if self.subset == 'train':
                        data_1, data_2, label = self.data[i]
                    else:
                        data_1, data_2, label, id, case_id, case_label = self.data[i]
                        batch_id.append(id)
                        batch_case_id.append(case_id)
                        batch_case_label.append(case_label)
                    batch_x1.append(data_1)
                    batch_x2.append(data_2)
                    batch_y.append(label)

                batch_x1 = torch.Tensor(np.array(batch_x1, dtype=float)).view(size, -1, 1)
                batch_x2 = torch.Tensor(np.array(batch_x2, dtype=float)).view(size, -1, 1)
                batch_y = torch.Tensor(np.array(batch_y, dtype=float)).view(size, -1)
                result = [batch_x1, batch_x2, batch_y]
                if self.subset == 'test':
                    batch_id = torch.Tensor(np.array(batch_id, dtype=int)).view(size, -1)
                    batch_case_id = torch.Tensor(np.array(batch_case_id, dtype=int)).view(size, -1)
                    batch_case_label = torch.Tensor(np.array(batch_case_label, dtype=int)).view(size, -1)
                    result.extend([batch_id, batch_case_id, batch_case_label])

                self.cache.append(tuple(result))
            self.cnt += size
            return result
        else:
            self.cnt = 0
            self.data = np.random.permutation(self.data)
            raise StopIteration