import os
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
import tqdm

from kontrast.config import ModelConfig
from kontrast.nn import Siamese, ContrastiveLoss
from kontrast.dataset.dataset import DataLoader
from utils.paths import ckpt_save_path
from utils.timer import timer


class Model:
    def __init__(self,
                 config: ModelConfig):
        self.config = config
        self.model = Siamese(input_dim=self.config.input_dim,
                             lstm_hidden_dim=self.config.lstm_hidden_dim,
                             lstm_output_dim=self.config.lstm_output_dim,
                             input_len=self.config.input_len)
        self.model.to(self.config.device)

        self.loss_fn = ContrastiveLoss().to(self.config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

    @timer
    def train(self, dataloader: DataLoader, filename: str):
        """
        Train the model with a given dataloader.
        Args:
            dataloader:     Dataset dataloader.
            filename:       The dataset filename, used for save checkpoint.
        """

        n_epoch = self.config.epoch
        device = self.config.device

        model = self.model
        loss_fn = self.loss_fn
        optimizer = self.optimizer

        for epoch in range(n_epoch):
            loss_sum = 0
            for batch_x1, batch_x2, batch_y in tqdm.tqdm(dataloader):
                batch_x1 = batch_x1.to(device)
                batch_x2 = batch_x2.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                out_1, out_2 = model(batch_x1, batch_x2)
                loss = loss_fn.forward(out_1, out_2, batch_y)
                loss_sum += loss * batch_x1.shape[0]
                loss.backward()
                optimizer.step()
            print(f'epoch {epoch:3d}: loss = {loss_sum / (1e-5 + dataloader.sample_count()):.4f}')

        self.model = model
        self.dump(os.path.join(ckpt_save_path, f'{filename}.ckpt'))

    @timer
    def test(self, dataloader: DataLoader):
        """
        Test the model with a given dataloader.
        Note for each kpi id, there are multiple X^Ps (for P model).
        We need an minimum aggregation, shown in the code.
        Args:
            dataloader:     Dataset dataloader.

        Returns:
            DataFrame
        """

        device = self.config.device
        result = []

        with torch.no_grad():
            for batch_x1, batch_x2, batch_y, batch_id, batch_case_id, batch_case_label in tqdm.tqdm(dataloader):
                batch_x1 = batch_x1.to(device)
                batch_x2 = batch_x2.to(device)

                out_1, out_2 = self.model(batch_x1, batch_x2)
                loss = F.pairwise_distance(out_1, out_2, keepdim=True).cpu().numpy()
                batch_id = batch_id.numpy()
                batch_y = batch_y.numpy()
                batch_case_id = batch_case_id.numpy()
                batch_case_label = batch_case_label.numpy()
                batch_result = np.concatenate([batch_id, batch_y, batch_case_id, batch_case_label, loss], axis=-1)
                result.append(batch_result)

        if len(result) > 0:
            result = np.concatenate(result, axis=0)
            result_df = pd.DataFrame(result, columns=['id', 'label', 'case_id', 'case_label', 'dist'])
            # Minimum aggregation
            group = result_df.groupby('id').agg({
                'label': 'mean',
                'case_id': 'mean',
                'case_label': 'mean',
                'dist': 'min'})
            group.reset_index(inplace=True)

            return group
        else:
            return None


    def dump(self, filename: str):
        """
        Dump the model to a checkpoint file.
        Args:
            filename:   Checkpoint filename.
        """

        state = {
            'net': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)

    def load(self, filename: str):
        """
        Load the model from a checkpoint file.
        Args:
            filename:   Checkpoint filename.
        """

        ckpt_path = os.path.join(ckpt_save_path, f'{filename}.ckpt')
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

    def inference(self, x_1: np.ndarray, x_2: np.ndarray):
        """
        Test the model with a given KPI time series segment pair(s).
        Args:
            x_1:    Time series segment a ([L]) or Time series segment set A ([batch, L]).
            x_2:    Time series segment b ([L]) or Time series segment set B ([batch, L]).
        Returns:
            ndarray
        """

        device = self.config.device
        input_dim = self.config.input_dim
        batch_size = 1 if len(x_1.shape) == 1 else x_1.shape[0]
        x_1 = torch.Tensor(x_1).view((batch_size, -1, input_dim)).to(device)
        x_2 = torch.Tensor(x_2).view((batch_size, -1, input_dim)).to(device)
        with torch.no_grad():
            out_1, out_2 = self.model(x_1, x_2)
            d = F.pairwise_distance(out_1, out_2).item()
            return d