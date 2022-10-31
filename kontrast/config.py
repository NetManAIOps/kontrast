import torch


class ModelConfig:
    """
    Kontrast model configs.
    """
    def __init__(self,
                 epoch: int,
                 input_len: int,
                 lstm_hidden_dim: int=20,
                 lstm_output_dim: int=20,
                 lr: float=0.001,
                 input_dim: int=1,
                 device: torch.device = torch.device('cpu')) -> None:
        """
        Args:
            epoch:              Training epochs.
            input_len:          Neural network input length (L).
            lstm_hidden_dim:    The hidden dimension of LSTM.
            lstm_output_dim:    The output dimension of LSTM.
            lr:                 Learning rate.
            input_dim:          The input dimension of LSTM.
            device:             Which device should be used.
        """

        self.input_dim = input_dim
        self.epoch = epoch
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_output_dim = lstm_output_dim
        self.lr = lr
        self.device = device
        self.input_len = input_len
