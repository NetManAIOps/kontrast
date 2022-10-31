import torch
from torch import nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, input_len):
        """
        Bidirectional LSTM module.
        Args:
            input_dim:      The input dimension.
            hidden_dim:     The hidden dimension.
            output_dim:     The output dimension.
            input_len:      Length of the input.
        """

        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_len = input_len

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * input_len * 2, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, seq):
        batch_size = seq.shape[0]
        seq = seq.view(batch_size, -1, self.input_dim)
        res, _ = self.lstm(seq)
        lstm_out = res.reshape(-1, self.hidden_dim * self.input_len * 2)
        output = self.fc(lstm_out)

        return output

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        """
        Contrastive loss function.

        S. Chopra, R. Hadsell, and Y. LeCun,
        “Learning a similarity metric discriminatively, with application to face verification,”
        in 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), vol. 1. IEEE, 2005, pp. 539–546.

        Args:
            margin:
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out_1, out_2, label):
        euclidean_distance = F.pairwise_distance(out_1, out_2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class Siamese(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_output_dim, input_len):
        super(Siamese, self).__init__()
        self.bilstm = BiLSTM(input_dim, lstm_hidden_dim, lstm_output_dim, input_len)

    def forward(self, seq_1, seq_2):
        out_1 = self.bilstm(seq_1)
        out_2 = self.bilstm(seq_2)
        return out_1, out_2
