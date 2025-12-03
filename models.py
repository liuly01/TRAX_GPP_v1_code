import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="gpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, input_seq):
        # input_seq: [batch, seq_len, input_size]
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]

        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        output, (h, c) = self.lstm(input_seq, (h_0, c_0))
        # output: [batch, seq_len, hidden_size]
        # h: [num_layers, batch, hidden_size]
        return output, h


class LSTMMain(nn.Module):
    def __init__(
        self,
        input_size,
        output_len,
        lstm_hidden,
        num_layers,
        batch_size,
        p,
        device="cpu",
    ):
        super(LSTMMain, self).__init__()
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(num_layers)

        self.lstmunit = LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_size=batch_size,
            device=device,
        )

        self.linear = nn.Linear(self.lstm_hidden, output_len)
        self.dropout = nn.Dropout(p=p)

    def forward(self, input_seq):
        # input_seq: [batch, seq_len, input_size]
        ula, h_out = self.lstmunit(input_seq)

        # reshape LSTM outputs to apply linear layer on all time steps
        out = ula.contiguous().view(ula.shape[0] * ula.shape[1], self.lstm_hidden)
        out = self.dropout(out)
        out = self.linear(out)

        # reshape back to [batch, seq_len, output_len]
        out = out.view(ula.shape[0], ula.shape[1], -1)

        # use last time step as final prediction: [batch, output_len]
        out = out[:, -1, :]
        return out
