import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Configure
        input_shape = 63
        self.hidden_size = 64
        self.num_lstm_layers = 2

        # Choose between MLP, RNN and CNN.
        self.lstm = nn.LSTM(input_size=input_shape,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_lstm_layers,
                            batch_first=True,
                            )
                            
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=3)

    def forward(self, x):
        h0 = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_size)
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

