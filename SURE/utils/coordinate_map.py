import numpy as np
import torch
from torch import nn
from skorch import NeuralNetRegressor


class CoordinateMapping(nn.Module):
    def __init__(self, num_cor1=10, num_cor2=10, num_units=256, nonlin=nn.ReLU(), dropout_rate=0.01,
                 max_epochs=10, learning_rate=0.1, device='cuda', dtype=torch.float32, iterator_train__shuffle=True):
        super().__init__()

        self.dense0 = nn.Linear(num_cor1, num_units, device=device, dtype=dtype)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(num_units, num_units, device=device, dtype=dtype)
        self.output = nn.Linear(num_units, num_cor2, device=device, dtype=dtype)
        self.regressor = NeuralNetRegressor(self, max_epochs=max_epochs, lr=learning_rate,
                                            device=device, iterator_train__shuffle=iterator_train__shuffle)
        self.to(device)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.output(X)
        return X
    
    def fit(self, X, Y):
        self.regressor.fit(X=X,y=Y)

    def mapping(self, X):
        return self.regressor.predict(X)
