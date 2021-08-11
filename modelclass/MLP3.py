import torch
import torch.nn.functional as F
import torch.nn as nn

print('here new model')

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.drop = args.drop

        self.MLP3 = nn.Sequential(
            nn.Linear(306, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=args.drop),

            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=args.drop),

            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=args.drop),

            nn.Linear(1000, 360),

            nn.Sigmoid()

        )

    def forward(self, x):
        y_pred = self.MLP3(x[:,:306]) # only the audio parts

        return y_pred

