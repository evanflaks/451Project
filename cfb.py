import torch
import torch.nn as nn

class FootballNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(FootballNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        return self.network(x)



class MultiObjectiveLoss(nn.Module):
    def __init__(self, alpha=0.6):
        super().__init__()
        self.alpha = alpha

    def forward(self, preds, targets):
        mse_loss = ((preds - targets) ** 2).mean()
        score_diff_pred = preds[:, 0] - preds[:, 1]
        score_diff_true = targets[:, 0] - targets[:, 1]
        diff_loss = ((score_diff_pred - score_diff_true) ** 2).mean()
        return self.alpha * mse_loss + (1 - self.alpha) * diff_loss

