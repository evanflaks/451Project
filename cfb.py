import torch
import torch.nn as nn
    
class FootballNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),     # e.g. 256 if hidden_size=128
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Dropout(dropout),

            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size // 2),   # e.g. 64 if hidden_size=128
            nn.ReLU(),

            nn.Linear(hidden_size // 2, 2)              # output layer for [home_score, away_score]
)

    def forward(self, x):
        return self.network(x)

    

class FootballNetSplit(nn.Module):
    def __init__(self, input_dim_home, input_dim_away, hidden=128, dropout=0.2):
        super().__init__()
        self.home_net = nn.Sequential(
            nn.Linear(input_dim_home, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.away_net = nn.Sequential(
            nn.Linear(input_dim_away, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.merge = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x):
        home_x = x[:, :self.home_net[0].in_features]
        away_x = x[:, self.home_net[0].in_features:]
        home_out = self.home_net(home_x)
        away_out = self.away_net(away_x)
        combined = torch.cat([home_out, away_out], dim=1)
        return self.merge(combined)



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


