import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTwoTowerModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.post_tower = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        
    def forward(self, user_features, post_features):
        user_emb = F.normalize(self.user_tower(user_features), p=2, dim=1)
        post_emb = F.normalize(self.post_tower(post_features), p=2, dim=1)
        return user_emb, post_emb